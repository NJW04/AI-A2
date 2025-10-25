# train.py
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.transforms import compute_class_weights, to_tensor
from data.wine_white import (
    CLASS_NAMES,
    load_or_fetch_wine_white,
    make_splits,
    standardize,
)
from models.mlp import MLPClassifier
from utils.io import default_artifacts_dir, ensure_dir, save_json
from utils.log import CSVLogger
from utils.metrics import compute_metrics
from utils.seed import set_seed


@dataclass
class TrainConfig:
    hidden_sizes: List[int] = None
    dropout: float = 0.2
    batchnorm: bool = False
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    class_weights: bool = False
    seed: int = 42
    data_dir: str | None = None

    def finalize(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256, 256]


def _prepare_datasets(
    data_dir: str | None, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    df, meta = load_or_fetch_wine_white(data_dir=data_dir)
    feature_names = meta["feature_names"]
    splits = make_splits(df, seed=seed, data_dir=data_dir)

    X = df[feature_names].values
    y = df["label_idx"].values

    X_train = X[splits["train_idx"]]
    y_train = y[splits["train_idx"]]
    X_val = X[splits["val_idx"]]
    y_val = y[splits["val_idx"]]
    X_test = X[splits["test_idx"]]
    y_test = y[splits["test_idx"]]

    X_train_s, X_val_s, X_test_s, scaler = standardize(
        X_train, X_val, X_test, cache_dir=data_dir
    )
    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, meta


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
):
    Xtr_t, ytr_t = to_tensor(X_train, y_train)
    Xva_t, yva_t = to_tensor(X_val, y_val)
    ds_tr = TensorDataset(Xtr_t, ytr_t)
    ds_va = TensorDataset(Xva_t, yva_t)
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, generator=g
    )
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            labels.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return compute_metrics(y_true, y_pred)


def run_training(config: TrainConfig, artifacts_dir: Path | None = None) -> Tuple[Dict, Path]:
    config.finalize()
    set_seed(config.seed)

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, meta = _prepare_datasets(
        data_dir=config.data_dir, seed=config.seed
    )

    # Prepare artifacts dir
    run_dir = artifacts_dir or default_artifacts_dir(
        "wine_white", tag=f"mlp_seed{config.seed}"
    )
    ensure_dir(run_dir)
    # Copy scaler.pkl from cache to run_dir (if exists)
    cache_dir = Path(config.data_dir) if config.data_dir else Path(".cache/wine_white")
    scaler_path = cache_dir / "scaler.pkl"
    if scaler_path.exists():
        shutil.copy2(scaler_path, run_dir / "scaler.pkl")

    # Build loaders
    train_loader, val_loader = _make_loaders(
        X_train, y_train, X_val, y_val, batch_size=config.batch_size
    )

    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    num_classes = len(CLASS_NAMES)
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
    ).to(device)

    # Loss (optional class weights)
    class_weights = None
    if config.class_weights:
        class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Loggers
    train_logger = CSVLogger(run_dir / "train_log.csv", ["epoch", "loss"])
    val_logger = CSVLogger(run_dir / "val_log.csv", ["epoch", "accuracy", "macro_f1"])

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_logger.log({"epoch": epoch, "loss": epoch_loss})

        # Validation
        val_metrics = _evaluate(model, val_loader, device=device)
        val_logger.log({"epoch": epoch, **val_metrics})

        # Early stopping on macro-F1 (validation)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best.pt")
            # Save best.json
            best_payload = {
                "model": "mlp",
                "hidden_sizes": config.hidden_sizes,
                "dropout": config.dropout,
                "batchnorm": config.batchnorm,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "patience": config.patience,
                "class_weights": config.class_weights,
                "seed": config.seed,
                "val_metrics": val_metrics,
                "best_epoch": best_epoch,
                "class_names": CLASS_NAMES,
            }
            save_json(best_payload, run_dir / "best.json")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"[epoch {epoch:03d}] loss={epoch_loss:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f} "
            f"(best={best_val_f1:.4f} @ {best_epoch})"
        )
        if epochs_no_improve >= config.patience:
            print(f"Early stopping after {epoch} epochs (no improvement for {config.patience}).")
            break

    train_logger.close()
    val_logger.close()

    # Ensure best.pt exists
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, run_dir / "best.pt")

    # Return best validation metrics
    best_json = json.loads((run_dir / "best.json").read_text())
    return best_json["val_metrics"], run_dir


def parse_hidden_sizes(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP on Wine White (5-class) with early stopping on val macro-F1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hidden-sizes", type=str, default="256,256,256")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Where to store artifacts (default: artifacts/wine_white/<timestamp>_mlp_seed<seed>)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing winequality-white.csv (defaults to ./.cache/wine_white/)",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=args.class_weights,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    metrics, run_dir = run_training(cfg, artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None)
    print("[train] Best validation metrics:", metrics)
    print(f"[train] Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
