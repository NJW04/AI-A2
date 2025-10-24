#!/usr/bin/env python3
"""
Train an MLP on Breast Cancer (default) or Dry Bean with early stopping (val macro-F1).

Artifacts are saved under: artifacts/<DATASET>/<DATESTAMP>/
- best.pt (state_dict)
- best.json (hyperparams + best val metrics)
- meta.json (feature & class names)
- scaler.pkl (for tabular datasets)
- train_log.csv / val_log.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.transforms import compute_class_weights, to_torch_tensors
from models.mlp import build_mlp
from utils.io import project_paths, write_json
from utils.log import append_csv
from utils.metrics import compute_metrics
from utils.seed import set_seed

# Dataset backends
from data.breast_cancer import (
    BreastCancerMeta,
    load_or_fetch_breast_cancer as bc_load,
    make_splits as bc_splits,
    save_scaler as bc_save_scaler,
    standardize as bc_standardize,
)
from data.drybean import (
    DryBeanMeta,
    load_or_download_drybean as db_load,
    make_splits as db_splits,
    save_scaler as db_save_scaler,
    standardize as db_standardize,
)


def _parse_hidden_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _epoch_step(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    device = next(model.parameters()).device
    running_loss = 0.0
    n_total = 0
    ys, yps = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * yb.size(0)
        n_total += yb.size(0)
        yhat = torch.argmax(logits, dim=1)
        ys.append(yb.detach().cpu().numpy())
        yps.append(yhat.detach().cpu().numpy())

    mean_loss = running_loss / max(1, n_total)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(yps) if yps else np.array([], dtype=np.int64)
    return mean_loss, y_true, y_pred


def _create_artifacts_dir(dataset: str, user_dir: str | None) -> Path:
    base = project_paths()["artifacts"] / dataset
    if user_dir:
        adir = Path(user_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        adir = base / stamp
    adir.mkdir(parents=True, exist_ok=True)
    return adir


def _build_loaders_for_dataset(
    dataset: str, batch_size: int, seed: int
):
    """
    Returns loaders, meta, csv_path, scaler_cache_path for the chosen dataset.
    """
    if dataset == "breast_cancer":
        csv = bc_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, meta = bc_splits(csv, seed=seed)
        X_tr, X_val, X_te, scaler = bc_standardize(X_tr, X_val, X_te)
        scaler_path = bc_save_scaler(scaler, csv, seed=seed, val_size=0.15, test_size=0.15)
    elif dataset == "drybean":
        csv = db_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, meta = db_splits(csv, seed=seed)
        X_tr, X_val, X_te, scaler = db_standardize(X_tr, X_val, X_te)
        scaler_path = db_save_scaler(scaler, csv, seed=seed, val_size=0.15, test_size=0.15)
    else:
        raise ValueError("Unsupported dataset. Choose from {breast_cancer,drybean}.")

    X_tr_t, y_tr_t = to_torch_tensors(X_tr, y_tr)
    X_val_t, y_val_t = to_torch_tensors(X_val, y_val)
    X_te_t, y_te_t = to_torch_tensors(X_te, y_te)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_te_t, y_te_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, meta, Path(csv), Path(scaler_path)


def train_main():
    parser = argparse.ArgumentParser(
        description="Train an MLP on Breast Cancer (default) or Dry Bean with early stopping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["breast_cancer", "drybean"], default="breast_cancer",
                        help="Dataset to train on.")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay.")
    parser.add_argument("--hidden-sizes", type=str, default="256,256,256",
                        help="Comma-separated hidden sizes for the MLP.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the MLP.")
    parser.add_argument("--batchnorm", action="store_true", default=True, help="Use BatchNorm in MLP.")
    parser.add_argument("--no-batchnorm", action="store_false", dest="batchnorm", help="Disable BatchNorm.")
    parser.add_argument("--class-weights", action="store_true",
                        help="Use class weights in CrossEntropy.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Where to save artifacts (default: artifacts/<DATASET>/<DATESTAMP>).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = _create_artifacts_dir(args.dataset, args.artifacts_dir)

    # Data
    train_loader, val_loader, test_loader, meta, csv_path, scaler_cache_path = _build_loaders_for_dataset(
        args.dataset, args.batch_size, args.seed
    )
    input_dim = len(meta.feature_names)
    # For breast_cancer: 2 classes; for drybean: 7 classes
    if hasattr(meta, "class_names"):
        num_classes = len(meta.class_names)  # type: ignore[attr-defined]
        class_names = meta.class_names       # type: ignore[attr-defined]
    else:
        num_classes = int(train_loader.dataset.tensors[1].max().item() + 1)  # type: ignore[attr-defined]
        class_names = [str(i) for i in range(num_classes)]

    hidden = _parse_hidden_sizes(args.hidden_sizes)
    model = build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=hidden,
        dropout=args.dropout,
        batchnorm=bool(args.batchnorm),
    ).to(device)

    # Optional class weights
    y_train = train_loader.dataset.tensors[1].numpy()  # type: ignore[attr-defined]
    weight_tensor = None
    if args.class_weights:
        cw = compute_class_weights(y_train)
        weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logs
    train_log = artifacts / "train_log.csv"
    val_log = artifacts / "val_log.csv"

    best_f1 = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_yp = _epoch_step(model, train_loader, criterion, optimizer)
        tr_metrics = compute_metrics(tr_y, tr_yp)
        append_csv(
            train_log,
            {"epoch": epoch, "loss": tr_loss, **tr_metrics},
            fieldnames_ordered=["epoch", "loss", "accuracy", "macro_f1", "micro_f1"],
        )

        with torch.no_grad():
            va_loss, va_y, va_yp = _epoch_step(model, val_loader, criterion, optimizer=None)
            va_metrics = compute_metrics(va_y, va_yp)
            append_csv(
                val_log,
                {"epoch": epoch, "loss": va_loss, **va_metrics},
                fieldnames_ordered=["epoch", "loss", "accuracy", "macro_f1", "micro_f1"],
            )

        print(f"[epoch {epoch:03d}] "
              f"train loss={tr_loss:.4f} macroF1={tr_metrics['macro_f1']:.4f} | "
              f"val loss={va_loss:.4f} macroF1={va_metrics['macro_f1']:.4f}")

        # Early stopping on val macro-F1
        if va_metrics["macro_f1"] > best_f1:
            best_f1 = va_metrics["macro_f1"]
            torch.save(model.state_dict(), artifacts / "best.pt")
            write_json(
                artifacts / "best.json",
                {
                    "epoch": epoch,
                    "val": va_metrics,
                    "config": {
                        "dataset": args.dataset,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "hidden_sizes": hidden,
                        "dropout": args.dropout,
                        "batchnorm": bool(args.batchnorm),
                        "class_weights": bool(args.class_weights),
                        "patience": args.patience,
                        "seed": args.seed,
                    },
                    "artifacts_dir": str(artifacts),
                },
            )
            # Save meta + scaler for evaluation
            write_json(artifacts / "meta.json", {"feature_names": getattr(meta, "feature_names", []),
                                                 "class_names": getattr(meta, "class_names", [])})
            try:
                sp = Path(scaler_cache_path)
                if sp.exists():
                    joblib.dump(joblib.load(sp), artifacts / "scaler.pkl")
            except Exception as e:
                print(f"[warn] could not copy scaler: {e}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs.")
                break

    print(f"[train] Best val macro-F1: {best_f1:.4f}. Checkpoint at {artifacts/'best.pt'}")
    print(f"[artifacts] Directory: {artifacts}")


if __name__ == "__main__":
    train_main()
