# eval.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.wine_white import CLASS_NAMES, load_or_fetch_wine_white, make_splits, standardize
from models.mlp import MLPClassifier
from utils.io import ensure_dir, read_json, save_json
from utils.metrics import compute_metrics, plot_confusion
from utils.seed import set_seed


def _load_scaler(artifacts_dir: Path) -> object:
    p = artifacts_dir / "scaler.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"Expected scaler at {p}. Ensure you trained the model and copied scaler.pkl."
        )
    return joblib.load(p)


def _prepare_test(
    artifacts_dir: Path, data_dir: str | None, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    df, meta = load_or_fetch_wine_white(data_dir=data_dir)
    feature_names = meta["feature_names"]
    splits = make_splits(df, seed=seed, data_dir=data_dir)

    X = df[feature_names].values
    y = df["label_idx"].values

    X_test = X[splits["test_idx"]]
    y_test = y[splits["test_idx"]]

    # Load scaler from artifacts (do not refit)
    scaler = _load_scaler(artifacts_dir)
    X_test_s = scaler.transform(X_test).astype("float32")
    return X_test_s, y_test, meta


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MLP checkpoint on Wine White TEST split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Artifacts directory containing scaler.pkl and best.json (defaults to checkpoint parent).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing winequality-white.csv (defaults to ./.cache/wine_white/)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    run_dir = Path(args.artifacts_dir) if args.artifacts_dir else checkpoint_path.parent
    ensure_dir(run_dir)

    # Load config from best.json
    best_json_path = run_dir / "best.json"
    if not best_json_path.exists():
        raise FileNotFoundError(f"Missing {best_json_path}.")
    best_cfg = read_json(best_json_path)

    # Prepare data (test only)
    X_test, y_test, meta = _prepare_test(run_dir, args.data_dir, args.seed)

    device = torch.device("cpu")
    input_dim = X_test.shape[1]
    num_classes = len(CLASS_NAMES)
    hidden_sizes = best_cfg.get("hidden_sizes", [256, 256, 256])
    dropout = float(best_cfg.get("dropout", 0.2))
    batchnorm = bool(best_cfg.get("batchnorm", False))

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batchnorm=batchnorm,
    ).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Predict on test
    Xte_t = torch.from_numpy(X_test.astype("float32"))
    dtest = TensorDataset(Xte_t, torch.from_numpy(y_test.astype("int64")))
    loader = DataLoader(dtest, batch_size=128, shuffle=False)

    preds, probs, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(np.argmax(p, axis=1))
            probs.append(p)
            labels.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    prob_arr = np.concatenate(probs)

    # Save predictions/probabilities
    out_preds = run_dir / "predictions.csv"
    out_probs = run_dir / "probs_5cols.csv"
    np.savetxt(out_preds, np.c_[np.arange(len(y_true)), y_true, y_pred], delimiter=",", fmt="%d", header="index,y_true,y_pred", comments="")
    np.savetxt(out_probs, prob_arr, delimiter=",", fmt="%.6f", header="very_low,low,medium,high,very_high", comments="")

    # Metrics + confusion matrix
    metrics = compute_metrics(y_true, y_pred)
    save_json(metrics, run_dir / "metrics_test.json")
    plot_confusion(y_true, y_pred, labels=CLASS_NAMES, out_path=run_dir / "confmat_test.png")

    print("[eval] Test metrics:", metrics)
    print(f"[eval] Saved {out_preds}, {out_probs}, and confmat_test.png at: {run_dir}")


if __name__ == "__main__":
    main()
