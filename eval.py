#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the TEST split only.

Outputs to the provided artifacts directory:
- metrics_test.json (accuracy, macro_f1, micro_f1, roc_auc)
- confmat_test.png
- predictions.csv (index,y_true,y_pred,y_prob_pos)

Usage:
  python eval.py --dataset breast_cancer --checkpoint artifacts/breast_cancer/<RUN>/best.pt --artifacts-dir artifacts/breast_cancer/<RUN>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data.breast_cancer import load_or_fetch_breast_cancer as bc_load, make_splits as bc_splits
from data.drybean import load_or_download_drybean as db_load, make_splits as db_splits
from data.transforms import to_torch_tensors
from models.mlp import build_mlp
from utils.io import read_json, write_json
from utils.metrics import compute_metrics, plot_confusion
from utils.seed import set_seed


def _load_test_for_dataset(dataset: str, seed: int, artifacts_dir: Path):
    """Return (test_dataset, y_true, class_names, config, input_dim, num_classes)."""
    best = read_json(artifacts_dir / "best.json", default={})
    meta = read_json(artifacts_dir / "meta.json", default={})
    class_names = meta.get("class_names", None)
    config = best.get("config", {})
    if "seed" in config:
        seed = int(config["seed"])

    # Load raw splits
    if dataset == "breast_cancer":
        csv = bc_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, _ = bc_splits(csv, seed=seed)
    elif dataset == "drybean":
        csv = db_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, _ = db_splits(csv, seed=seed)
    else:
        raise ValueError("Unsupported dataset.")

    # Transform test with the saved scaler (fitted on train)
    scaler_path = artifacts_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Did you run train.py?")
    scaler = joblib.load(scaler_path)
    X_te = scaler.transform(X_te)

    X_te_t, y_te_t = to_torch_tensors(X_te, y_te)
    test_ds = TensorDataset(X_te_t, y_te_t)

    input_dim = X_te.shape[1]
    if class_names:
        num_classes = len(class_names)
    else:
        num_classes = int(y_te.max() + 1)
    return test_ds, y_te, class_names, config, input_dim, num_classes


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MLP on the TEST set only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["breast_cancer", "drybean"], default="breast_cancer",
                        help="Dataset to evaluate on.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Artifacts directory from training.")
    parser.add_argument("--batch-size", type=int, default=256, help="Eval batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to create the splits (fallback).")
    args = parser.parse_args()

    set_seed(args.seed)
    artifacts_dir = Path(args.artifacts_dir)
    ckpt = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds, y_true, class_names, config, input_dim, num_classes = _load_test_for_dataset(
        args.dataset, args.seed, artifacts_dir
    )

    # Build model skeleton from training config
    hidden = config.get("hidden_sizes", [256, 256, 256])
    if isinstance(hidden, str):
        hidden = [int(x) for x in hidden.split(",")]
    dropout = float(config.get("dropout", 0.2))
    batchnorm = bool(config.get("batchnorm", True))

    model = build_mlp(input_dim, num_classes, hidden, dropout, batchnorm).to(device)

    # Load weights
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Eval loop
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ys, yps, prob_pos = [], [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = softmax(logits).cpu().numpy()
            yp = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            yps.append(yp)
            # For binary problems, store prob of positive class (index 1)
            if probs.shape[1] >= 2:
                prob_pos.extend(probs[:, 1].tolist())
            else:
                prob_pos.extend([np.nan] * probs.shape[0])

    y_true_np = np.concatenate(ys)
    y_pred_np = np.concatenate(yps)
    metrics = compute_metrics(y_true_np, y_pred_np)

    # ROC-AUC: binary → positive class index 1; multiclass → macro OVR
    roc_auc = None
    if num_classes == 2 and len(prob_pos) == len(y_true_np):
        try:
            roc_auc = float(roc_auc_score(y_true_np, np.array(prob_pos)))
        except Exception:
            roc_auc = None
    else:
        # Optional: macro AUC OVR if probabilities are available
        # (we did not retain full prob matrix; compute not available here for multiclass)
        roc_auc = None

    metrics["roc_auc"] = roc_auc
    write_json(artifacts_dir / "metrics_test.json", metrics)
    print("[eval] TEST metrics:", metrics)

    # Confusion matrix and predictions
    labels = class_names if class_names else [str(i) for i in range(num_classes)]
    plot_confusion(y_true_np, y_pred_np, labels, artifacts_dir / "confmat_test.png")

    pred_out = artifacts_dir / "predictions.csv"
    with pred_out.open("w") as f:
        f.write("index,y_true,y_pred,y_prob_pos\n")
        for i, (yt, yp) in enumerate(zip(y_true_np.tolist(), y_pred_np.tolist())):
            ppos = prob_pos[i] if i < len(prob_pos) else ""
            f.write(f"{i},{yt},{yp},{ppos}\n")
    print(f"[eval] Wrote predictions to {pred_out}")


if __name__ == "__main__":
    main()
