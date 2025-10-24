#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the TEST split only (Breast Cancer).

Outputs to the provided artifacts directory:
- metrics_test.json (accuracy, macro_f1, micro_f1, roc_auc)
- confmat_test.png
- predictions.csv (index,y_true,y_pred,y_prob_pos)

Usage:
  python eval.py --checkpoint artifacts/breast_cancer/<RUN>/best.pt --artifacts-dir artifacts/breast_cancer/<RUN>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data.breast_cancer import load_breast_cancer_csv, make_splits
from data.transforms import to_torch_tensors
from models.mlp import build_mlp
from utils.io import read_json, write_json
from utils.metrics import compute_metrics, plot_confusion
from utils.seed import set_seed


def _load_test(artifacts_dir: Path, seed: int):
    best = read_json(artifacts_dir / "best.json", default={})
    meta = read_json(artifacts_dir / "meta.json", default={})
    class_names = meta.get("class_names", ["benign", "malignant"])
    config = best.get("config", {})
    if "seed" in config:
        seed = int(config["seed"])

    csv = load_breast_cancer_csv()
    X_tr, y_tr, X_val, y_val, X_te, y_te, _ = make_splits(csv, seed=seed)

    scaler_path = artifacts_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Did you run train.py?")
    scaler = joblib.load(scaler_path)
    X_te = scaler.transform(X_te)

    X_te_t, y_te_t = to_torch_tensors(X_te, y_te)
    test_ds = TensorDataset(X_te_t, y_te_t)
    return test_ds, y_te, class_names, config, X_te.shape[1], len(class_names)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MLP on the Breast Cancer TEST set only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Artifacts directory from training.")
    parser.add_argument("--batch-size", type=int, default=256, help="Eval batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to create the splits (fallback).")
    args = parser.parse_args()

    set_seed(args.seed)
    artifacts_dir = Path(args.artifacts_dir)
    ckpt = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds, y_true, class_names, config, input_dim, num_classes = _load_test(artifacts_dir, args.seed)

    # Build model from training config
    hidden = config.get("hidden_sizes", [256, 256, 256])
    if isinstance(hidden, str):
        hidden = [int(x) for x in hidden.split(",")]
    dropout = float(config.get("dropout", 0.2))
    batchnorm = bool(config.get("batchnorm", True))

    model = build_mlp(input_dim, num_classes, hidden, dropout, batchnorm).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

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
            prob_pos.extend(probs[:, 1].tolist())

    y_true_np = np.concatenate(ys)
    y_pred_np = np.concatenate(yps)
    metrics = compute_metrics(y_true_np, y_pred_np)

    roc_auc = float(roc_auc_score(y_true_np, np.array(prob_pos)))
    metrics["roc_auc"] = roc_auc
    write_json(artifacts_dir / "metrics_test.json", metrics)
    print("[eval] TEST metrics:", metrics)

    labels = class_names
    plot_confusion(y_true_np, y_pred_np, labels, artifacts_dir / "confmat_test.png")

    pred_out = artifacts_dir / "predictions.csv"
    with pred_out.open("w") as f:
        f.write("index,y_true,y_pred,y_prob_pos\n")
        for i, (yt, yp, pp) in enumerate(zip(y_true_np.tolist(), y_pred_np.tolist(), prob_pos)):
            f.write(f"{i},{yt},{yp},{pp}\n")
    print(f"[eval] Wrote predictions to {pred_out}")


if __name__ == "__main__":
    main()
