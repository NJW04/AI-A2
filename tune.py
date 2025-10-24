#!/usr/bin/env python3
"""
Hyperparameter tuning for the Dry Bean MLP.
Uses Optuna if available; otherwise falls back to a small manual grid search.

Search space:
- lr ∈ [1e-4 … 3e-3] (log)
- weight_decay ∈ [1e-6 … 1e-3] (log)
- dropout ∈ [0.0 … 0.4]
- width ∈ {128, 256, 384}
- depth ∈ {2, 3, 4}
- batchnorm ∈ {on, off}

Objective: maximize val macro-F1 with early stopping.
Outputs:
- tuning/trials.csv
- artifacts/best.json (best hyperparams only, not a trained model)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.drybean import load_or_download_drybean, make_splits, standardize
from data.transforms import compute_class_weights, to_torch_tensors
from models.mlp import build_mlp
from utils.io import project_paths, write_json
from utils.metrics import compute_metrics
from utils.seed import set_seed

# Try to import Optuna; it's optional
try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


def _build_loaders(batch_size: int, seed: int):
    csv = load_or_download_drybean()
    X_tr, y_tr, X_val, y_val, X_te, y_te, meta = make_splits(csv, seed=seed)
    X_tr, X_val, X_te, _ = standardize(X_tr, X_val, X_te)
    X_tr_t, y_tr_t = to_torch_tensors(X_tr, y_tr)
    X_val_t, y_val_t = to_torch_tensors(X_val, y_val)
    tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, num_workers=0)
    va = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False, num_workers=0)
    return tr, va, meta, y_tr


def _run_once(config: Dict, batch_size: int, seed: int, patience: int, max_epochs: int) -> float:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_loader, va_loader, meta, y_train = _build_loaders(batch_size, seed)

    hidden = [config["width"]] * config["depth"]
    model = build_mlp(
        input_dim=len(meta.feature_names),
        num_classes=len(meta.class_names),
        hidden_sizes=hidden,
        dropout=config["dropout"],
        batchnorm=config["batchnorm"],
    ).to(device)

    # Optionally use class weights for stability during tuning
    cw = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best = -1.0
    no_imp = 0
    for _ in range(max_epochs):
        # train
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
        # val
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb).argmax(dim=1)
                ys.append(yb.cpu().numpy())
                yps.append(yp.cpu().numpy())
        m = compute_metrics(np.concatenate(ys), np.concatenate(yps))
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    return float(best)


def tune_optuna(n_trials: int, seed: int, patience: int, epochs: int, batch_size: int, trials_csv: Path):
    def objective(trial):
        cfg = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "width": trial.suggest_categorical("width", [128, 256, 384]),
            "depth": trial.suggest_categorical("depth", [2, 3, 4]),
            "batchnorm": trial.suggest_categorical("batchnorm", [True, False]),
        }
        score = _run_once(cfg, batch_size, seed, patience, epochs)
        trial.set_user_attr("macro_f1", score)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # write trials csv
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    with trials_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "lr", "weight_decay", "dropout", "width", "depth", "batchnorm", "macro_f1"])
        for t in study.trials:
            p = t.params
            writer.writerow([t.number, p["lr"], p["weight_decay"], p["dropout"], p["width"], p["depth"], p["batchnorm"], t.user_attrs.get("macro_f1", None)])

    best_params = study.best_trial.params
    best_params["macro_f1"] = study.best_value
    return best_params


def tune_grid(n_trials: int, seed: int, patience: int, epochs: int, batch_size: int, trials_csv: Path):
    # A small discrete grid; we cap at n_trials if requested smaller than full grid size
    grid = {
        "lr": [1e-4, 5e-4, 1e-3, 2e-3],
        "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
        "dropout": [0.0, 0.2, 0.4],
        "width": [128, 256, 384],
        "depth": [2, 3, 4],
        "batchnorm": [True, False],
    }
    combos = list(itertools.product(*grid.values()))
    # Limit number of evaluated combos to n_trials
    combos = combos[:n_trials]

    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    best_cfg, best_score = None, -1.0
    with trials_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "lr", "weight_decay", "dropout", "width", "depth", "batchnorm", "macro_f1"])
        for i, vals in enumerate(combos):
            cfg = dict(zip(grid.keys(), vals))
            score = _run_once(cfg, batch_size, seed, patience, epochs)
            writer.writerow([i, cfg["lr"], cfg["weight_decay"], cfg["dropout"], cfg["width"], cfg["depth"], cfg["batchnorm"], score])
            if score > best_score:
                best_cfg, best_score = cfg, score
    best_cfg["macro_f1"] = best_score
    return best_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Dry Bean MLP (Optuna if available, else grid).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Number of trials to run.")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs per trial.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience per trial.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    paths = project_paths()
    trials_csv = paths["artifacts"] / "tuning" / "trials.csv"

    if HAS_OPTUNA:
        print("[tune] Using Optuna.")
        best = tune_optuna(args.n_trials, args.seed, args.patience, args.epochs, args.batch_size, trials_csv)
    else:
        print("[tune] Optuna not available; using small grid.")
        best = tune_grid(args.n_trials, args.seed, args.patience, args.epochs, args.batch_size, trials_csv)

    best_json = paths["artifacts"] / "best.json"
    write_json(best_json, {"best_hyperparams": best, "note": "Use these values in train.py for a strong configuration."})
    print(f"[tune] Best (val macro-F1={best['macro_f1']:.4f}): {best}")
    print(f"[tune] Trials written to: {trials_csv}")
    print(f"[tune] Best hyperparams written to: {best_json}")


if __name__ == "__main__":
    main()
