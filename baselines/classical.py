#!/usr/bin/env python3
"""
Classical baselines on the same cached train/val/test splits:

Algorithms:
- k-NN (k=5 default)
- GaussianNB
- Logistic Regression (max_iter=2000)

Metrics: accuracy and macro-F1 on VAL and TEST.
Writes: artifacts/baselines_metrics.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data.breast_cancer import load_or_fetch_breast_cancer as bc_load, make_splits as bc_splits, standardize as bc_std
from data.drybean import load_or_download_drybean as db_load, make_splits as db_splits, standardize as db_std
from utils.io import project_paths, write_json


def _load_dataset(dataset: str, seed: int):
    if dataset == "breast_cancer":
        csv = bc_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, _ = bc_splits(csv, seed=seed)
        X_tr, X_val, X_te, _ = bc_std(X_tr, X_val, X_te)
    elif dataset == "drybean":
        csv = db_load()
        X_tr, y_tr, X_val, y_val, X_te, y_te, _ = db_splits(csv, seed=seed)
        X_tr, X_val, X_te, _ = db_std(X_tr, X_val, X_te)
    else:
        raise ValueError("Unsupported dataset. Use 'breast_cancer' or 'drybean'.")
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def run(dataset: str, algo: str, k: int, seed: int = 42) -> dict:
    X_tr, y_tr, X_val, y_val, X_te, y_te = _load_dataset(dataset, seed)

    if algo == "knn":
        model = KNeighborsClassifier(n_neighbors=k)
    elif algo == "gnb":
        model = GaussianNB()
    elif algo == "logreg":
        model = LogisticRegression(max_iter=2000)
    else:
        raise ValueError("Unsupported --algo. Use 'knn', 'gnb', or 'logreg'.")

    model.fit(X_tr, y_tr)

    def m(xx, yy):
        yp = model.predict(xx)
        return {
            "accuracy": float(accuracy_score(yy, yp)),
            "macro_f1": float(f1_score(yy, yp, average="macro")),
        }

    results = {
        "dataset": dataset,
        "algorithm": algo,
        "k": k if algo == "knn" else None,
        "val": m(X_val, y_val),
        "test": m(X_te, y_te),
        "seed": seed,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run classical baselines (k-NN, GaussianNB, LogisticRegression) on selected dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["breast_cancer", "drybean"], default="breast_cancer",
                        help="Dataset to use for baselines.")
    parser.add_argument("--algo", choices=["knn", "gnb", "logreg"], required=True, help="Baseline algorithm.")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = parser.parse_args()

    res = run(args.dataset, args.algo, args.k, seed=args.seed)
    paths = project_paths()
    out = paths["artifacts"] / "baselines_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, res)
    print(f"[baselines] Wrote: {out}")
    print(res)


if __name__ == "__main__":
    main()
