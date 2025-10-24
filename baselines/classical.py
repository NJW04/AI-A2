#!/usr/bin/env python3
"""
Classical baselines on the same cached train/val/test splits (Breast Cancer only):

Algorithms:
- Logistic Regression (max_iter=2000)
- k-NN (k=5 default)
- GaussianNB

Metrics: accuracy and macro-F1 on VAL and TEST.
Writes: artifacts/baselines_metrics.json
"""
from __future__ import annotations

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data.breast_cancer import load_breast_cancer_csv, make_splits, standardize
from utils.io import project_paths, write_json


def run(algo: str, k: int, seed: int = 42) -> dict:
    csv = load_breast_cancer_csv()
    X_tr, y_tr, X_val, y_val, X_te, y_te, _ = make_splits(csv, seed=seed)
    X_tr, X_val, X_te, _ = standardize(X_tr, X_val, X_te)

    if algo == "logreg":
        model = LogisticRegression(max_iter=2000)
    elif algo == "knn":
        model = KNeighborsClassifier(n_neighbors=k)
    elif algo == "gnb":
        model = GaussianNB()
    else:
        raise ValueError("Unsupported --algo. Use 'logreg', 'knn', or 'gnb'.")

    model.fit(X_tr, y_tr)

    def m(xx, yy):
        yp = model.predict(xx)
        return {
            "accuracy": float(accuracy_score(yy, yp)),
            "macro_f1": float(f1_score(yy, yp, average="macro")),
        }

    results = {
        "dataset": "breast_cancer",
        "algorithm": algo,
        "k": k if algo == "knn" else None,
        "val": m(X_val, y_val),
        "test": m(X_te, y_te),
        "seed": seed,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run classical baselines (LogReg, k-NN, GaussianNB) on Breast Cancer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=["logreg", "knn", "gnb"], required=True, help="Baseline algorithm.")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = parser.parse_args()

    res = run(args.algo, args.k, seed=args.seed)
    paths = project_paths()
    out = paths["artifacts"] / "baselines_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, res)
    print(f"[baselines] Wrote: {out}")
    print(res)


if __name__ == "__main__":
    main()
