# baselines/logreg.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from data.wine_white import load_or_fetch_wine_white, make_splits, standardize, CLASS_NAMES
from utils.io import ensure_dir, save_json, default_artifacts_dir
from utils.metrics import compute_metrics
from utils.seed import set_seed


def run_baseline(seed: int, artifacts_dir: Path, data_dir: str | None = None) -> Dict:
    set_seed(seed)
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

    # Standardize using train only; persist scaler to cache & copy to artifacts
    X_train_s, X_val_s, X_test_s, scaler = standardize(
        X_train, X_val, X_test, cache_dir=data_dir
    )
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=None,
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_val_pred = clf.predict(X_val_s)
    y_test_pred = clf.predict(X_test_s)

    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Persist artifacts
    save_json(
        {
            "model": "logistic_regression",
            "seed": seed,
            "class_names": CLASS_NAMES,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        artifacts_dir / "metrics_logreg.json",
    )
    joblib.dump(clf, artifacts_dir / "logreg.joblib")

    # Pretty print
    print("[baseline/logreg] Validation metrics:", val_metrics)
    print("[baseline/logreg] Test metrics:", test_metrics)
    return {"val": val_metrics, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Multinomial Logistic Regression on Wine White (5-class)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory to store artifacts. Defaults to artifacts/wine_white/<timestamp>_logreg_seed<seed>",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing winequality-white.csv (defaults to ./.cache/wine_white/)",
    )
    args = parser.parse_args()

    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else default_artifacts_dir("wine_white", tag=f"logreg_seed{args.seed}")
    )
    ensure_dir(artifacts_dir)
    run_baseline(seed=args.seed, artifacts_dir=artifacts_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
