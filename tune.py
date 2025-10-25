# tune.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import optuna

from train import TrainConfig, run_training
from utils.io import default_artifacts_dir, ensure_dir, save_json
from utils.log import CSVLogger


def objective(trial: optuna.trial.Trial, base_dir: Path, data_dir: str | None, seed: int) -> float:
    width = trial.suggest_categorical("width", [128, 256, 384])
    depth = trial.suggest_categorical("depth", [2, 3, 4])
    hidden_sizes = [width] * depth
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    batchnorm = trial.suggest_categorical("batchnorm", [False, True])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    cfg = TrainConfig(
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batchnorm=batchnorm,
        epochs=40,  # keep tuning CPU-friendly
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=6,
        class_weights=False,
        seed=seed,
        data_dir=data_dir,
    )

    # Each trial goes to its own subdir (but we still track best.json at the top)
    trial_dir = base_dir / f"trial_{trial.number:03d}"
    metrics, _ = run_training(cfg, artifacts_dir=trial_dir)
    trial.set_user_attr("val_macro_f1", metrics["macro_f1"])
    return metrics["macro_f1"]


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight hyperparameter search for Wine White MLP (Optuna).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Where to put tuning runs (default: artifacts/wine_white/<timestamp>_tune_seed<seed>)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing winequality-white.csv (defaults to ./.cache/wine_white/)",
    )
    args = parser.parse_args()

    base_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else default_artifacts_dir("wine_white", tag=f"tune_seed{args.seed}")
    )
    ensure_dir(base_dir)
    print(f"[tune] Writing trials to {base_dir}")

    study = optuna.create_study(direction="maximize")
    with CSVLogger(base_dir / "trials.csv", ["trial", "val_macro_f1"]) as trial_logger:
        study.optimize(lambda t: objective(t, base_dir, args.data_dir, args.seed), n_trials=args.n_trials)
        for t in study.trials:
            trial_logger.log({"trial": t.number, "val_macro_f1": t.user_attrs.get("val_macro_f1", "")})

    best = study.best_trial
    params = best.params
    params["val_macro_f1"] = best.value
    save_json(params, base_dir / "best.json")
    print("[tune] Best trial:", params)


if __name__ == "__main__":
    main()
