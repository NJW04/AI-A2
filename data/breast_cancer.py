#!/usr/bin/env python3
"""
Breast Cancer Wisconsin (Diagnostic) dataset utilities.

- Preferred: load a CSV named `wdbc.csv` from a provided data_dir or default cache.
  Expected columns: `id`, `diagnosis` in {B, M}, and the 30 numeric features named like
  `radius_mean`, ..., `fractal_dimension_worst`. If an `Unnamed: 32` column exists, drop it.

- Fallback: if CSV is not present, use sklearn.datasets.load_breast_cancer() and build an
  equivalent DataFrame with the same 30 feature names (Kaggle-style), plus `id` and `diagnosis`
  (B/M). Save a copy to cache as `wdbc_from_sklearn.csv` so future runs are instant.

- Provide 70/15/15 stratified split with fixed seed; StandardScaler fit on train only;
  cache splits to .npz and persist scaler.pkl for reuse.

This module mirrors the API style used by data/drybean.py to keep your training code simple.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as sk_load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.io import project_paths, read_json, write_json


@dataclass
class BreastCancerMeta:
    feature_names: List[str]
    class_names: List[str]  # ["benign", "malignant"]


def _default_data_dir() -> Path:
    paths = project_paths()
    d = paths["cache"] / "breast_cancer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kaggle_style_feature_names() -> List[str]:
    # The canonical 30 feature names used by WDBC CSV on Kaggle.
    base = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
    ]
    suffixes = ["mean", "se", "worst"]
    names = []
    for suf in suffixes:
        for b in base:
            names.append(f"{b}_{suf}")
    return names


def _sk_to_kaggle(col: str) -> str:
    # Convert sklearn names like "mean radius", "radius error", "worst concave points"
    # to Kaggle-style: "radius_mean", "radius_se", "concave_points_worst"
    col = col.strip().lower()
    col = col.replace("fractal dimension", "fractal_dimension")
    col = col.replace("concave points", "concave_points")
    tokens = col.split()

    if tokens[0] == "mean":
        return "_".join(tokens[1:] + ["mean"])
    if tokens[-1] == "error":
        return "_".join(tokens[:-1] + ["se"])
    if tokens[0] == "worst":
        return "_".join(tokens[1:] + ["worst"])
    # Fallback (should not happen for this dataset)
    return col.replace(" ", "_")


def _ensure_dataframe_from_sklearn(cache_dir: Path) -> Path:
    """Build a Kaggle-compatible CSV from sklearn and persist to cache."""
    data = sk_load_breast_cancer()
    # Map sklearn feature names to Kaggle-style
    cols = [_sk_to_kaggle(c) for c in data.feature_names]
    df = pd.DataFrame(data.data, columns=cols)
    # diagnosis: sklearn target 0=malignant, 1=benign → map to 'M'/'B'
    diag = np.where(data.target == 0, "M", "B")
    # Create an integer id (for compatibility); not used in modeling
    ids = np.arange(100000, 100000 + df.shape[0])
    out = pd.DataFrame({"id": ids, "diagnosis": diag})
    df = pd.concat([out, df], axis=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cache_dir / "wdbc_from_sklearn.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def load_or_fetch_breast_cancer(data_dir: Path | str | None = None) -> Path:
    """
    Try to load `wdbc.csv` from `data_dir` or default cache. If not found,
    fall back to sklearn to create `wdbc_from_sklearn.csv` and return its path.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    else:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    # Preferred CSV name
    preferred = data_dir / "wdbc.csv"
    if preferred.exists():
        return preferred

    # If user set DATA_DIR, check there too
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        env_csv = Path(env_dir) / "wdbc.csv"
        if env_csv.exists():
            return env_csv

    # Fallback to sklearn (and cache result)
    fallback = data_dir / "wdbc_from_sklearn.csv"
    if fallback.exists():
        return fallback

    # Create from sklearn and persist
    created = _ensure_dataframe_from_sklearn(data_dir)
    return created


def _splits_cache_paths(cache_dir: Path, seed: int, val_size: float, test_size: float) -> Dict[str, Path]:
    tag = f"seed{seed}_val{int(val_size*100)}_test{int(test_size*100)}"
    return {
        "npz": cache_dir / f"splits_{tag}.npz",
        "meta": cache_dir / f"meta_{tag}.json",
        "scaler": cache_dir / "scaler.pkl",
    }


def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop unnamed column if present
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    # Normalize diagnosis capitalization and strip spaces
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column in the CSV.")
    df["diagnosis"] = df["diagnosis"].astype(str).str.upper().str.strip()
    return df


def make_splits(
    csv_path: Path | str,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BreastCancerMeta]:
    """
    Create/reuse stratified 70/15/15 splits with fixed seed.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, meta
    """
    csv_path = Path(csv_path)
    cache_dir = csv_path.parent
    paths = _splits_cache_paths(cache_dir, seed, val_size, test_size)

    # Reuse cached splits if present
    if paths["npz"].exists() and paths["meta"].exists():
        npz = np.load(paths["npz"])
        meta_d = read_json(paths["meta"])
        meta = BreastCancerMeta(feature_names=meta_d["feature_names"], class_names=meta_d["class_names"])
        return (npz["X_train"], npz["y_train"], npz["X_val"], npz["y_val"], npz["X_test"], npz["y_test"], meta)

    df = _read_csv(csv_path)

    # Determine feature columns: all except 'id' and 'diagnosis'
    feature_names = [c for c in df.columns if c not in {"id", "diagnosis"}]
    # If the CSV came from sklearn, ensure consistent order with Kaggle-style naming
    kaggle_order = _kaggle_style_feature_names()
    if set(feature_names) == set(kaggle_order):
        feature_names = kaggle_order

    X = df[feature_names].to_numpy(dtype=np.float32)
    # Map B/M → 0/1 with class_names ["benign","malignant"]
    y = np.where(df["diagnosis"].astype(str).str.upper().values == "M", 1, 0).astype(np.int64)
    meta = BreastCancerMeta(feature_names=feature_names, class_names=["benign", "malignant"])

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Then create validation split relative to remaining data
    remaining = 1.0 - test_size
    val_rel = val_size / remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=seed, stratify=y_trainval
    )

    # Cache splits and meta for reuse
    np.savez_compressed(
        paths["npz"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    write_json(paths["meta"], {"feature_names": feature_names, "class_names": meta.class_names})
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


def standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train only; transform val and test.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    return Xtr, Xva, Xte, scaler


def save_scaler(scaler: StandardScaler, csv_path: Path | str, seed: int, val_size: float, test_size: float) -> Path:
    """
    Persist scaler to the dataset cache directory as 'scaler.pkl'.
    """
    csv_path = Path(csv_path)
    cache_dir = csv_path.parent
    paths = _splits_cache_paths(cache_dir, seed, val_size, test_size)
    joblib.dump(scaler, paths["scaler"])
    return paths["scaler"]
