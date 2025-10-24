#!/usr/bin/env python3
"""
Breast Cancer Wisconsin (Diagnostic) dataset utilities.

This loader uses ONLY a CSV named 'wdbc.csv' placed in the REPO ROOT.
Expected columns:
- id (int), diagnosis in {B, M}
- 30 numeric features (names with spaces or underscores are fine, e.g.,
  'concave points_mean' or 'concave_points_mean')
If 'Unnamed: 32' exists, it is dropped automatically.

Provides:
- load_breast_cancer_csv(): returns Path to <repo>/wdbc.csv (raises helpful error if missing)
- make_splits(...): stratified 70/15/15; caches splits to .cache/breast_cancer/
- standardize(...): StandardScaler fit on train only; caches scaler.pkl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.io import project_paths, read_json, write_json


@dataclass
class BreastCancerMeta:
    feature_names: List[str]
    class_names: List[str]  # ["benign", "malignant"]


def _default_data_dir() -> Path:
    """Cache dir used for splits/scaler."""
    paths = project_paths()
    d = paths["cache"] / "breast_cancer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_breast_cancer_csv() -> Path:
    """
    Always load CSV from repo root: <repo>/wdbc.csv.
    Raises with a clear message if not found.
    """
    root = project_paths()["root"]
    csv_path = root / "wdbc.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing 'wdbc.csv' in repo root.\n"
            f"Place your dataset at: {csv_path}\n"
            f"Expected columns: id, diagnosis (B/M), 30 numeric features."
        )
    return csv_path


def _splits_cache_paths(cache_dir: Path, seed: int, val_size: float, test_size: float) -> Dict[str, Path]:
    tag = f"seed{seed}_val{int(val_size*100)}_test{int(test_size*100)}"
    return {
        "npz": cache_dir / f"splits_{tag}.npz",
        "meta": cache_dir / f"meta_{tag}.json",
        "scaler": cache_dir / "scaler.pkl",
    }


def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop stray column if present
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    # Normalize diagnosis text
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column in wdbc.csv.")
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
    cache_dir = _default_data_dir()
    paths = _splits_cache_paths(cache_dir, seed, val_size, test_size)

    if paths["npz"].exists() and paths["meta"].exists():
        npz = np.load(paths["npz"])
        meta_d = read_json(paths["meta"])
        meta = BreastCancerMeta(feature_names=meta_d["feature_names"], class_names=meta_d["class_names"])
        return (npz["X_train"], npz["y_train"], npz["X_val"], npz["y_val"], npz["X_test"], npz["y_test"], meta)

    df = _read_csv(csv_path)
    feature_names = [c for c in df.columns if c not in {"id", "diagnosis"}]

    X = df[feature_names].to_numpy(dtype=np.float32)
    # Map B/M → 0/1 with class_names ["benign","malignant"]
    y = np.where(df["diagnosis"].values == "M", 1, 0).astype(np.int64)
    meta = BreastCancerMeta(feature_names=feature_names, class_names=["benign", "malignant"])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    remaining = 1.0 - test_size
    val_rel = val_size / remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=seed, stratify=y_trainval
    )

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


def save_scaler(scaler: StandardScaler, seed: int, val_size: float, test_size: float) -> Path:
    """
    Persist scaler to the dataset cache directory as 'scaler.pkl'.
    """
    cache_dir = _default_data_dir()
    paths = _splits_cache_paths(cache_dir, seed, val_size, test_size)
    joblib.dump(scaler, paths["scaler"])
    return paths["scaler"]
