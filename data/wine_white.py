# data/wine_white.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CACHE_SUBDIR = Path(".cache/wine_white")
CSV_FILENAME = "winequality-white.csv"

CLASS_NAMES = ["very_low", "low", "medium", "high", "very_high"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def _ensure_cache_dir(dirpath: Optional[Path]) -> Path:
    if dirpath is None:
        dirpath = CACHE_SUBDIR
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def _csv_path(data_dir: Optional[Path | str]) -> Path:
    base = _ensure_cache_dir(None if data_dir is None else Path(data_dir))
    return base / CSV_FILENAME


def _bin_quality(q: int) -> str:
    # Deterministic 5-class binning
    if q in {3, 4}:
        return "very_low"
    if q == 5:
        return "low"
    if q == 6:
        return "medium"
    if q == 7:
        return "high"
    if q in {8, 9}:
        return "very_high"
    raise ValueError(f"Unexpected raw quality score: {q}")


def load_or_fetch_wine_white(
    data_dir: Path | str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load the Wine Quality (white) CSV from <data_dir or ./.cache/wine_white/>.
    - Expects file named 'winequality-white.csv' (semicolon-delimited, UCI format).
    - Creates 5-class 'label' and integer 'label_idx' columns.
    Returns: (DataFrame, metadata dict with 'feature_names', 'class_names').
    """
    csv_path = _csv_path(data_dir)
    if not csv_path.exists():
        msg = (
            f"[wine_white] Missing CSV at: {csv_path}\n"
            "Please manually place 'winequality-white.csv' in that directory.\n"
            "Tip (Colab/local): download from a trusted source (e.g., UCI/Kaggle) and then:\n"
            f"  mkdir -p {csv_path.parent}\n"
            f"  cp /path/to/winequality-white.csv {csv_path}\n"
            "Or pass --data-dir <directory-containing-CSV> to CLI scripts.\n"
        )
        raise FileNotFoundError(msg)

    # Read CSV (UCI files are semicolon-separated)
    df = pd.read_csv(csv_path, sep=";")

    # Ensure expected columns
    expected = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unexpected CSV schema. Missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    feature_names = expected[:-1]
    # Cast features to float32
    df[feature_names] = df[feature_names].astype("float32")

    # Create 5-class labels
    df["label"] = df["quality"].astype(int).apply(_bin_quality)
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)

    meta = {"feature_names": feature_names, "class_names": CLASS_NAMES}

    # Persist a tiny metadata JSON in cache for convenience/debugging
    cache_dir = _ensure_cache_dir(Path(data_dir) if data_dir else None)
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return df, meta


def make_splits(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    data_dir: Path | str | None = None,
) -> Dict[str, np.ndarray]:
    """
    Stratified 70/15/15 split on 'label_idx'. Persist indices to
    ./.cache/wine_white/splits_seed{seed}_5bins.npz
    """
    assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and (val_size + test_size) < 0.9
    cache_dir = _ensure_cache_dir(Path(data_dir) if data_dir else None)
    split_path = cache_dir / f"splits_seed{seed}_5bins.npz"
    if split_path.exists():
        arrs = np.load(split_path)
        return {k: arrs[k] for k in arrs.files}

    y = df["label_idx"].values
    idx_all = np.arange(len(df))
    # First split: train vs (val+test)
    rest_size = val_size + test_size
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx_all, y, test_size=rest_size, random_state=seed, stratify=y
    )
    # Second split: val vs test (split rest evenly according to provided sizes)
    rest_val_ratio = val_size / (val_size + test_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest, y_rest, test_size=(1 - rest_val_ratio), random_state=seed, stratify=y_rest
    )

    np.savez_compressed(
        split_path,
        train_idx=np.sort(idx_train),
        val_idx=np.sort(idx_val),
        test_idx=np.sort(idx_test),
        seed=np.array([seed]),
    )
    return {
        "train_idx": np.sort(idx_train),
        "val_idx": np.sort(idx_val),
        "test_idx": np.sort(idx_test),
    }


def standardize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    cache_dir: Path | str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train only; transform val/test; save scaler.pkl.
    """
    cache_path = _ensure_cache_dir(Path(cache_dir) if cache_dir else None)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_test_s = scaler.transform(X_test).astype("float32")
    joblib.dump(scaler, cache_path / "scaler.pkl")
    return X_train_s, X_val_s, X_test_s, scaler
