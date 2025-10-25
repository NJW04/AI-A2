# utils/io.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Path | str) -> Any:
    return joblib.load(path)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def default_artifacts_dir(project: str, tag: Optional[str] = None, base: str = "artifacts") -> Path:
    t = timestamp()
    sub = f"{t}" + (f"_{tag}" if tag else "")
    return ensure_dir(Path(base) / project / sub)
