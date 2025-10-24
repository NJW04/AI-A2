#!/usr/bin/env python3
"""I/O helpers and canonical project paths (Colab-friendly)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def project_paths() -> Dict[str, Path]:
    """
    Returns canonical paths used across the repo.
    Root is two levels up from this file (../..).
    """
    root = Path(__file__).resolve().parents[1]  # .../utils/ -> repo root
    cache = root / ".cache"
    artifacts = root / "artifacts"
    cache.mkdir(exist_ok=True, parents=True)
    artifacts.mkdir(exist_ok=True, parents=True)
    return {"root": root, "cache": cache, "artifacts": artifacts}
