#!/usr/bin/env python3
"""Simple JSON and CSV logging helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

from .io import write_json


def log_json(path: Path | str, obj: dict) -> None:
    write_json(path, obj)


def append_csv(
    path: Path | str, dict_row: Dict[str, object], fieldnames_ordered: Optional[Iterable[str]] = None
) -> None:
    """
    Append a row to a CSV file; create header if the file does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames_ordered is None:
        fieldnames_ordered = list(dict_row.keys())
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ordered)
        if not exists:
            writer.writeheader()
        writer.writerow(dict_row)
