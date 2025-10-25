# utils/log.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class CSVLogger:
    """
    Minimal CSV logger. Writes header on first write if file does not exist.
    """

    def __init__(self, path: Path | str, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict):
        # Write only known keys; fill missing with blanks
        cleaned = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(cleaned)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
