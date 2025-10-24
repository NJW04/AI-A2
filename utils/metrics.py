#!/usr/bin/env python3
"""Metrics utilities: accuracy, F1, and confusion matrix plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, micro-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    return {"accuracy": float(acc), "macro_f1": float(macro), "micro_f1": float(micro)}


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path | str) -> Path:
    """
    Save a confusion matrix PNG using matplotlib.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, display_labels=labels, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
