# utils/metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_recall = recall_score(y_true, y_pred, average=None).astype(float).tolist()
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
    }


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path | str
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
