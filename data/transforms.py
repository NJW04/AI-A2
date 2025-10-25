# data/transforms.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights: n_samples / (n_classes * count_c)
    Returns torch.float32 tensor of shape [num_classes].
    """
    counts = np.bincount(y, minlength=num_classes).astype(float)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    # Normalize to mean=1 for numerical stability
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def to_tensor(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x.astype("float32")), torch.from_numpy(y.astype("int64"))
