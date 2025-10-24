#!/usr/bin/env python3
"""Tabular transforms and utilities for Dry Bean."""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute class weights using sklearn's 'balanced' logic.
    Returns a float32 numpy array of shape (num_classes,).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return weights.astype(np.float32)


def to_torch_tensors(
    X: np.ndarray, y: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to torch tensors with correct dtype.
    X -> float32, y -> int64
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return X_t, y_t
