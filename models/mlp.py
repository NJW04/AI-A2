#!/usr/bin/env python3
"""Configurable MLP classifier for tabular data (Dry Bean)."""

from __future__ import annotations
from typing import Iterable, List

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Simple MLP for multi-class classification with optional BatchNorm and Dropout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Iterable[int] = (256, 256, 256),
        dropout: float = 0.2,
        batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_sizes: Iterable[int],
    dropout: float,
    batchnorm: bool,
) -> MLPClassifier:
    """Factory method used by train/tune scripts."""
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batchnorm=batchnorm,
    )
