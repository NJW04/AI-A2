# models/mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_sizes: List[int],
    dropout: float = 0.2,
    batchnorm: bool = False,
) -> nn.Module:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        last_dim = h
    layers.append(nn.Linear(last_dim, num_classes))
    return nn.Sequential(*layers)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: List[int] = [256, 256, 256],
        dropout: float = 0.2,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        logits = self.net(x)
        return logits
