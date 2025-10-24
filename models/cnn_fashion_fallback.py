#!/usr/bin/env python3
"""
Tiny CNN used ONLY if the fallback dataset 'fashion' is selected.
Kept intentionally small to run on CPU quickly.
"""
from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_fashion_dataloaders(
    batch_size: int = 64, seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returns train/val/test dataloaders and class names for Fashion-MNIST.
    """
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(root=".cache/fashion", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=".cache/fashion", train=False, download=True, transform=transform)

    # 50k train, 10k val from the 60k training split
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [50_000, 10_000], generator=g)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = list(full_train.classes)
    return train_loader, val_loader, test_loader, class_names
