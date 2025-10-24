#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running from repo root when using Colab:
# %cd /content/project && %env PYTHONPATH=/content/project

python train.py \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --hidden-sizes "256,256,256" \
  --dropout 0.2 \
  --batchnorm \
  --class-weights \
  --patience 8 \
  --seed 42
