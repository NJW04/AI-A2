#!/usr/bin/env bash
set -euo pipefail

DATASET="breast_cancer"

# One good default config: 3x256 MLP, dropout=0.2, lr=1e-3, wd=1e-4, patience=8, class weights ON.
python train.py \
  --dataset "${DATASET}" \
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
