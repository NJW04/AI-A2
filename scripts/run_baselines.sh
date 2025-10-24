#!/usr/bin/env bash
set -euo pipefail

DATASET="breast_cancer"

# Run classical baselines end-to-end on cached splits.
python baselines/classical.py --dataset "${DATASET}" --algo logreg
python baselines/classical.py --dataset "${DATASET}" --algo knn --k 5
python baselines/classical.py --dataset "${DATASET}" --algo gnb
