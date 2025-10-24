#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running from repo root when using Colab:
# In Colab cell before this script: %cd /content/project && %env PYTHONPATH=/content/project

python baselines/classical.py --algo logreg
python baselines/classical.py --algo knn --k 5
python baselines/classical.py --algo gnb
