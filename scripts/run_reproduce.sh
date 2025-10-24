#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running from repo root when using Colab:
# %cd /content/project && %env PYTHONPATH=/content/project

# 1) Baselines
bash scripts/run_baselines.sh

# 2) Train default strong MLP (artifacts/breast_cancer/<STAMP>)
bash scripts/run_train.sh

# 3) Auto-discover latest artifacts directory and evaluate on TEST
LATEST_RUN="$(ls -dt artifacts/breast_cancer/*/ | head -n1 | sed 's:/*$::')"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "No artifacts found under artifacts/breast_cancer/." >&2
  exit 1
fi

echo "Latest artifacts: ${LATEST_RUN}"
python eval.py --checkpoint "${LATEST_RUN}/best.pt" --artifacts-dir "${LATEST_RUN}"
