#!/usr/bin/env bash
set -euo pipefail

DATASET="breast_cancer"

# 1) Baselines
bash scripts/run_baselines.sh

# 2) Train default strong MLP (artifacts/<DATASET>/<STAMP>)
bash scripts/run_train.sh

# 3) Auto-discover latest artifacts directory for the chosen dataset and evaluate on TEST
LATEST_RUN="$(ls -dt artifacts/${DATASET}/*/ | head -n1 | sed 's:/*$::')"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "No artifacts found under artifacts/${DATASET}/." >&2
  exit 1
fi

echo "Latest artifacts: ${LATEST_RUN}"
python eval.py --dataset "${DATASET}" --checkpoint "${LATEST_RUN}/best.pt" --artifacts-dir "${LATEST_RUN}"
