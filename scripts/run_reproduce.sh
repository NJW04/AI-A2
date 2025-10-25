# scripts/run_reproduce.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

echo "==> Running baseline"
bash scripts/run_baseline.sh

echo "==> Training MLP"
bash scripts/run_train.sh

LATEST_RUN="$(ls -dt artifacts/wine_white/* | head -n1)"
echo "==> Evaluating latest run: $LATEST_RUN"
python -u eval.py --checkpoint "$LATEST_RUN/best.pt" --artifacts-dir "$LATEST_RUN"
