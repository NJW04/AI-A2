# scripts/run_train.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

python -u train.py \
  --hidden-sizes "256,256,256" \
  --dropout 0.2 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 8 \
  --class-weights \
  --seed 42

echo "Training complete. See artifacts/wine_white for the latest run dir."
