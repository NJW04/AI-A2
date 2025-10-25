# scripts/run_baseline.sh
#!/usr/bin/env bash
set -euo pipefail

# Robust repo-root detection
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

# Default artifacts dir
ART_DIR="artifacts/wine_white/$(date +%Y%m%d-%H%M%S)_logreg_seed42"
mkdir -p "$ART_DIR"

python -u baselines/logreg.py \
  --seed 42 \
  --artifacts-dir "$ART_DIR"

echo "Baseline artifacts in: $ART_DIR"
