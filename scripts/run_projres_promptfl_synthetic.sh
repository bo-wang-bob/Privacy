#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJRES_PYTHON_BIN="${PROJRES_PYTHON:-python}"
PROJRES_OUTPUT_ROOT="${PROJRES_RESULTS_DIR:-results/projres_validation}"

cd "$REPOSITORY_ROOT"
mkdir -p "$PROJRES_OUTPUT_ROOT"

exec "$PROJRES_PYTHON_BIN" \
  scripts/validate_projres_promptfl_synthetic.py \
  --classes 8 \
  --dimension 16 \
  --batch-sizes 1,2,4,7,8,12,16 \
  --prompt-widths 4,8,16,32 \
  --trials 10 \
  --ridge 1e-4 \
  --seed 42 \
  --output "$PROJRES_OUTPUT_ROOT/synthetic_validation.json" \
  "$@"
