#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJRES_PYTHON_BIN="${PROJRES_PYTHON:-python}"
PROJRES_CONFIG_PATH="${PROJRES_CONFIG:-configs/federated_prompt_paper.yaml}"
PROJRES_OUTPUT_ROOT="${PROJRES_RESULTS_DIR:-results/projres_validation}"
PROJRES_SWEEP_PHASE="${1:-mechanism}"

if [[ $# -gt 0 ]]; then
  shift
fi

case "$PROJRES_SWEEP_PHASE" in
  mechanism)
    PROJRES_BATCH_SIZES="1,4,8,16,32"
    PROJRES_LOCAL_STEPS="1,2,5"
    PROJRES_CONTEXT_LENGTHS="16"
    PROJRES_ALPHAS="0.1"
    PROJRES_SEEDS="42"
    ;;
  jacobian)
    PROJRES_BATCH_SIZES="8"
    PROJRES_LOCAL_STEPS="1"
    PROJRES_CONTEXT_LENGTHS="4,8,16,32"
    PROJRES_ALPHAS="0.1"
    PROJRES_SEEDS="42,43,44"
    ;;
  heterogeneity)
    PROJRES_BATCH_SIZES="8"
    PROJRES_LOCAL_STEPS="2"
    PROJRES_CONTEXT_LENGTHS="16"
    PROJRES_ALPHAS="0.1,0.5,1.0"
    PROJRES_SEEDS="42,43,44"
    ;;
  full)
    PROJRES_BATCH_SIZES="1,4,8,16,32"
    PROJRES_LOCAL_STEPS="1,2,5"
    PROJRES_CONTEXT_LENGTHS="4,8,16,32"
    PROJRES_ALPHAS="0.1,0.5,1.0"
    PROJRES_SEEDS="42,43,44"
    ;;
  *)
    echo "Unknown sweep phase: $PROJRES_SWEEP_PHASE" >&2
    echo "Expected one of: mechanism, jacobian, heterogeneity, full" >&2
    exit 2
    ;;
esac

PROJRES_EXECUTION_ARGS=(--execute)
if [[ "${PROJRES_DRY_RUN:-0}" == "1" ]]; then
  PROJRES_EXECUTION_ARGS=()
fi

cd "$REPOSITORY_ROOT"
mkdir -p "$PROJRES_OUTPUT_ROOT/$PROJRES_SWEEP_PHASE"

exec "$PROJRES_PYTHON_BIN" \
  scripts/run_projres_promptfl_validation_sweep.py \
  --config "$PROJRES_CONFIG_PATH" \
  --output-dir "$PROJRES_OUTPUT_ROOT/$PROJRES_SWEEP_PHASE" \
  --batch-sizes "$PROJRES_BATCH_SIZES" \
  --local-steps "$PROJRES_LOCAL_STEPS" \
  --n-ctx "$PROJRES_CONTEXT_LENGTHS" \
  --dirichlet-alphas "$PROJRES_ALPHAS" \
  --seeds "$PROJRES_SEEDS" \
  --max-candidates 128 \
  --ridge 1e-4 \
  --lift-iterations 20 \
  "${PROJRES_EXECUTION_ARGS[@]}" \
  "$@"
