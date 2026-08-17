#!/usr/bin/env bash
set -euo pipefail

python_bin="${PROJRES_MLP_PYTHON:-python}"
config_path="${PROJRES_MLP_CONFIG:-configs/clip_mlp_projres.yaml}"
output_path="${PROJRES_MLP_OUTPUT:-results/projres_mlp_strict.json}"

"${python_bin}" scripts/validate_projres_mlp_real.py \
  --config "${config_path}" \
  --target-client "${PROJRES_MLP_CLIENT:-all}" \
  --output "${output_path}" \
  "$@"
