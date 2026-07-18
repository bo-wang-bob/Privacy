#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_NAME="${FEDMIA_ENVIRONMENT:-pfedba}"

cd "$REPOSITORY_ROOT"
exec micromamba run -n "$ENVIRONMENT_NAME" python \
  analysis_scripts/run_fedmia_complex_sweep.py \
  --spec configs/fedmia_pathological_full_sweep.yaml \
  "$@"
