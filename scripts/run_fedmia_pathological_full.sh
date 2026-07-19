#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPOSITORY_ROOT"
exec python \
  analysis_scripts/run_fedmia_complex_sweep.py \
  --spec configs/fedmia_pathological_full_sweep.yaml \
  "$@"
