#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPOSITORY_ROOT"
exec python \
  analysis_scripts/run_fedmia_complex_sweep.py \
  --spec configs/promptres_prompt_methods_fewshot_sweep.yaml \
  --gpus 1 \
  --jobs 1 \
  --methods promptfl \
  --datasets cifar100,caltech101 \
  --rounds 50 \
  "$@"
