#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPTMIA_PYTHON_BIN="${PROMPTMIA_PYTHON:-python}"
PROMPTMIA_CONFIG_PATH="${PROMPTMIA_CONFIG:-configs/promptmia_promptfl_real.yaml}"
PROMPTMIA_DATA_ROOT_PATH="${PROMPTMIA_DATA_ROOT:-./data}"
PROMPTMIA_CACHE_ROOT_PATH="${PROMPTMIA_CACHE_DIR:-./checkpoints/clip-vit-base-patch32}"
PROMPTMIA_GPU_VALUE="${PROMPTMIA_GPU:-0}"
PROMPTMIA_SEED_VALUE="${PROMPTMIA_SEED:-42}"
PROMPTMIA_TARGET_CLIENT_VALUE="${PROMPTMIA_TARGET_CLIENT:-0}"

cd "$REPOSITORY_ROOT"

# exec performs exactly one formal training-and-audit run.
exec "$PROMPTMIA_PYTHON_BIN" main.py \
  --config "$PROMPTMIA_CONFIG_PATH" \
  --data_root "$PROMPTMIA_DATA_ROOT_PATH" \
  --cache_dir "$PROMPTMIA_CACHE_ROOT_PATH" \
  --gpu "$PROMPTMIA_GPU_VALUE" \
  --seed "$PROMPTMIA_SEED_VALUE" \
  --target_client_id "$PROMPTMIA_TARGET_CLIENT_VALUE" \
  --aggregator promptfl \
  --attack promptmia \
  --defense none \
  "$@"
