#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CLIP_ADAPTER_TRAIN_PYTHON:-${VISUAL_ADAPTER_TRAIN_PYTHON:-}}" ]]; then
  python_bin="${CLIP_ADAPTER_TRAIN_PYTHON:-$VISUAL_ADAPTER_TRAIN_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
else
  python_bin="python3"
fi

config_path="${CLIP_ADAPTER_TRAIN_CONFIG:-${VISUAL_ADAPTER_TRAIN_CONFIG:-configs/clip_adapter_privacy.yaml}}"
learning_rate="${CLIP_ADAPTER_TRAIN_LR:-${VISUAL_ADAPTER_TRAIN_LR:-0.001}}"
dry_run=false
forward_args=()

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

cd "$repository_root"

command=(
  "$python_bin"
  main.py
  --config "$config_path"
  --dataset_name caltech101
  --learning_rate "$learning_rate"
  --num_global_iters 50
  --local_epochs 1
  --partition_mode iid
  --gpu 0
  "${forward_args[@]}"
  --model_type clip_adapter
  --attack none
  --defense none
)

if [[ "$dry_run" == true ]]; then
  printf 'COMMAND'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
