#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${VISUAL_ADAPTER_TRAIN_PYTHON:-}" ]]; then
  python_bin="$VISUAL_ADAPTER_TRAIN_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
else
  python_bin="python3"
fi

config_path="${VISUAL_ADAPTER_TRAIN_CONFIG:-configs/visual_adapter_privacy.yaml}"
learning_rate="${VISUAL_ADAPTER_TRAIN_LR:-0.01}"
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
  --local_epochs 5
  --dirichlet_alpha 0.1
  --gpu 0
  "${forward_args[@]}"
  --model_type visual_adapter
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
