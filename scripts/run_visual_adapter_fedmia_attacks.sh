#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CLIP_ADAPTER_MIA_PYTHON:-}" ]]; then
  python_bin="$CLIP_ADAPTER_MIA_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
else
  python_bin="python3"
fi
spec_path="${CLIP_ADAPTER_MIA_SPEC:-configs/visual_adapter_fedmia_attacks_sweep.yaml}"
gpus="${CLIP_ADAPTER_MIA_GPUS:-${CLIP_ADAPTER_MIA_GPU:-0}}"
jobs="${CLIP_ADAPTER_MIA_JOBS:-1}"

cd "$repository_root"
exec "$python_bin" \
  scripts/run_clip_mlp_fedmia_sweep.py \
  --spec "$spec_path" \
  --gpus "$gpus" \
  --jobs "$jobs" \
  "$@"
