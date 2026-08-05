#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CLIP_MIA_PYTHON:-}" ]]; then
  python_bin="$CLIP_MIA_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
else
  python_bin="python3"
fi

models="${CLIP_MIA_MODELS:-clip_mlp,visual_adapter}"
gpus="${CLIP_MIA_GPUS:-${CLIP_MIA_GPU:-0}}"
jobs="${CLIP_MIA_JOBS:-1}"
clip_mlp_learning_rate="${CLIP_MLP_MIA_LR:-0.1}"
visual_adapter_learning_rate="${CLIP_ADAPTER_MIA_LR:-0.01}"
forward_args=()
show_help=false

while (($#)); do
  case "$1" in
    --models)
      if (($# < 2)); then
        echo "--models requires a comma-separated value." >&2
        exit 2
      fi
      models="$2"
      shift 2
      ;;
    --models=*)
      models="${1#*=}"
      shift
      ;;
    -h|--help)
      show_help=true
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

cd "$repository_root"

if [[ "$show_help" == true ]]; then
  cat <<'EOF'
Run all frozen-CLIP membership-inference sweeps.

Unified options:
  --models clip_mlp,visual_adapter   Select one or both models (default: both).

Model-specific learning-rate defaults:
  clip_mlp=0.1, visual_adapter=0.01
  --learning-rate RATE overrides the default for every selected model.

All options below are forwarded to each selected sweep, including:
  --datasets CSV        --attacks CSV          --target-client ID
  --seed VALUE          --gpus CSV             --jobs VALUE
  --learning-rate RATE  --rounds VALUE         --dirichlet-alpha VALUE
  --skip-projres        --force                --dry-run
  --summarize-only      --max-runs VALUE

Strict ProjRes runs on the first trainable MLP/Adapter downsampling layer.
EOF
  "$python_bin" scripts/run_clip_mlp_fedmia_sweep.py --help
  exit 0
fi

normalized_models=()
IFS=',' read -r -a requested_models <<< "$models"
for requested in "${requested_models[@]}"; do
  requested="${requested//[[:space:]]/}"
  case "${requested,,}" in
    all)
      normalized_models=(clip_mlp visual_adapter)
      break
      ;;
    clip_mlp|clip-mlp|mlp)
      normalized_models+=(clip_mlp)
      ;;
    visual_adapter|visual-adapter|clip_adapter|clip-adapter|adapter)
      normalized_models+=(visual_adapter)
      ;;
    "")
      ;;
    *)
      echo "Unknown model '$requested'; use clip_mlp, visual_adapter, or all." >&2
      exit 2
      ;;
  esac
done

if ((${#normalized_models[@]} == 0)); then
  echo "At least one model must be selected." >&2
  exit 2
fi

run_model() {
  local model="$1"
  local spec
  local learning_rate
  case "$model" in
    clip_mlp)
      spec="${CLIP_MLP_MIA_SPEC:-configs/clip_mlp_fedmia_attacks_sweep.yaml}"
      learning_rate="$clip_mlp_learning_rate"
      ;;
    visual_adapter)
      spec="${CLIP_ADAPTER_MIA_SPEC:-configs/visual_adapter_fedmia_attacks_sweep.yaml}"
      learning_rate="$visual_adapter_learning_rate"
      ;;
  esac
  echo "================================================================================"
  echo "ULTIMATE CLIP MIA SWEEP | model=$model | spec=$spec"
  echo "================================================================================"
  "$python_bin" scripts/run_clip_mlp_fedmia_sweep.py \
    --spec "$spec" \
    --gpus "$gpus" \
    --jobs "$jobs" \
    --learning-rate "$learning_rate" \
    "${forward_args[@]}"
}

seen=""
for model in "${normalized_models[@]}"; do
  if [[ ",$seen," == *",$model,"* ]]; then
    continue
  fi
  run_model "$model"
  seen="${seen:+$seen,}$model"
done

echo "Completed unified CLIP membership-inference sweep for: $seen"
