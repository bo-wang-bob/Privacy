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
visual_adapter_learning_rate="${CLIP_ADAPTER_MIA_LR:-0.001}"
partition_mode="${CLIP_MIA_PARTITION_MODE:-iid}"
partition_mode_explicit=false
if [[ -n "${CLIP_MIA_PARTITION_MODE:-}" ]]; then
  partition_mode_explicit=true
fi
dirichlet_alpha_requested=false
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
    --partition-mode)
      if (($# < 2)); then
        echo "--partition-mode requires iid or dirichlet." >&2
        exit 2
      fi
      partition_mode="$2"
      partition_mode_explicit=true
      shift 2
      ;;
    --partition-mode=*)
      partition_mode="${1#*=}"
      partition_mode_explicit=true
      shift
      ;;
    --dirichlet-alpha)
      if (($# < 2)); then
        echo "--dirichlet-alpha requires a positive value." >&2
        exit 2
      fi
      dirichlet_alpha_requested=true
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --dirichlet-alpha=*)
      dirichlet_alpha_requested=true
      forward_args+=("$1")
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

if [[ "$dirichlet_alpha_requested" == true && "$partition_mode_explicit" == false ]]; then
  partition_mode="dirichlet"
fi
case "${partition_mode,,}" in
  iid|dirichlet)
    partition_mode="${partition_mode,,}"
    ;;
  *)
    echo "Unknown partition mode '$partition_mode'; use iid or dirichlet." >&2
    exit 2
    ;;
esac

cd "$repository_root"

if [[ "$show_help" == true ]]; then
  cat <<'EOF'
Run all frozen-CLIP membership-inference sweeps.

Unified options:
  --models clip_mlp,visual_adapter   Select one or both models (default: both).
  --partition-mode iid|dirichlet    Data partition (default: iid).

Model-specific learning-rate defaults:
  clip_mlp=0.1, visual_adapter=0.001
  --learning-rate RATE overrides the default for every selected model.
  Both model specs keep the client learning rate constant (decay=1.0).

All options below are forwarded to each selected sweep, including:
  --datasets CSV        --attacks CSV          --target-client ID|all
  --seed VALUE          --gpus CSV             --jobs VALUE
  --learning-rate RATE  --learning-rate-decay RATE
  --learning-rate-decay-interval ROUNDS
  --rounds VALUE        --dirichlet-alpha VALUE (selects dirichlet unless
                         --partition-mode was explicitly provided)
  --projres-round N|last
  --skip-projres        --force                --dry-run
  --summarize-only      --max-runs VALUE

ProjRes runs only for Visual Adapter, using its selected real one-batch FedSGD
upload and attacking the first Adapter downsampling layer. CLIP-MLP runs only
the six generic attacks in this unified entry point.
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
    --partition-mode "$partition_mode" \
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
