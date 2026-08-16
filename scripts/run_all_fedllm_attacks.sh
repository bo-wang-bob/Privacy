#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${FEDLLM_MIA_PYTHON:-}" ]]; then
  python_bin="$FEDLLM_MIA_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
else
  python_bin="python3"
fi

models="${FEDLLM_MIA_MODELS:-bert,gpt2}"
datasets="${FEDLLM_MIA_DATASETS:-sst5,cola,imdb}"
gpus="${FEDLLM_MIA_GPUS:-${FEDLLM_MIA_GPU:-0}}"
jobs="${FEDLLM_MIA_JOBS:-1}"
dry_run=false
max_runs=0
max_runs_explicit=false
show_help=false
projres_flag="--projres"
forward_args=()

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
    --datasets|--dataset)
      if (($# < 2)); then
        echo "$1 requires a comma-separated value." >&2
        exit 2
      fi
      datasets="$2"
      shift 2
      ;;
    --datasets=*|--dataset=*)
      datasets="${1#*=}"
      shift
      ;;
    --gpus|--gpu)
      if (($# < 2)); then
        echo "$1 requires a comma-separated value." >&2
        exit 2
      fi
      gpus="$2"
      shift 2
      ;;
    --gpus=*|--gpu=*)
      gpus="${1#*=}"
      shift
      ;;
    --jobs)
      if (($# < 2)); then
        echo "--jobs requires a positive integer." >&2
        exit 2
      fi
      jobs="$2"
      shift 2
      ;;
    --jobs=*)
      jobs="${1#*=}"
      shift
      ;;
    --max-runs)
      if (($# < 2)); then
        echo "--max-runs requires a positive integer." >&2
        exit 2
      fi
      max_runs="$2"
      max_runs_explicit=true
      shift 2
      ;;
    --max-runs=*)
      max_runs="${1#*=}"
      max_runs_explicit=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --projres)
      projres_flag="--projres"
      shift
      ;;
    --no-projres|--skip-projres)
      projres_flag="$1"
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

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer." >&2
  exit 2
fi
if ! [[ "$max_runs" =~ ^[0-9]+$ ]] || (
  [[ "$max_runs_explicit" == true ]] && ((max_runs == 0))
); then
  echo "--max-runs must be a positive integer when provided." >&2
  exit 2
fi

cd "$repository_root"

if [[ "$show_help" == true ]]; then
  cat <<'EOF'
Run the unified federated-LLM Adapter membership-inference sweep.

Unified options:
  --models bert,gpt2       Select models (default: both).
  --datasets sst5,cola,imdb
                            Select datasets (default: all three).
  --gpus 0,1               Assign tasks to GPUs round-robin (default: 0).
  --jobs N                 Maximum concurrent processes (default: 1).
  --dry-run                Print commands only; do not load task configs.
  --max-runs N             Keep only the first N expanded tasks.

Options forwarded to each single-task Python runner include:
  --rounds N               --seed N
  --results-dir PATH       --attacks CSV
  --target-client-id N     --require-cuda | --no-require-cuda

ProjRes options:
  --projres                Run strict ProjRes every 50 completed rounds.
  --no-projres             Disable ProjRes.
  --skip-projres           Alias for --no-projres.

Every real task prints its resolved configuration only when that task starts.
Each task evaluates all ten common attacks and strict ProjRes every 50 rounds.
BERT tasks also run observational ICLR ranking every 50 completed rounds.
EOF
  "$python_bin" scripts/run_fedllm_adapter.py --help
  exit 0
fi

normalized_models=()
IFS=',' read -r -a requested_models <<< "$models"
for requested in "${requested_models[@]}"; do
  requested="${requested//[[:space:]]/}"
  case "${requested,,}" in
    all)
      normalized_models=(bert gpt2)
      break
      ;;
    bert|bert_base|bert-base|bert_adapter)
      normalized_models+=(bert)
      ;;
    gpt|gpt2|gpt2_large|gpt2-large|gpt2_adapter)
      normalized_models+=(gpt2)
      ;;
    "")
      ;;
    *)
      echo "Unknown model '$requested'; use bert, gpt2, or all." >&2
      exit 2
      ;;
  esac
done
if ((${#normalized_models[@]} == 0)); then
  echo "At least one model must be selected." >&2
  exit 2
fi

normalized_datasets=()
IFS=',' read -r -a requested_datasets <<< "$datasets"
for requested in "${requested_datasets[@]}"; do
  requested="${requested//[[:space:]]/}"
  case "${requested,,}" in
    all)
      normalized_datasets=(sst5 cola imdb)
      break
      ;;
    sst5|sst-5)
      normalized_datasets+=(sst5)
      ;;
    cola)
      normalized_datasets+=(cola)
      ;;
    imdb)
      normalized_datasets+=(imdb)
      ;;
    "")
      ;;
    *)
      echo "Unknown dataset '$requested'; use sst5, cola, imdb, or all." >&2
      exit 2
      ;;
  esac
done
if ((${#normalized_datasets[@]} == 0)); then
  echo "At least one dataset must be selected." >&2
  exit 2
fi

normalized_gpus=()
IFS=',' read -r -a requested_gpus <<< "$gpus"
for requested in "${requested_gpus[@]}"; do
  requested="${requested//[[:space:]]/}"
  if ! [[ "$requested" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU '$requested'; --gpus requires non-negative integers." >&2
    exit 2
  fi
  normalized_gpus+=("$requested")
done
if ((${#normalized_gpus[@]} == 0)); then
  echo "At least one GPU must be selected." >&2
  exit 2
fi

config_for_task() {
  local model="$1"
  local dataset="$2"
  case "$model/$dataset" in
    bert/sst5)
      echo "configs/bert_base_sst5_adapter.yaml"
      ;;
    bert/cola)
      echo "configs/bert_base_cola_adapter.yaml"
      ;;
    bert/imdb)
      echo "configs/bert_base_imdb_adapter.yaml"
      ;;
    gpt2/*)
      echo "configs/gpt2_large_sst5_adapter.yaml"
      ;;
    *)
      echo "No configuration for model=$model dataset=$dataset" >&2
      return 1
      ;;
  esac
}

run_task() {
  local model="$1"
  local dataset="$2"
  local gpu="$3"
  local config="$4"
  echo "================================================================================"
  echo "FEDLLM MIA TASK START | model=$model | dataset=$dataset | gpu=$gpu"
  echo "================================================================================"
  "$python_bin" scripts/run_fedllm_adapter.py \
    --config "$config" \
    --dataset "$dataset" \
    --gpu "$gpu" \
    "$projres_flag" \
    "${forward_args[@]}"
}

task_models=()
task_datasets=()
task_gpus=()
task_configs=()
task_index=0
stop_expansion=false
seen_tasks=","
for model in "${normalized_models[@]}"; do
  for dataset in "${normalized_datasets[@]}"; do
    task_key="$model/$dataset"
    if [[ "$seen_tasks" == *",$task_key,"* ]]; then
      continue
    fi
    if ((max_runs > 0 && task_index >= max_runs)); then
      stop_expansion=true
      break
    fi
    gpu="${normalized_gpus[$((task_index % ${#normalized_gpus[@]}))]}"
    task_models+=("$model")
    task_datasets+=("$dataset")
    task_gpus+=("$gpu")
    task_configs+=("$(config_for_task "$model" "$dataset")")
    seen_tasks+="$task_key,"
    ((task_index += 1))
  done
  if [[ "$stop_expansion" == true ]]; then
    break
  fi
done

echo "Expanded ${#task_models[@]} FedLLM membership-inference tasks | models=${models} | datasets=${datasets} | jobs=${jobs} | gpus=${gpus}"

if [[ "$dry_run" == true ]]; then
  for index in "${!task_models[@]}"; do
    command=(
      "$python_bin" scripts/run_fedllm_adapter.py
      --config "${task_configs[$index]}"
      --dataset "${task_datasets[$index]}"
      --gpu "${task_gpus[$index]}"
      "$projres_flag"
      "${forward_args[@]}"
    )
    printf 'COMMAND'
    printf ' %q' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

pids=()
labels=()
failed=0

wait_oldest_task() {
  local pid="${pids[0]}"
  local label="${labels[0]}"
  if wait "$pid"; then
    echo "COMPLETED FedLLM task | $label"
  else
    echo "FAILED FedLLM task | $label" >&2
    failed=1
  fi
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
}

for index in "${!task_models[@]}"; do
  while ((${#pids[@]} >= jobs)); do
    wait_oldest_task
  done
  run_task \
    "${task_models[$index]}" \
    "${task_datasets[$index]}" \
    "${task_gpus[$index]}" \
    "${task_configs[$index]}" &
  pids+=("$!")
  labels+=("${task_models[$index]}/${task_datasets[$index]}@gpu${task_gpus[$index]}")
done

while ((${#pids[@]})); do
  wait_oldest_task
done

if ((failed)); then
  exit 1
fi
echo "Completed unified FedLLM membership-inference sweep."
