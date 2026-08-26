#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
epsilons="${BERT_DP_EPSILONS:-1,3,5,8}"
seeds="${BERT_DP_SEEDS:-42}"
gpus="${BERT_DP_GPUS:-0}"
jobs="${BERT_DP_JOBS:-1}"
include_nondp=true
require_cuda_flag="--require-cuda"
dry_run=false
forward_args=()

if [[ -n "${BERT_DP_PYTHON:-}" ]]; then
  python_bin="$BERT_DP_PYTHON"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi

while (($#)); do
  case "$1" in
    --epsilons)
      epsilons="$2"
      shift 2
      ;;
    --epsilons=*)
      epsilons="${1#*=}"
      shift
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --seeds=*)
      seeds="${1#*=}"
      shift
      ;;
    --gpus|--gpu)
      gpus="$2"
      shift 2
      ;;
    --gpus=*|--gpu=*)
      gpus="${1#*=}"
      shift
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --jobs=*)
      jobs="${1#*=}"
      shift
      ;;
    --include-nondp)
      include_nondp=true
      shift
      ;;
    --no-nondp)
      include_nondp=false
      shift
      ;;
    --require-cuda)
      require_cuda_flag="--require-cuda"
      shift
      ;;
    --no-require-cuda)
      require_cuda_flag="--no-require-cuda"
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Sweep BERT-Base/SST-5 ProjRes over Record-DP privacy budgets.

Options:
  --epsilons 1,3,5,8    Positive epsilon values (default: 1,3,5,8)
  --seeds 42,43,44      Independent training seeds (default: 42)
  --gpus 0,1            GPU ids assigned round-robin (default: 0)
  --jobs N              Maximum concurrent runs (default: 1)
  --[no-]include-nondp  Include the no-DP ProjRes baseline (default: yes)
  --no-require-cuda     Permit CPU execution; full runs are very slow
  --dry-run             Print every resolved command without training

Unknown options are forwarded to run_fedllm_adapter.py, for example:
  --rounds 100 --results-dir ./results
EOF
      exit 0
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

IFS=',' read -r -a epsilon_values <<< "$epsilons"
IFS=',' read -r -a seed_values <<< "$seeds"
IFS=',' read -r -a gpu_values <<< "$gpus"
if ((${#epsilon_values[@]} == 0 || ${#seed_values[@]} == 0 || ${#gpu_values[@]} == 0)); then
  echo "epsilon, seed, and GPU lists must be non-empty." >&2
  exit 2
fi
for epsilon in "${epsilon_values[@]}"; do
  if ! [[ "$epsilon" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v value="$epsilon" 'BEGIN { exit !(value > 0) }'; then
    echo "Invalid positive epsilon: $epsilon" >&2
    exit 2
  fi
done
for seed in "${seed_values[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid non-negative seed: $seed" >&2
    exit 2
  fi
done
for gpu in "${gpu_values[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid non-negative GPU id: $gpu" >&2
    exit 2
  fi
done

task_kinds=()
task_epsilons=()
task_seeds=()
task_gpus=()
task_index=0
for seed in "${seed_values[@]}"; do
  if [[ "$include_nondp" == true ]]; then
    task_kinds+=("nondp")
    task_epsilons+=("none")
    task_seeds+=("$seed")
    task_gpus+=("${gpu_values[$((task_index % ${#gpu_values[@]}))]}")
    ((task_index += 1))
  fi
  for epsilon in "${epsilon_values[@]}"; do
    task_kinds+=("record_dp")
    task_epsilons+=("$epsilon")
    task_seeds+=("$seed")
    task_gpus+=("${gpu_values[$((task_index % ${#gpu_values[@]}))]}")
    ((task_index += 1))
  done
done

build_command() {
  local kind="$1"
  local epsilon="$2"
  local seed="$3"
  local gpu="$4"
  if [[ "$kind" == "nondp" ]]; then
    command=(
      "$python_bin" scripts/run_fedllm_adapter.py
      --config configs/bert_base_sst5_adapter.yaml
      --defense none
      --attacks projres
      --projres
      --seed "$seed"
      --gpu "$gpu"
      "$require_cuda_flag"
      "${forward_args[@]}"
    )
  else
    command=(
      "$python_bin" scripts/run_fedllm_adapter.py
      --config configs/bert_base_sst5_adapter_record_dp.yaml
      --defense record_dp
      --target-epsilon "$epsilon"
      --attacks projres
      --projres
      --seed "$seed"
      --gpu "$gpu"
      "$require_cuda_flag"
      "${forward_args[@]}"
    )
  fi
}

echo "Expanded ${#task_kinds[@]} BERT ProjRes tasks | epsilons=$epsilons | seeds=$seeds | jobs=$jobs | gpus=$gpus | nondp=$include_nondp"

if [[ "$dry_run" == true ]]; then
  for index in "${!task_kinds[@]}"; do
    build_command \
      "${task_kinds[$index]}" \
      "${task_epsilons[$index]}" \
      "${task_seeds[$index]}" \
      "${task_gpus[$index]}"
    printf 'COMMAND'
    printf ' %q' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

cd "$repository_root"
pids=()
labels=()
failed=0

wait_oldest_task() {
  local pid="${pids[0]}"
  local label="${labels[0]}"
  if wait "$pid"; then
    echo "COMPLETED BERT ProjRes task | $label"
  else
    echo "FAILED BERT ProjRes task | $label" >&2
    failed=1
  fi
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
}

for index in "${!task_kinds[@]}"; do
  while ((${#pids[@]} >= jobs)); do
    wait_oldest_task
  done
  build_command \
    "${task_kinds[$index]}" \
    "${task_epsilons[$index]}" \
    "${task_seeds[$index]}" \
    "${task_gpus[$index]}"
  label="${task_kinds[$index]}/eps${task_epsilons[$index]}/seed${task_seeds[$index]}@gpu${task_gpus[$index]}"
  echo "START BERT ProjRes task | $label"
  "${command[@]}" &
  pids+=("$!")
  labels+=("$label")
done

while ((${#pids[@]})); do
  wait_oldest_task
done
if ((failed)); then
  exit 1
fi
echo "Completed BERT Record-DP ProjRes privacy-budget sweep."
