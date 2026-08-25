#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${RECORD_DP_MODEL:-resnet}"
dry_run=false
forward_args=()

if [[ -n "${RECORD_DP_PYTHON:-}" ]]; then
  python_bin="$RECORD_DP_PYTHON"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi

while (($#)); do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --model=*)
      model="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Run a record-level DP federated-learning benchmark.

  --model resnet   ResNet18/CIFAR-100 FedAvg (default)
  --model bert     BERT-Base Adapter/SST-5 one-batch FedSGD
  --dry-run        Print the resolved command only

Additional options are forwarded to the selected Python entry point.
EOF
      exit 0
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

case "$model" in
  resnet)
    command=(
      "$python_bin" main.py
      --config configs/resnet18_cifar100_record_dp.yaml
      "${forward_args[@]}"
    )
    ;;
  bert)
    command=(
      "$python_bin" scripts/run_fedllm_adapter.py
      --config configs/bert_base_sst5_adapter_record_dp.yaml
      "${forward_args[@]}"
    )
    ;;
  *)
    echo "--model must be resnet or bert." >&2
    exit 2
    ;;
esac

cd "$repository_root"
printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$dry_run" == false ]]; then
  "${command[@]}"
fi
