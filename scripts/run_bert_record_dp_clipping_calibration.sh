#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir=""
pilot=false
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
    --run-dir)
      run_dir="$2"
      shift 2
      ;;
    --run-dir=*)
      run_dir="${1#*=}"
      shift
      ;;
    --pilot)
      pilot=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Analyze per-record BERT Adapter gradient norms using a completed non-private run.

By default, the latest completed BERT/SST-5/no-defense FedSGD task is used.
The same fixed records are evaluated at the initialized model and every model
checkpoint available in the task directory.

  --run-dir DIR  Select a completed non-private training task explicitly.
  --pilot        CPU-friendly check: 6 clients x 1 record, 200 bootstraps.

Additional options are forwarded to calibrate_bert_record_dp_clipping.py.
Formal default: all 30 clients x 16 records, 1000 client bootstraps.
EOF
      exit 0
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

cd "$repository_root"
if [[ -z "$run_dir" ]]; then
  run_dir="$(
    find results -mindepth 1 -maxdepth 1 -type d \
      -name '*_bert_adapter_sst5_fedsgd_none_*' -print \
      | sort \
      | tail -n 1
  )"
fi
if [[ -z "$run_dir" || ! -f "$run_dir/run_config.yaml" || ! -f "$run_dir/final_transformer_adapter.pt" ]]; then
  echo "No completed non-private BERT/SST-5 run was found; pass --run-dir." >&2
  exit 2
fi

command=(
  "$python_bin" scripts/calibrate_bert_record_dp_clipping.py
  --run-dir "$run_dir"
)
if [[ "$pilot" == true ]]; then
  command+=(
    --client-ids 0,5,10,15,20,25
    --samples-per-client 1
    --microbatch-size 1
    --bootstrap-replicates 200
  )
fi
command+=("${forward_args[@]}")

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
