#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pilot=false
dry_run=false
forward_args=()

if [[ -n "${BERT_CLIENT_DP_PYTHON:-}" ]]; then
  python_bin="$BERT_CLIENT_DP_PYTHON"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi

while (($#)); do
  case "$1" in
    --pilot)
      pilot=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Run non-private BERT FedSGD and calibrate local client-DP clipping norm S.

The formal default runs all 500 rounds and observes all 30 clients. It writes
non-private analysis artifacts under analysis_scripts/, never results/.

Options:
  --pilot     Two training rounds, six observed clients, 200 bootstraps.
              All 30 clients still train so the FedSGD protocol is unchanged.
  --dry-run   Print the resolved experiment without loading BERT or training.

Additional options are forwarded to calibrate_bert_local_client_dp_clipping.py:
  --gpu 0 --client-ids all --target-epsilon 3 --delta 1e-5
  --rounds 500 --accounting-rounds 500 --output-dir PATH
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
command=(
  "$python_bin" scripts/calibrate_bert_local_client_dp_clipping.py
  --config configs/bert_base_sst5_adapter.yaml
)
if [[ "$pilot" == true ]]; then
  command+=(
    --rounds 2
    --accounting-rounds 500
    --client-ids 0,5,10,15,20,25
    --bootstrap-replicates 200
  )
fi
if [[ "$dry_run" == true ]]; then
  command+=(--dry-run)
fi
command+=("${forward_args[@]}")

printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
