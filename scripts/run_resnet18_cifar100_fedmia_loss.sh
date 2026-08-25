#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="configs/resnet18_cifar100_fedmia_loss.yaml"
dry_run=false
forward_args=()

if [[ -n "${FEDMIA_PYTHON:-}" ]]; then
  python_bin="$FEDMIA_PYTHON"
elif [[ -x /root/.local/share/mamba/envs/pfedba/bin/python ]]; then
  python_bin="/root/.local/share/mamba/envs/pfedba/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Run the paper-aligned FedMIA-Loss experiment with ResNet18 on CIFAR-100.

The stable paper settings live in:
  configs/resnet18_cifar100_fedmia_loss.yaml

Common overrides forwarded to main.py:
  --gpu N
  --data_root PATH
  --results_dir PATH
  --seed N
  --no-require-cuda

Use --dry-run to print the command without starting training.
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
command=("$python_bin" main.py --config "$config_path" "${forward_args[@]}")
printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$dry_run" == false ]]; then
  "${command[@]}"
fi
