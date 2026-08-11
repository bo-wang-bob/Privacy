#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

export CLIP_MIA_MODELS=clip_lora
exec bash scripts/run_all_clip_fedmia_attacks.sh "$@"
