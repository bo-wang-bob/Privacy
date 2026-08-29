#!/usr/bin/env bash
set -euo pipefail

# Deprecated compatibility entry point.
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repository_root/scripts/run_clip_adapter_fedmia_attacks.sh" "$@"
