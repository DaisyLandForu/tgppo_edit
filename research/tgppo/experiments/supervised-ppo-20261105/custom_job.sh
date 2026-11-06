#!/usr/bin/env bash
set -euo pipefail
supervised_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RESEARCH_ROOT=$(cd -- "$supervised_package_dir/../../.." && pwd)
export LIGHT_LAUNCH_EPOCH=$(date +%s)
source "$supervised_package_dir/env.sh"
export PYTHONDONTWRITEBYTECODE=1
supervised_log_dir="$RESEARCH_ROOT/tgppo/runs/supervised-ppo-20261105/launcher-logs"
mkdir -p "$supervised_log_dir"
supervised_log_path=$(mktemp "$supervised_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee -a "$supervised_log_path") 2>&1
printf 'Log: %s\n' "$supervised_log_path"
cd /tmp
exec "$TGPPO_PYTHON" -u "$supervised_package_dir/code/launch.py" "$@"
