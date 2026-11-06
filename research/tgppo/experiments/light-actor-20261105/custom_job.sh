#!/usr/bin/env bash
set -euo pipefail
light_launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LIGHT_LAUNCH_EPOCH=$(date +%s)
source "$light_launch_dir/env.sh"
light_log_dir="$RESEARCH_ROOT/tgppo/runs/light-actor-20261105/launcher-logs"
mkdir -p "$light_log_dir"
light_log_path=$(mktemp "$light_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee -a "$light_log_path") 2>&1
printf 'Log: %s\n' "$light_log_path"
exec "$TGPPO_PYTHON" "$light_launch_dir/code/launch.py" "$@"
