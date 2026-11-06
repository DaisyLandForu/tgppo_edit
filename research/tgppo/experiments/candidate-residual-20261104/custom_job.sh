#!/usr/bin/env bash
set -euo pipefail
cr_launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" != tests ]]; then
  exec bash "$cr_launch_dir/cpu-flex/custom_job.sh" "$@"
fi
source "$cr_launch_dir/env.sh"
export CR_LAUNCH_EPOCH=$(date +%s)
cr_log_dir="$RESEARCH_ROOT/tgppo/runs/candidate-residual-20261104/launcher-logs"
mkdir -p "$cr_log_dir"
cr_log_path=$(mktemp "$cr_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee -a "$cr_log_path") 2>&1
printf 'Log: %s\n' "$cr_log_path"
if [[ "${1:-}" == tests ]]; then
  exec "$TGPPO_PYTHON" -m unittest discover -s "$cr_launch_dir/tests" -v
fi
exec "$TGPPO_PYTHON" "$cr_launch_dir/code/experiment.py" "$@"
