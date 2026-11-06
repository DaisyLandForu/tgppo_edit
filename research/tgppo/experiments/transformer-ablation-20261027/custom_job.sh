#!/usr/bin/env bash
# Launch one command per Crater Custom job. All paths are on the shared volume.
{
set -euo pipefail
ablation_launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ablation_launch_dir/env.sh"
ablation_log_dir="$RESEARCH_ROOT/tgppo/runs/transformer-ablation-20261027/launcher-logs"
mkdir -p "$ablation_log_dir"
ablation_log_path=$(mktemp "$ablation_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee -a "$ablation_log_path") 2>&1
printf 'Command:'
printf ' %q' "$@"
printf '\nLog: %s\n' "$ablation_log_path"
if [[ "${1:-}" == tests ]]; then
  exec "$TGPPO_PYTHON" -m unittest discover -s "$ablation_launch_dir/tests" -p 'test_ablation*.py' -v
fi
exec "$TGPPO_PYTHON" "$ablation_launch_dir/code/ablation.py" "$@"
}
