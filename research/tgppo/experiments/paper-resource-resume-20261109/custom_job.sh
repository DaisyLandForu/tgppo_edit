#!/usr/bin/env bash
set -euo pipefail
resource_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PAPER_LAUNCH_EPOCH=$(date +%s)
resource_experiment=""
resource_previous=""
for resource_argument in "$@"; do
  if [[ "$resource_previous" == --experiment ]]; then resource_experiment="$resource_argument"; fi
  resource_previous="$resource_argument"
done
case "$resource_experiment" in
  aligned) source "$resource_package_dir/../paper-aligned-20261106/env.sh" ;;
  trainmode) source "$resource_package_dir/../paper-trainmode-20261108/env.sh" ;;
  *) printf 'Specify --experiment aligned|trainmode\n' >&2; exit 2 ;;
esac
if [[ "${1:-}" == run ]]; then
  resource_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-resource-resume-20261109/launcher-logs"
  mkdir -p "$resource_log_dir"
  resource_log=$(mktemp "$resource_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
  exec > >(tee -a "$resource_log") 2>&1
  printf 'Resource resume log: %s\n' "$resource_log"
fi
cd /tmp
exec "$TGPPO_PYTHON" -u "$resource_package_dir/code/resource_resume.py" "$@"
