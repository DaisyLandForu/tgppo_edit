#!/usr/bin/env bash
set -euo pipefail
recovery_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PAPER_LAUNCH_EPOCH=$(date +%s)
recovery_experiment=""
recovery_previous=""
for recovery_argument in "$@"; do
  if [[ "$recovery_previous" == --experiment ]]; then recovery_experiment="$recovery_argument"; fi
  recovery_previous="$recovery_argument"
done
case "$recovery_experiment" in
  aligned) source "$recovery_package_dir/../paper-aligned-20261106/env.sh" ;;
  trainmode) source "$recovery_package_dir/../paper-trainmode-20261108/env.sh" ;;
  *) printf 'Specify --experiment aligned|trainmode\n' >&2; exit 2 ;;
esac
if [[ "${1:-}" == run ]]; then
  recovery_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-recovery-20261109/launcher-logs"
  mkdir -p "$recovery_log_dir"
  recovery_log=$(mktemp "$recovery_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
  exec > >(tee -a "$recovery_log") 2>&1
  printf 'Recovery log: %s\n' "$recovery_log"
fi
cd /tmp
exec "$TGPPO_PYTHON" -u "$recovery_package_dir/code/recover.py" "$@"
