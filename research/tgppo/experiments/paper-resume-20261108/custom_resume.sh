#!/usr/bin/env bash
set -euo pipefail
resume_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PAPER_LAUNCH_EPOCH=$(date +%s)
source "$resume_package_dir/../paper-aligned-20261106/env.sh"
case "${1:-}" in
  run)
    resume_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-resume-20261108/launcher-logs"
    mkdir -p "$resume_log_dir"
    resume_log=$(mktemp "$resume_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
    exec > >(tee -a "$resume_log") 2>&1
    printf 'Resume log: %s\n' "$resume_log"
    ;;
  check) ;;
  *) printf 'Usage: %s run --variant b0|mlp|mlp_bc --seed 11|22 [--resume] | check\n' "$0"; exit 2 ;;
esac
cd /tmp
exec "$TGPPO_PYTHON" -u "$resume_package_dir/code/resume.py" "$@"
