#!/usr/bin/env bash
set -euo pipefail
paper_launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PAPER_LAUNCH_EPOCH=$(date +%s)
source "$paper_launcher_dir/env.sh"
case "${1:-}" in
  run|check|status) ;;
  *) printf 'Usage: %s run --variant b0|mlp|mlp_bc --seed 11|22 [--device cpu|cuda] [--resume]\n' "$0"; exit 2 ;;
esac
paper_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-trainmode-20261108/launcher-logs"
mkdir -p "$paper_log_dir"
paper_log=$(mktemp "$paper_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee -a "$paper_log") 2>&1
printf 'Log: %s\n' "$paper_log"
cd /tmp
exec "$TGPPO_PYTHON" -u "$paper_launcher_dir/code/paper.py" "$@"
