#!/usr/bin/env bash
set -euo pipefail
paper_launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PAPER_LAUNCH_EPOCH=$(date +%s)
source "$paper_launcher_dir/env.sh"
export CUDA_VISIBLE_DEVICES=""
case "${1:-}" in
  normalize|eval|summarize|check|status) ;;
  *) printf 'CPU entry: normalize --shard N --shards M | eval --shard 0..16 | summarize | check | status\n'; exit 2 ;;
esac
cd /tmp
exec "$TGPPO_PYTHON" -u "$paper_launcher_dir/code/paper.py" "$@"
