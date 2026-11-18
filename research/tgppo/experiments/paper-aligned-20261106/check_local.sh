#!/usr/bin/env bash
set -euo pipefail
paper_check_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$paper_check_dir/env.sh"
export CUDA_VISIBLE_DEVICES=""
cd /tmp
exec timeout 600 "$TGPPO_PYTHON" -u -m unittest discover -s "$paper_check_dir/tests" -v
