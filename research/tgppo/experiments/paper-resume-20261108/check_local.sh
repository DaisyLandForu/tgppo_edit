#!/usr/bin/env bash
set -euo pipefail
resume_check_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$resume_check_dir/../paper-aligned-20261106/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$resume_check_dir/code:$PYTHONPATH"
cd /tmp
exec timeout 60 "$TGPPO_PYTHON" -u -m unittest discover -s "$resume_check_dir/tests" -v
