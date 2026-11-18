#!/usr/bin/env bash
set -euo pipefail
eval_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$eval_package_dir/../paper-aligned-20261106/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$eval_package_dir:$PYTHONPATH"
exec "$TGPPO_PYTHON" -u "$eval_package_dir/runner.py" "$@"
