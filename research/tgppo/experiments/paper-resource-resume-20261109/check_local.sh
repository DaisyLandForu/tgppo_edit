#!/usr/bin/env bash
set -euo pipefail
resource_test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$resource_test_dir/../paper-trainmode-20261108/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$resource_test_dir/code:$resource_test_dir/../paper-recovery-20261109/code:$PYTHONPATH"
cd /tmp
exec timeout 90 "$TGPPO_PYTHON" -u -m unittest discover -s "$resource_test_dir/tests" -v
