#!/usr/bin/env bash
set -euo pipefail
test66_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/python3 -u "$test66_package_dir/job.py" "$@"
