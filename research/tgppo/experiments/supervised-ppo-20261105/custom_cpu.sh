#!/usr/bin/env bash
set -euo pipefail
supervised_cpu_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/python3 -u "$supervised_cpu_dir/evaluation/job.py" "$@"
