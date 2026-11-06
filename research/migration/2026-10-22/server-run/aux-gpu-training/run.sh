#!/usr/bin/env bash
# Parse once, independently of all formal experiment launchers.
{
set -euo pipefail
aux_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
research_dir=$(cd -- "$aux_dir/../../../.." && pwd)
aux_python=${AUXTRAIN_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TZ=Asia/Shanghai
aux_output="$research_dir/tgppo/runs/server6-20261022/aux-gpu-training"
mkdir -p "$aux_output"
aux_log=$(mktemp "$aux_output/launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee "$aux_log") 2>&1
printf 'Auxiliary training host=%s log=%s\n' "$(hostname)" "$aux_log"
exec env -u PYTHONPATH "$aux_python" "$aux_dir/train.py" "$@"
}
