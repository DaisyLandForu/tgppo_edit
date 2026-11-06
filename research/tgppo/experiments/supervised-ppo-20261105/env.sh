#!/usr/bin/env bash
light_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RESEARCH_ROOT=${RESEARCH_ROOT:-$(cd -- "$light_package_dir/../../.." && pwd)}
export TGPPO_PYTHON=${TGPPO_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
export SCIPOPTDIR="$RESEARCH_ROOT/tgppo/solver"
export LD_LIBRARY_PATH="$SCIPOPTDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$RESEARCH_ROOT/tgppo/experiments/stage4/metrics_binding/src:$light_package_dir/code"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
