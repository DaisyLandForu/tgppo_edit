#!/usr/bin/env bash
set -euo pipefail
server_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RESEARCH_ROOT=${RESEARCH_ROOT:-$(cd -- "$server_script_dir/../../.." && pwd)}
export SCIPOPTDIR="$RESEARCH_ROOT/tgppo/solver"
export PATH="$SCIPOPTDIR/bin:$PATH"
export LD_LIBRARY_PATH="$SCIPOPTDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$RESEARCH_ROOT/tgppo/experiments/stage4/metrics_binding/src:$RESEARCH_ROOT/tgppo/experiments/server-20261022/code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
export SERVER_CONFIG=${SERVER_CONFIG:-$server_script_dir/protocol.json}
export SERVER_RUN_ROOT=${SERVER_RUN_ROOT:-$RESEARCH_ROOT/tgppo/runs/server6-20261022}
export TGPPO_PYTHON=${TGPPO_PYTHON:-python}
