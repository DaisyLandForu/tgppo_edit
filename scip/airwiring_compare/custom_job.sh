#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export AIRWIRING_SCIP_PREFIX=${AIRWIRING_SCIP_PREFIX:-/usr/local/scip-8.0.4}
export LD_LIBRARY_PATH="$AIRWIRING_SCIP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
python=${AIRWIRING_PYTHON:-/home/duweiyue25/conda/envs/mamba-branching/bin/python}
case "${1:-help}" in
  prepare|summarize|help)
    exec python3 "$here/compare.py" "$@" ;;
  sssp|mamba|tgppo|smoke)
    [[ -x "$python" ]] || { echo "Missing Python: $python" >&2; exit 2; }
    exec "$python" "$here/compare.py" "$@" ;;
  check-mamba-cpu)
    shift
    [[ -x "$python" ]] || { echo "Missing Python: $python" >&2; exit 2; }
    exec "$python" "$here/checks/check_mamba_cpu.py" "$@" ;;
  *) echo "Usage: bash $0 {prepare|sssp|mamba|tgppo|summarize|smoke|check-mamba-cpu|help} [options]" >&2; exit 2 ;;
esac
