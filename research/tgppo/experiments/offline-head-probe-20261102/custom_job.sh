#!/usr/bin/env bash
# One foreground command per Custom job; read the whole block before execution.
{
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
probe_python=${TGPPO_HEAD_PROBE_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
[[ -x "$probe_python" ]] || { echo "Missing Python: $probe_python"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='' TZ=Asia/Shanghai
unset PYTHONPATH
mode=${1:-}
case "$mode" in
  tests)
    [[ $# == 1 ]] || exit 2
    exec "$probe_python" -m unittest discover -s "$probe_dir/tests" -p 'test_*.py' -v
    ;;
  prepare|preflight|status|summarize) shift ;;
  run)
    [[ $# -ge 2 ]] || { echo 'Usage: custom_job.sh run original-s11|linear-s11|original-s22|linear-s22'; exit 2; }
    probe_job=$2;shift 2
    case "$probe_job" in original-s11|linear-s11|original-s22|linear-s22) ;; *) echo 'Job is outside the four-job matrix';exit 2;; esac
    ;;
  *) echo 'Usage: custom_job.sh tests|prepare|preflight|status|summarize OR run JOB_ID';exit 2 ;;
esac
probe_logs="$probe_dir/launcher-logs"
mkdir -p "$probe_logs"
probe_log=$(mktemp "$probe_logs/${mode}-${probe_job:-all}-$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
exec > >(tee "$probe_log") 2>&1
printf 'Mode=%s Job=%s Log=%s\n' "$mode" "${probe_job:-none}" "$probe_log"
if [[ "$mode" == run ]]; then
  exec "$probe_python" "$probe_dir/code/runner.py" run --job "$probe_job" "$@"
else
  exec "$probe_python" "$probe_dir/code/runner.py" "$mode" "$@"
fi
}
