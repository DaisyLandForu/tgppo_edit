#!/usr/bin/env bash
# New standalone launcher. The original experiment launchers are untouched.
{
set -euo pipefail
recovery_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TGPPO_PYTHON=${TGPPO_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
source "$recovery_dir/../env.sh"
export PYTHONDONTWRITEBYTECODE=1
cd "$RESEARCH_ROOT"
mkdir -p "$SERVER_RUN_ROOT/custom-logs"
recovery_log=$(mktemp "$SERVER_RUN_ROOT/custom-logs/mspp16-${1:-unknown}-${2:-all}-s${3:-none}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee "$recovery_log") 2>&1
printf 'MSPP16 recovery started %s host=%s log=%s; CPU execution, no CUDA\n' "$(date -Iseconds)" "$(hostname)" "$recovery_log"
exec "$TGPPO_PYTHON" "$recovery_dir/recover.py" --root "$RESEARCH_ROOT" --run-root "$SERVER_RUN_ROOT" "$@"
}
