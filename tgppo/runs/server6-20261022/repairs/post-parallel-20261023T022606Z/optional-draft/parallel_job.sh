#!/usr/bin/env bash
# Parse the complete launcher before starting; never reread a changed tail.
{
set -euo pipefail
launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TGPPO_PYTHON=${TGPPO_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
source "$launch_dir/env.sh"
cd "$RESEARCH_ROOT"
mkdir -p "$SERVER_RUN_ROOT/custom-logs"
log_file=$(mktemp "$SERVER_RUN_ROOT/custom-logs/parallel-${1:-unknown}-${2:-all}-s${3:-none}-part${4:-none}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee "$log_file") 2>&1
printf 'Started %s host=%s log=%s\n' "$(date -Iseconds)" "$(hostname)" "$log_file"
exec "$TGPPO_PYTHON" "$launch_dir/parallel_post.py" --root "$RESEARCH_ROOT" \
    --config "$SERVER_CONFIG" --run-root "$SERVER_RUN_ROOT" "$@"
}
