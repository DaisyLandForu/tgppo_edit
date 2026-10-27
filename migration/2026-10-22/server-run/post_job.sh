#!/usr/bin/env bash
# Parse once; shared-file edits cannot change the tail of a running job.
{
set -euo pipefail
launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TGPPO_PYTHON=${TGPPO_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
source "$launch_dir/env.sh"
mode=${1:-}
case "$mode" in
    audit|preflight|freeze|status) [[ $# == 1 ]] || exit 2 ;;
    select|evaluate)
        if [[ "$mode" == evaluate && ${2:-} == native && $# == 2 ]]; then :
        else
            [[ $# == 3 ]] || { echo 'Usage: post_job.sh select|evaluate GROUP 11|22'; exit 2; }
            case "$2/$3" in B0/11|B0/22|R0/11|R0/22|Rjoint/11|Rjoint/22) ;; *) exit 2;; esac
        fi ;;
    *) echo 'Usage: post_job.sh audit|preflight|freeze|status OR select|evaluate GROUP SEED OR evaluate native'; exit 2 ;;
esac
cd "$RESEARCH_ROOT"
mkdir -p "$SERVER_RUN_ROOT/custom-logs"
log_file=$(mktemp "$SERVER_RUN_ROOT/custom-logs/post-${mode}-${2:-all}-s${3:-none}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee "$log_file") 2>&1
printf 'Started %s host=%s mode=%s log=%s\n' "$(date -Iseconds)" "$(hostname)" "$mode" "$log_file"
exec "$TGPPO_PYTHON" "$launch_dir/post_training.py" --root "$RESEARCH_ROOT" \
    --config "$SERVER_CONFIG" --run-root "$SERVER_RUN_ROOT" "$@"
}
