#!/usr/bin/env bash
# Entry for Crater Custom jobs; one process in the foreground, persistent logs.
set -euo pipefail
launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TGPPO_PYTHON=${TGPPO_PYTHON:-/home/duweiyue25/conda/envs/tgppo/bin/python}
source "$launch_dir/env.sh"
mode=${1:-}
case "$mode" in
    prepare|baselines|bridge|pilot) [[ $# == 1 ]] || exit 2 ;;
    train|select) [[ $# == 3 ]] || { echo 'Usage: custom_job.sh train|select B0|R0|Rjoint 11|22'; exit 2; }
        case "$2/$3" in B0/11|B0/22|R0/11|R0/22|Rjoint/11|Rjoint/22) ;; *) echo 'Outside authorized six-job matrix'; exit 2;; esac ;;
    *) echo 'Usage: custom_job.sh prepare|baselines|bridge|pilot OR train|select GROUP SEED'; exit 2 ;;
esac
[[ -x "$TGPPO_PYTHON" ]] || { echo "Missing Python: $TGPPO_PYTHON"; exit 1; }
cd "$RESEARCH_ROOT"
log_dir="$SERVER_RUN_ROOT/custom-logs"
mkdir -p "$log_dir"
log_file=$(mktemp "$log_dir/${mode}-${2:-all}-s${3:-none}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee "$log_file") 2>&1
printf 'Started %s host=%s pid=%s log=%s\n' "$(date -Iseconds)" "$(hostname)" "$$" "$log_file"
# Fail before expensive native measurements if GPU/image/resources are unsuitable.
if [[ "$mode" == prepare ]]; then "$launch_dir/run.sh" preflight --device cuda; fi
pilot() {
    "$TGPPO_PYTHON" "$RESEARCH_ROOT/tgppo/experiments/server-20261022/code/server_pilot.py" \
        --root "$RESEARCH_ROOT" --config "$SERVER_CONFIG" --run-root "$SERVER_RUN_ROOT"
}
case "$mode" in
    prepare)
        # Kernel flock is released on exit; never delete old state to unlock a job.
        exec 9>"$SERVER_RUN_ROOT/.prepare.lock"
        flock -n 9 || { echo 'Another preparation job owns this run root'; exit 1; }
        "$TGPPO_PYTHON" "$launch_dir/verify_environment.py" --root "$RESEARCH_ROOT" \
            --output "$log_file.checks.json"
        "$launch_dir/run.sh" baselines
        "$launch_dir/gpu_bridge.sh"
        pilot
        ;;
    baselines) exec "$launch_dir/run.sh" baselines ;;
    bridge) exec "$launch_dir/gpu_bridge.sh" ;;
    pilot) pilot ;;
    train) exec "$launch_dir/run.sh" train --group "$2" --seed "$3" --device cuda --resume ;;
    select) exec "$launch_dir/run.sh" select --group "$2" --seed "$3" ;;
esac
