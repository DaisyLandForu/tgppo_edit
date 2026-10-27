#!/usr/bin/env bash
# Stage 4 only. Uses the hashed stage-3 checkpoint and native baselines.
set -eo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash docs/stage4/reproduce.sh /absolute/new/run-directory'
  exit 2
fi
stage4_runroot=$(realpath -m "$1")
if [ -e "$stage4_runroot" ]; then
  echo 'Choose a new output directory to preserve existing evidence.'
  exit 2
fi
mkdir -p "$stage4_runroot"
source /root/research/tgppo/activate.sh
stage4_worktree=/root/research/tgppo/experiments/stage4
export PYTHONPATH="$stage4_worktree/metrics_binding/src:$stage4_worktree/code"
for stage4_command in bridge collect cache fit; do
  python "$stage4_worktree/code/stage4.py" "$stage4_command" --run-root "$stage4_runroot" > "$stage4_runroot/$stage4_command.log" 2>&1
done
python /root/research/docs/stage4/summarize.py --run-root "$stage4_runroot" --report-dir "$stage4_runroot/report" > "$stage4_runroot/audit.log" 2>&1
python /root/research/docs/stage4/value_scale_diagnostics.py --run-root "$stage4_runroot" --report-dir "$stage4_runroot/report" > "$stage4_runroot/value-scale.log" 2>&1
env -u PYTHONPATH /root/anaconda3/bin/python /root/research/docs/stage4/plot_results.py --report-dir "$stage4_runroot/report"
echo "Stage-4 artifacts saved: $stage4_runroot"
