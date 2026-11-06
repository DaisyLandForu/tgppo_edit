#!/usr/bin/env bash
# Stage 3 only. Whole pipeline can take tens of minutes; keeps all failures.
set -eo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash docs/stage3/reproduce.sh /absolute/new/run-directory'
  exit 2
fi
stage3_output=$(realpath -m "$1")
if [ -e "$stage3_output" ]; then
  echo 'Choose a new output directory to preserve existing evidence.'
  exit 2
fi
mkdir -p "$stage3_output"
source /root/research/tgppo/activate.sh
stage3_code=/root/research/tgppo/experiments/stage3/code
export PYTHONPATH=/root/research/tgppo/experiments/stage3/metrics_binding/src:"$stage3_code"
python "$stage3_code/stage3.py" prepare --run-root "$stage3_output" > "$stage3_output/native.log" 2>&1
python "$stage3_code/stage3.py" pilot --run-root "$stage3_output" > "$stage3_output/pilot.log" 2>&1
if [ -f "$stage3_output/pilot/learning_error.json" ] && [ -f "$stage3_output/pilot/collected-batch-01.pt" ]; then
  python "$stage3_code/repair_pilot.py" --run-root "$stage3_output" > "$stage3_output/repair.log" 2>&1 &
  stage3_repair_pid=$!
  while kill -0 "$stage3_repair_pid" 2>/dev/null; do
    if [ -f "$stage3_output/pilot-repaired/learning-progress.json" ] && [ ! -f "$stage3_output/pilot-repaired/forecast_budget_guard.json" ]; then
      python /root/research/docs/stage3/monitor_pilot_budget.py --repair --run-root "$stage3_output" > "$stage3_output/guard.log" 2>&1 || true
    fi
    sleep 2
  done
  wait "$stage3_repair_pid"
fi
python "$stage3_code/stage3.py" evaluate --run-root "$stage3_output" > "$stage3_output/evaluation.log" 2>&1
python /root/research/docs/stage3/summarize.py --run-root "$stage3_output" --report-dir "$stage3_output/report" > "$stage3_output/audit.log" 2>&1
env -u PYTHONPATH /root/anaconda3/bin/python /root/research/docs/stage3/plot_results.py --run-root "$stage3_output" --report-dir "$stage3_output/report"
echo "Stage-3 artifacts saved: $stage3_output"
