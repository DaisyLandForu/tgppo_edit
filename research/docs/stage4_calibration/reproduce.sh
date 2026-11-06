#!/usr/bin/env bash
set -eo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash docs/stage4_calibration/reproduce.sh /absolute/new/output-directory'
  exit 2
fi
calibration_output=$(realpath -m "$1")
if [ -e "$calibration_output" ]; then
  echo 'Choose a new directory to preserve earlier experiments.'
  exit 2
fi
mkdir -p "$calibration_output"
source /root/research/tgppo/activate.sh
calibration_code=/root/research/tgppo/experiments/stage4-calibration/code
export PYTHONPATH=/root/research/tgppo/experiments/stage4/metrics_binding/src:"$calibration_code"
python "$calibration_code/calibrate.py" select --run-root "$calibration_output" > "$calibration_output/selection.log" 2>&1
python "$calibration_code/calibrate.py" fit --run-root "$calibration_output" > "$calibration_output/fit.log" 2>&1
python /root/research/docs/stage4_calibration/audit.py --run-root "$calibration_output" --report-dir "$calibration_output/report" > "$calibration_output/audit.log" 2>&1
env -u PYTHONPATH /root/anaconda3/bin/python /root/research/docs/stage4_calibration/plot_results.py --report-dir "$calibration_output/report"
echo "Calibration evidence saved: $calibration_output"
