#!/usr/bin/env bash
# Stage-2 correctness verification only; deliberately no formal experiment matrix.
set -eo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash /root/research/docs/stage2/reproduce.sh /absolute/new/output-directory'
  exit 2
fi
stage2_output=$(realpath -m "$1")
if [ -e "$stage2_output" ]; then
  echo 'Output directory already exists; choose a new one to preserve evidence.'
  exit 2
fi
mkdir -p "$stage2_output"
source /root/research/tgppo/activate.sh
stage2_code=/root/research/tgppo/experiments/stage2/code
export PYTHONPATH="$stage2_code"
python "$stage2_code/prepare_baselines.py" \
  --instance-dir /root/research/tgppo/datasets/train_instances \
  --instances lseu stein27 misc03 pp08aCUTS \
  --cutoffs /root/research/tgppo/datasets/cutoffs.json \
  --output "$stage2_output/native-short" --time-limit 2 \
  --solver-seeds 0 --workers 2 --watchdog 45 > "$stage2_output/native.log" 2>&1
python "$stage2_code/train.py" \
  --instance-dir /root/research/tgppo/datasets/train_instances \
  --instances lseu stein27 misc03 pp08aCUTS \
  --cutoffs /root/research/tgppo/datasets/cutoffs.json \
  --output "$stage2_output/train-short" --time-limit 2 \
  --solver-seeds 0 --workers 2 --watchdog 45 --training-seed 11 \
  --config /root/research/docs/stage2/short_hparams.json \
  --baseline "$stage2_output/native-short/baselines.json" \
  --batches 1 --jobs-per-batch 4 > "$stage2_output/train.log" 2>&1
python "$stage2_code/test.py" \
  --instance-dir /root/research/tgppo/datasets/train_instances \
  --instances vpm2 dcmulti \
  --cutoffs /root/research/tgppo/datasets/cutoffs.json \
  --output "$stage2_output/eval-short" --time-limit 2 \
  --solver-seeds 0 --workers 2 --watchdog 45 \
  --checkpoint "$stage2_output/train-short/latest.pt" > "$stage2_output/eval.log" 2>&1
export PYTHONPATH=/root/research/tgppo/repo/code
timeout 45s python /root/research/docs/stage2/run_upstream_bridge.py \
  --run-dir "$stage2_output" --output "$stage2_output/upstream-bridge.json" \
  > "$stage2_output/upstream-bridge.log" 2>&1
echo "Stage-2 short verification saved: $stage2_output"
