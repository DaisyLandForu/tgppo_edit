#!/usr/bin/env bash
set -euo pipefail
source /root/research/tgppo/activate.sh
metadata="$TGPPO_ROOT/datasets/train_instances_infos.pkl"
if [[ ! -f "$metadata" ]]; then
  echo '请先生成训练集基线（可能需要数小时）：'
  echo 'source /root/research/tgppo/activate.sh'
  echo 'python prepare_baselines.py --split train --time-limit 3600 --max-workers 2'
  exit 1
fi
python - "$metadata" <<'PY'
import pickle, sys
from pathlib import Path
with open(sys.argv[1], 'rb') as f:
    rows = pickle.load(f)
names = {p.name.split('.')[0] for p in Path('/root/research/tgppo/datasets/train_instances').glob('*.mps.gz')}
if set(rows) != names:
    raise SystemExit('训练集基线尚未生成完整；请重新运行 prepare_baselines.py，它会继续已有进度。')
PY
run_dir="$(mktemp -d "$TGPPO_ROOT/runs/train-$(date +%Y%m%d-%H%M%S)-XXXX")"
cp "$TGPPO_ROOT/configs/paper_hparams.json" "$run_dir/hparams.json"
git -C "$TGPPO_ROOT/repo" rev-parse HEAD > "$run_dir/source-commit.txt"
printf '%q ' "$@" > "$run_dir/extra-arguments.txt"
echo "结果目录：$run_dir"
cd "$run_dir"
# One iteration over all 25 training instances is the initial trial default.
# Pass --train_iterations ... / --max_workers ... to override explicitly.
python "$TGPPO_ROOT/repo/code/train.py" \
  --instances_dir "$TGPPO_ROOT/datasets/train_instances" \
  --instances_info_dict "$metadata" \
  --logs_dir "$run_dir/logs" \
  --best_params_json "$run_dir/hparams.json" \
  --output_model "$run_dir/models/model.pt" \
  --checkpoint_dir "$run_dir/checkpoints" \
  --time_limit 3600 --per_job_timeout 3900 --scip_setting sandbox \
  --seeds 0 --train_iterations 1 --checkpoint_every 1 --max_workers 2 \
  "$@" 2>&1 | tee "$run_dir/console.log"
