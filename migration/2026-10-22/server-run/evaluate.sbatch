#!/usr/bin/env bash
#SBATCH --job-name=tgppo-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=tgppo-test-%A_%a.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Export SERVER_LAUNCH_DIR}"
source "$SERVER_LAUNCH_DIR/env.sh"
METHOD_NAMES=(B0 R0 Rjoint)
TRAIN_SEEDS=(11 22)
task_index=${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}
method_name=${METHOD_NAMES[$((task_index / 2))]}
train_seed=${TRAIN_SEEDS[$((task_index % 2))]}
checkpoint_kind=${CHECKPOINT_KIND:-best}
case $checkpoint_kind in best|final) ;; *) echo 'CHECKPOINT_KIND must be best or final'; exit 2;; esac
checkpoint_path="$SERVER_RUN_ROOT/training/$method_name-s$train_seed/actor_$checkpoint_kind.pt"
exec srun "$SERVER_LAUNCH_DIR/run.sh" evaluate --checkpoint "$checkpoint_path" --seed "$train_seed" \
  --instances-dir "${TEST_INSTANCES_DIR:-$RESEARCH_ROOT/tgppo/datasets/test_instances}" \
  --names-json "${TEST_NAMES_JSON:-$SERVER_LAUNCH_DIR/test66.json}" \
  --label "${TEST_LABEL:-official66}-$checkpoint_kind"
