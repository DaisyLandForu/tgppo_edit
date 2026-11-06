#!/usr/bin/env bash
#SBATCH --job-name=tgppo-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=tgppo-train-%A_%a.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Submit with submit.sh or export SERVER_LAUNCH_DIR}"
source "$SERVER_LAUNCH_DIR/env.sh"
METHOD_NAMES=(B0 R0 Rjoint)
TRAIN_SEEDS=(11 22)
task_index=${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}
method_name=${METHOD_NAMES[$((task_index / 2))]}
train_seed=${TRAIN_SEEDS[$((task_index % 2))]}
exec srun "$SERVER_LAUNCH_DIR/run.sh" train --group "$method_name" --seed "$train_seed" --device cuda --resume
