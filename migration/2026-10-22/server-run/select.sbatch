#!/usr/bin/env bash
#SBATCH --job-name=tgppo-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=tgppo-dev-%A_%a.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Submit with submit.sh or export SERVER_LAUNCH_DIR}"
source "$SERVER_LAUNCH_DIR/env.sh"
METHOD_NAMES=(B0 R0 Rjoint)
TRAIN_SEEDS=(11 22)
task_index=${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}
method_name=${METHOD_NAMES[$((task_index / 2))]}
train_seed=${TRAIN_SEEDS[$((task_index % 2))]}
exec srun "$SERVER_LAUNCH_DIR/run.sh" select --group "$method_name" --seed "$train_seed"
