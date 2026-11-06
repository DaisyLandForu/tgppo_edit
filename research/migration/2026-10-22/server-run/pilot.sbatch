#!/usr/bin/env bash
#SBATCH --job-name=tgppo-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=tgppo-pilot-%j.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Submit with submit.sh}"
exec srun bash "$SERVER_LAUNCH_DIR/custom_job.sh" pilot
