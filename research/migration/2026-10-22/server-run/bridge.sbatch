#!/usr/bin/env bash
#SBATCH --job-name=tgppo-bridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=tgppo-bridge-%j.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Submit with submit.sh or export SERVER_LAUNCH_DIR}"
source "$SERVER_LAUNCH_DIR/env.sh"
exec srun "$SERVER_LAUNCH_DIR/gpu_bridge.sh"
