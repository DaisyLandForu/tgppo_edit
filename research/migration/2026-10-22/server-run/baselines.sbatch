#!/usr/bin/env bash
#SBATCH --job-name=tgppo-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=tgppo-native-%j.log
set -euo pipefail
: "${SERVER_LAUNCH_DIR:?Submit with submit.sh or export SERVER_LAUNCH_DIR}"
source "$SERVER_LAUNCH_DIR/env.sh"
exec srun "$SERVER_LAUNCH_DIR/run.sh" baselines
