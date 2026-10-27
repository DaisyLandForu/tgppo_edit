#!/usr/bin/env bash
set -euo pipefail
export SERVER_LAUNCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SERVER_LAUNCH_DIR/env.sh"
case ${1:-} in
  baselines)
    sbatch "${@:2}" "$SERVER_LAUNCH_DIR/baselines.sbatch"
    ;;
  bridge)
    sbatch "${@:2}" "$SERVER_LAUNCH_DIR/bridge.sbatch"
    ;;
  evaluate)
    sbatch "${@:2}" "$SERVER_LAUNCH_DIR/evaluate.sbatch"
    ;;
  train)
    sbatch "${@:2}" "$SERVER_LAUNCH_DIR/train.sbatch"
    ;;
  select)
    sbatch "${@:2}" "$SERVER_LAUNCH_DIR/select.sbatch"
    ;;
  *)
    echo 'Usage: submit.sh {baselines|bridge|train|select|evaluate} [sbatch options such as --partition, --account, --dependency]'
    exit 2
    ;;
esac
