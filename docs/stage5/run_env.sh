#!/usr/bin/env bash
source /root/research/tgppo/activate.sh
export PYTHONPATH=/root/research/tgppo/experiments/stage4/metrics_binding/src:/root/research/tgppo/experiments/stage5/code
exec python /root/research/tgppo/experiments/stage5/code/stage5.py "$@"
