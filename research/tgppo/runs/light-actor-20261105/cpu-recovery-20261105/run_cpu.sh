#!/bin/bash
set -euo pipefail
source /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/env.sh
export CUDA_VISIBLE_DEVICES=""
exec "$TGPPO_PYTHON" /home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation/cpu_refinalize.py
