#!/bin/bash
set -euo pipefail
source /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=""
cd /tmp
exec "$TGPPO_PYTHON" -u -c 'import torch,numpy,scipy.signal,h5py,sys,runpy; sys.path[:0]=["/home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/code","/home/duweiyue25/tgppo-transfer/research/tgppo/experiments/stage4/metrics_binding/src","/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation"]; runpy.run_path("/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation/cpu_refinalize_cached.py",run_name="__main__")'
