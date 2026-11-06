#!/usr/bin/env bash
set -euo pipefail
supervised_eval_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RESEARCH_ROOT=$(cd -- "$supervised_eval_dir/../../../.." && pwd)
source "$supervised_eval_dir/../env.sh"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="" TGPPO_SUPERVISED_EVAL="$supervised_eval_dir"
cd /tmp
exec "$TGPPO_PYTHON" -u -c 'import torch,numpy,scipy.signal,h5py,os,sys,runpy; from pathlib import Path; p=Path(os.environ["TGPPO_SUPERVISED_EVAL"]); root=p.parents[3];sys.path[:0]=[str(p),str(p.parent/"code"),str(root/"tgppo/experiments/stage4/metrics_binding/src"),str(root/"docs/methods/tgppo_light_actor_20261105/continuation")];runpy.run_path(str(p/"run.py"),run_name="__main__")'
