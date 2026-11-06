#!/usr/bin/env bash
set -euo pipefail
test66_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RESEARCH_ROOT=$(cd -- "$test66_package_dir/../../.." && pwd)
source "$test66_package_dir/../light-actor-20261105/env.sh"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="" TGPPO_TEST66_PACKAGE="$test66_package_dir"
cd /tmp
exec "$TGPPO_PYTHON" -u -c 'import torch,numpy,scipy.signal,h5py,os,sys,runpy; from pathlib import Path; p=Path(os.environ["TGPPO_TEST66_PACKAGE"]);root=p.parents[2];sys.path[:0]=[str(p),str(root/"tgppo/experiments/light-actor-20261105/code"),str(root/"tgppo/experiments/stage4/metrics_binding/src"),str(root/"docs/methods/tgppo_light_actor_20261105/continuation")];runpy.run_path(str(p/"run.py"),run_name="__main__")'
