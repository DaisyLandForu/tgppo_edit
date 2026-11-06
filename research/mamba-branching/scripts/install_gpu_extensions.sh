#!/usr/bin/env bash
# Install ABI-matched release wheels; no nvcc or visible GPU is needed to install.
set -euo pipefail
source "$(dirname -- "$0")/runtime.sh"
"$MAMBA_PYTHON" - <<'PY'
import subprocess
import sys
import torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == '2.2.2+cu118', torch.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
wheels = [
 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl',
 'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl',
]
from pathlib import Path
import os
cache = Path(os.environ['MAMBA_ROOT']) / 'setup/wheels'
artifacts = [str(cache / url.rsplit('/', 1)[-1]) if (cache / url.rsplit('/', 1)[-1]).is_file() else url for url in wheels]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', *artifacts], check=True)
import causal_conv1d_cuda
import selective_scan_cuda
from mamba_ssm import Mamba
print('CUDA extensions imported; actual GPU execution is still required.')
PY
