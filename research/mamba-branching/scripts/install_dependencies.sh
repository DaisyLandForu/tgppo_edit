#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/runtime.sh"
MAMBA_INSTALL_LOGS="$MAMBA_ROOT/setup/logs/install-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$MAMBA_INSTALL_LOGS"
printf 'Install logs: %s\n' "$MAMBA_INSTALL_LOGS"
"$MAMBA_PYTHON" -m pip install --index-url https://pypi.org/simple \
  -r "$MAMBA_ROOT/setup/requirements-base.txt" \
  > "$MAMBA_INSTALL_LOGS/pip-base.log" 2>&1
if [[ -f "$MAMBA_ROOT/setup/wheels/PySCIPOpt-4.3.0-cp310-cp310-linux_x86_64.whl" ]]; then
  "$MAMBA_PYTHON" -m pip install --no-deps \
    "$MAMBA_ROOT/setup/wheels/PySCIPOpt-4.3.0-cp310-cp310-linux_x86_64.whl" \
    > "$MAMBA_INSTALL_LOGS/pip-binding.log" 2>&1
else
  SCIPOPTDIR="${SCIPOPTDIR:-/usr/local/scip-8.0.4}" \
    "$MAMBA_PYTHON" -m pip install --no-build-isolation --no-deps \
    "$MAMBA_ROOT/setup/PySCIPOpt-4.3.0-features" \
    > "$MAMBA_INSTALL_LOGS/pip-binding.log" 2>&1
fi
if [[ -f "$MAMBA_ROOT/setup/wheels/torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl" ]]; then
  if [[ -f "$MAMBA_ROOT/setup/wheels/torch-dependencies-manifest.json" ]]; then
    "$MAMBA_PYTHON" - "$MAMBA_ROOT/setup/wheels" > "$MAMBA_INSTALL_LOGS/pip-torch.log" 2>&1 <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
files = []
for manifest in ['manifest.json', 'torch-dependencies-manifest.json']:
    for record in json.loads((root / manifest).read_text()):
        file = root / record['filename']
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == record['sha256'], file
        if manifest == 'torch-dependencies-manifest.json' or record['filename'].startswith('torch-'):
            files.append(str(file))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *files], check=True)
PY
  else
    "$MAMBA_PYTHON" -m pip install --index-url https://pypi.org/simple \
      "$MAMBA_ROOT/setup/wheels/torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl" \
      > "$MAMBA_INSTALL_LOGS/pip-torch.log" 2>&1
  fi
else
  "$MAMBA_PYTHON" -m pip install torch==2.2.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    > "$MAMBA_INSTALL_LOGS/pip-torch.log" 2>&1
fi
bash "$MAMBA_ROOT/scripts/install_gpu_extensions.sh" \
  > "$MAMBA_INSTALL_LOGS/pip-mamba.log" 2>&1
"$MAMBA_PYTHON" -m pip check > "$MAMBA_INSTALL_LOGS/pip-check.log" 2>&1
"$MAMBA_PYTHON" -m pip freeze --all > "$MAMBA_ROOT/setup/pip-freeze.txt"
conda list --prefix "$MAMBA_ENV" --explicit > "$MAMBA_ROOT/setup/conda-explicit-linux64.txt"
echo "Dependencies installed. Run cpu-preflight, then gpu-preflight on the allocated V100."
