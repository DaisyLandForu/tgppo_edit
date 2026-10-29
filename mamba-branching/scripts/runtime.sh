#!/usr/bin/env bash
# Source in launchers; direct execution prints the isolated Python path.
set -euo pipefail
MAMBA_ROOT="${MAMBA_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MAMBA_ENV="${MAMBA_ENV:-/home/duweiyue25/conda/envs/mamba-branching}"
MAMBA_PYTHON="$MAMBA_ENV/bin/python"
export MAMBA_ROOT MAMBA_ENV MAMBA_PYTHON
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ ! -x "$MAMBA_PYTHON" ]]; then
  echo "Missing isolated environment: $MAMBA_ENV" >&2
  exit 2
fi
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' "$MAMBA_PYTHON"
fi
