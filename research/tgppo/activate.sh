#!/usr/bin/env bash
# Usage: source /root/research/tgppo/activate.sh
source /root/anaconda3/etc/profile.d/conda.sh
conda activate tgppo
export TGPPO_ROOT=/root/research/tgppo
export SCIPOPTDIR="$TGPPO_ROOT/solver"
export PATH="$SCIPOPTDIR/bin:$PATH"
export PYTHONPATH="$TGPPO_ROOT/repo/code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
cd "$TGPPO_ROOT"
