#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scripts/runtime.sh"
case "${1:-}" in
  environment-check)
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/verify_environment.py"
    ;;
  cpu-preflight)
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/preflight.py" cpu
    ;;
  gpu-preflight)
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/preflight.py" gpu
    ;;
  gpu-ready)
    "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/verify_environment.py"
    "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/preflight.py" cpu
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/preflight.py" gpu
    ;;
  collection-pilot)
    shift
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/collection_pilot.py" "$@"
    ;;
  data-pilot)
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/data_pilot.py"
    ;;
  gpu-training-pilot)
    shift
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/gpu_training_pilot.py" "$@"
    ;;
  prepare|collect|build-data|evaluate|summarize|status)
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/formal_jobs.py" "$@"
    ;;
  train)
    shift
    exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/formal_train.py" "$@"
    ;;
  *)
    echo "Usage: bash $0 {prepare|collect|build-data|train|evaluate|summarize|status|environment-check|cpu-preflight|gpu-preflight|gpu-ready|collection-pilot|data-pilot|gpu-training-pilot} [options]" >&2
    exit 2
    ;;
esac
