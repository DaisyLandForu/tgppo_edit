#!/usr/bin/env bash
set -euo pipefail
device_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$device_package_dir/../paper-aligned-20261106/env.sh"
export PAPER_LAUNCH_EPOCH=$(date +%s)
if [[ "${1:-}" == run ]]; then
  device_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-device-resume-20261116/launcher-logs"
  mkdir -p "$device_log_dir"
  device_log=$(mktemp "$device_log_dir/$(date +%Y%m%dT%H%M%S)-XXXXXX.log")
  exec > >(tee -a "$device_log") 2>&1
  printf 'Device resume log: %s\n' "$device_log"
  device_seed=""
  device_previous=""
  for device_argument in "$@"; do
    if [[ "$device_previous" == --seed ]]; then device_seed="$device_argument"; fi
    case "$device_argument" in --seed=*) device_seed="${device_argument#--seed=}" ;; esac
    device_previous="$device_argument"
  done
  if [[ "$device_seed" == 11 ]]; then
    # Evidence for past ECC incidents; actual CUDA bridge remains mandatory in
    # Python before production checkpoint load/SCIP. No GPU reset or keepalive.
    timeout 20s nvidia-smi -L
    timeout 20s nvidia-smi --query-gpu=uuid,name,pci.bus_id,driver_version --format=csv || true
    timeout 20s nvidia-smi -q -d ECC,PAGE_RETIREMENT || true
  fi
fi
cd /tmp
exec "$TGPPO_PYTHON" -u "$device_package_dir/code/resume.py" "$@"
