#!/usr/bin/env bash
# Read-only device diagnostics and one tiny CUDA probe before the frozen resume entry.
set -euo pipefail
health_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
health_original="$health_package_dir/../paper-resource-resume-20261109/custom_job.sh"
health_device=cuda
health_previous=""
for health_argument in "$@"; do
  if [[ "$health_previous" == --device ]]; then health_device="$health_argument"; fi
  case "$health_argument" in --device=*) health_device="${health_argument#--device=}" ;; esac
  health_previous="$health_argument"
done
if [[ "${1:-}" != run || "$health_device" == cpu ]]; then
  exec bash "$health_original" "$@"
fi
case "$health_device" in
  cuda|gpu) ;;
  *) printf 'Expected --device cuda, gpu or cpu\n' >&2; exit 2 ;;
esac
source "$health_package_dir/../paper-trainmode-20261108/env.sh"
health_log_dir="$RESEARCH_ROOT/tgppo/runs/paper-gpu-health-resume-20261113"
mkdir -p "$health_log_dir"
health_log=$(mktemp "$health_log_dir/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
exec > >(tee -a "$health_log") 2>&1
printf 'GPU startup diagnostic log: %s\n' "$health_log"
printf 'UTC=%s pod_hostname=%s node_env=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "${NODE_NAME:-unknown}"
printf 'CUDA_VISIBLE_DEVICES=%s NVIDIA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset-or-empty}" "${NVIDIA_VISIBLE_DEVICES:-unset-or-empty}"
if ! command -v nvidia-smi >/dev/null; then
  printf 'GPU diagnostic failed: nvidia-smi unavailable; resume not started.\n' >&2
  exit 69
fi
if timeout 20s nvidia-smi -L; then :; else
  health_rc=$?
  printf 'GPU listing failed (exit=%s); resume not started.\n' "$health_rc" >&2
  exit "$health_rc"
fi
# Historical aggregate ECC counters are diagnostic evidence, not a rejection rule.
if ! timeout 20s nvidia-smi --query-gpu=uuid,name,pci.bus_id,driver_version --format=csv; then
  printf 'GPU identity query unavailable; retain listing above.\n' >&2
fi
if ! timeout 20s nvidia-smi -q -d ECC,PAGE_RETIREMENT; then
  printf 'Detailed ECC query unavailable; continue to the actual CUDA probe.\n' >&2
fi
if timeout 180s "$TGPPO_PYTHON" -u - <<'PY'
import json
import torch
torch.set_num_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('No usable CUDA device; resume not started')
probe = torch.ones(1, device='cuda')
probe.add_(1)
torch.cuda.synchronize()
if probe.item() != 2:
    raise RuntimeError('CUDA probe returned incorrect result; resume not started')
print(json.dumps(dict(cuda_probe_passed=True, torch_version=torch.__version__,
    cuda_runtime=torch.version.cuda, device=torch.cuda.get_device_name(0),
    visible_devices=torch.cuda.device_count(), training_steps=0)), flush=True)
PY
then
  printf 'Tiny CUDA probe passed; forwarding unchanged arguments to the existing resume entry.\n'
else
  health_rc=$?
  printf 'CUDA startup probe failed (exit=%s); resume not started. Ask the platform to check the logged GPU UUID.\n' "$health_rc" >&2
  exit "$health_rc"
fi
exec bash "$health_original" "$@"
