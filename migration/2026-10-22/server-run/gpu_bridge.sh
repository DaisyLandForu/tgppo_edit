#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$TGPPO_PYTHON" "$RESEARCH_ROOT/tgppo/experiments/server-20261022/code/server_bridge.py" --root "$RESEARCH_ROOT" --config "$SERVER_CONFIG" --run-root "$SERVER_RUN_ROOT"
