#!/usr/bin/env bash
# Keep the frozen trainer unchanged; cache only its immutable inputs locally.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$ROOT/scripts/local_training.py" "$@"
