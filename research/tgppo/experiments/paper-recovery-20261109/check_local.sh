#!/usr/bin/env bash
set -euo pipefail
recovery_test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for recovery_protocol in aligned trainmode; do
  case "$recovery_protocol" in
    aligned) source "$recovery_test_dir/../paper-aligned-20261106/env.sh" ;;
    trainmode) source "$recovery_test_dir/../paper-trainmode-20261108/env.sh" ;;
  esac
  export CUDA_VISIBLE_DEVICES=""
  export PAPER_RECOVERY_TEST_PROTOCOL="$recovery_protocol"
  export PYTHONPATH="$recovery_test_dir/code:$PYTHONPATH"
  cd /tmp
  timeout 150 "$TGPPO_PYTHON" -u -m unittest discover -s "$recovery_test_dir/tests" -v
done
