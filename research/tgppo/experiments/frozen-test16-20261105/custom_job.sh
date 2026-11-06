#!/usr/bin/env bash
set -euo pipefail
superseded_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo 'The unsubmitted test16 plan is superseded: forwarding to full66. Submit all4 shards0..3; see frozen-test66-20261105/docs/CUSTOM_JOBS.md.' >&2
exec bash "$superseded_package_dir/../frozen-test66-20261105/custom_job.sh" "$@"
