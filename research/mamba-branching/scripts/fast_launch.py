#!/usr/bin/env python3
"""Stage immutable inputs, then run the audited RAM/prefetch trainer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from local_training import ROOT, assert_trainer_idle, atomic_json, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--cache-root', default='/tmp/mamba-branching-' + str(os.getuid()))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage', choices=['all', 'cl', 'il'], default='all')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    parser.add_argument('--il-microbatch', choices=['auto', '1', '2', '4', '8'], default='auto')
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--checkpoint-every-updates', type=int, default=250)
    parser.add_argument('--checkpoint-max-seconds', type=float, default=60.)
    parser.add_argument('--console-every', type=int, default=100)
    args = parser.parse_args()
    assert 0 <= args.prefetch <= 4 and args.checkpoint_every_updates > 0
    assert args.checkpoint_max_seconds > 0 and args.console_every > 0
    run = Path(args.run_dir).resolve()
    if not args.prepare_only: assert_trainer_idle(run)
    view, report = stage(run, Path(args.cache_root).resolve())
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only: return 0
    assert_trainer_idle(run)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    atomic_json(run / 'training/local_launches' / (attempt + '.json'), {**report, 'engine': 'ram-prefetch-v1'})
    # runtime.sh owns the isolated interpreter and environment sanitation.
    command = ['bash', '-c', 'source "$1/scripts/runtime.sh"; shift; exec "$MAMBA_PYTHON" "$MAMBA_ROOT/scripts/fast_train.py" "$@"',
               'fast-train', str(ROOT), '--run-dir', str(view), '--resume', 'auto']
    for name in ['stage', 'stop_after_updates', 'il_microbatch', 'prefetch', 'checkpoint_every_updates',
                 'checkpoint_max_seconds', 'console_every']:
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    os.execvp(command[0], command)


if __name__ == '__main__':
    raise SystemExit(main())
