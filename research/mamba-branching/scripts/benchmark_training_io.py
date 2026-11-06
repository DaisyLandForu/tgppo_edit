#!/usr/bin/env python3
"""CPU-only input equivalence/timing probe; never performs a training update."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from formal_common import hardware, load_run, sha, stamp, write_json
from formal_data import IndexedData


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--local-view', required=True)
    parser.add_argument('--cl-samples', type=int, default=128)
    parser.add_argument('--il-windows', type=int, default=4)
    parser.add_argument('--report')
    args = parser.parse_args()
    assert args.cl_samples > 0 and args.il_windows > 0
    canonical, _, _ = load_run(args.run_dir)
    local, _, _ = load_run(args.local_view)
    for name in ['config.json', 'plan.json', 'run.json', 'dataset/manifest.json']:
        assert sha(canonical / name) == sha(local / name), name
    assert (local / 'training').resolve() == (canonical / 'training').resolve()
    report = {'status': 'passed', 'timestamp_utc': stamp(), 'hardware': hardware(),
              'canonical_run': str(canonical), 'local_view': str(local), 'probes': {},
              'scope': 'CPU input equivalence and sampled I/O only; no CUDA/resume/throughput claim',
              'script_sha256': sha(Path(__file__))}
    rng = random.Random(20261028)
    for split in ['train', 'valid']:
        for mode, count in [('cl', args.cl_samples), ('il', args.il_windows)]:
            a, b = IndexedData(canonical, split, mode), IndexedData(local, split, mode)
            try:
                assert len(a) == len(b) and len(a) > 0
                indices = [rng.randrange(len(a)) for _ in range(count)]
                started = time.monotonic()
                originals = [a[i] for i in indices]
                shared_seconds = time.monotonic() - started
                started = time.monotonic()
                cached = [b[i] for i in indices]
                local_seconds = time.monotonic() - started
                assert all(equal(x, y) for x, y in zip(originals, cached)), (split, mode)
                result = {'samples': count, 'inputs_exactly_equal': True,
                          'shared_seconds': shared_seconds, 'local_seconds': local_seconds,
                          'sampled_read_speedup': shared_seconds / local_seconds}
                report['probes'][split + '-' + mode] = result
                print(split + '-' + mode + ': ' + json.dumps(result), flush=True)
            finally:
                a.close()
                b.close()
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
