#!/usr/bin/env python3
"""Real SCIP/CUDA integration checks, including CL and IL process restarts."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

from formal_common import ROOT, lock, read_json, sha, stamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', required=True)
    args = parser.parse_args()
    base = Path(args.base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'started_utc': stamp(), 'scope': 'diagnostic integration only; no test instances',
              'script_sha256': sha(Path(__file__)), 'checks': {}}

    def run(name, command, expected=0):
        log = base / (name + '.log')
        with open(log, 'w') as stream:
            result = subprocess.run(['bash', str(ROOT / 'custom_job.sh'), *command], stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == expected, (name, result.returncode, expected, str(log))
        report['checks'][name] = {'passed': True, 'exit_code': result.returncode, 'log': str(log)}
        print(name + ': passed', flush=True)

    def read_latest(directory, stage):
        marker = read_json(directory / 'training' / stage / 'latest.json')
        return torch.load(directory / marker['checkpoint'], map_location='cpu')

    def compare(a, b):
        raw, scaled = [], []
        def visit(a, b):
            if isinstance(a, torch.Tensor):
                difference = (a - b).abs().max().item()
                scale = max(1., a.abs().max().item(), b.abs().max().item())
                assert difference / scale <= 32 * torch.finfo(torch.float32).eps, (difference, scale)
                raw.append(difference)
                scaled.append(difference / scale)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a:
                    visit(a[k], b[k])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    visit(x, y)
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b, (a, b)
        visit(a, b)
        return {'max_absolute_difference': max(raw, default=0), 'max_tensor_scaled_error': max(scaled, default=0)}

    try:
        continuous, resumed = base / 'continuous', base / 'resumed'
        if not (continuous / 'run.json').exists():
            run('prepare', ['prepare', '--run-dir', str(continuous), '--config', str(ROOT / 'configs/entry-smoke.json')])
            for shard in [0, 1]:
                run(f'collect-{shard}', ['collect', '--run-dir', str(continuous), '--shard-index', str(shard),
                                        '--num-shards', '2', '--workers', '2'])
            run('build', ['build-data', '--run-dir', str(continuous)])
        if not resumed.exists():
            shutil.copytree(continuous, resumed)
            assert not (resumed / 'training').exists(), 'Clone before training'
        run('continuous-train', ['train', '--run-dir', str(continuous)])
        run('cl-stop', ['train', '--run-dir', str(resumed), '--stage', 'cl', '--stop-after-updates', '2'], 75)
        stopped = read_latest(resumed, 'cl')
        assert stopped['progress']['updates'] == 2 and stopped['progress']['cursor'] == 64
        run('cl-resume', ['train', '--run-dir', str(resumed), '--stage', 'cl'])
        run('il-stop', ['train', '--run-dir', str(resumed), '--stage', 'il', '--stop-after-updates', '1'], 75)
        assert read_latest(resumed, 'il')['progress']['updates'] == 1
        run('il-resume', ['train', '--run-dir', str(resumed)])
        for stage in ['cl', 'il']:
            a, b = read_latest(continuous, stage), read_latest(resumed, stage)
            for field in ['embedding', 'optimizer', 'rng', 'sampler_rng'] + (['sequence'] if stage == 'il' else []):
                report['checks'][f'{stage}-restart-{field}'] = compare(a[field], b[field])
            for field in ['epoch', 'cursor', 'updates', 'completed', 'best_epoch']:
                assert a['progress'][field] == b['progress'][field]
            report['checks'][f'{stage}-restart-progress'] = {'passed': True, 'progress': b['progress']}
        selected_a, selected_b = read_json(continuous / 'training/selected.json'), read_json(resumed / 'training/selected.json')
        for key in ['seed', 'cl_epoch_1based', 'il_epoch_1based', 'validation_solver_seed', 'test_used_for_selection']:
            assert selected_a[key] == selected_b[key]
        assert selected_b['frozen'] and not selected_b['test_used_for_selection']
        digest = sha(resumed / 'training/selected.json')
        run('completed-resume-idempotent', ['train', '--run-dir', str(resumed)])
        assert sha(resumed / 'training/selected.json') == digest
        # Real process holds the advisory lock; a second trainer must refuse.
        with lock(resumed / 'training/.lock'):
            run('duplicate-trainer-rejected', ['train', '--run-dir', str(resumed)], 2)
        for shard in [0, 1]:
            run(f'evaluate-{shard}', ['evaluate', '--run-dir', str(resumed), '--shard-index', str(shard),
                                     '--num-shards', '2', '--workers', '1'])
        run('summarize', ['summarize', '--run-dir', str(resumed)])
        summary = read_json(resumed / 'evaluation/summary.json')
        assert summary['status'] == 'complete' and summary['completed_pairs'] == 2
        # Re-running a collection shard must retain the original commit markers.
        before = {str(p): sha(p) for p in continuous.glob('collection/*/done.json')}
        run('collection-resume-idempotent', ['collect', '--run-dir', str(continuous), '--shard-index', '0',
                                            '--num-shards', '2', '--workers', '2'])
        assert all(sha(p) == digest for p, digest in before.items())
        for guard in ['incomplete', 'source-change', 'corrupt-data']:
            target = base / ('guard-' + guard)
            shutil.copytree(continuous, target)
            if guard == 'incomplete':
                next(target.glob('collection/*/done.json')).unlink()
                run('incomplete-data-rejected', ['build-data', '--run-dir', str(target)], 2)
            elif guard == 'source-change':
                record = read_json(target / 'run.json')
                record['sources']['code/network.py'] = '0' * 64
                write_json(target / 'run.json', record)
                run('source-change-rejected', ['train', '--run-dir', str(target)], 2)
            else:
                manifest = read_json(target / 'dataset/manifest.json')
                path = target / manifest['trajectories'][0]['path']
                with open(path, 'ab') as stream:
                    stream.write(b'corruption-check')
                run('corrupt-data-rejected', ['train', '--run-dir', str(target)], 2)
        # Microbatch accumulation uses the same logical denominator and clips
        # gradients only after accumulation. Compare a real 2-window update.
        from formal_data import IndexedData
        from formal_train import optimizer_update, seed_all
        from network import TreeGateNet, GroupedMamba
        data = IndexedData(continuous, 'train', 'il')
        items = [data[0], data[1]]
        seed_all(123)
        e1, m1 = TreeGateNet(infimum=8).cuda(), GroupedMamba(8, 100).cuda()
        e2, m2 = copy.deepcopy(e1), copy.deepcopy(m1)
        o1 = torch.optim.AdamW(list(e1.parameters()) + list(m1.parameters()), lr=.001, weight_decay=.01)
        o2 = torch.optim.AdamW(list(e2.parameters()) + list(m2.parameters()), lr=.001, weight_decay=.01)
        optimizer_update('il', e1, m1, o1, items, 2, 1.)
        optimizer_update('il', e2, m2, o2, items, 1, 1.)
        report['checks']['microbatch-embedding'] = compare(e1.state_dict(), e2.state_dict())
        report['checks']['microbatch-sequence'] = compare(m1.state_dict(), m2.state_dict())
        data.close()
        report['status'] = 'passed'
        report['summary'] = summary['geometric_means']
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    report['finished_utc'] = stamp()
    write_json(base / 'verification.json', report)
    print(json.dumps({'status': report['status'], 'report': str(base / 'verification.json')}, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
