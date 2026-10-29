#!/usr/bin/env python3
"""Prepare, collect, index and evaluate a resumable single-seed Custom run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import h5py
import numpy as np

from formal_common import (ROOT, atomic_file, hardware, load_run, lock, prepare, read_json,
                           sha, stamp, task_done, write_json)

STOP = threading.Event()
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()


def request_stop(*_):
    STOP.set()
    with ACTIVE_LOCK:
        for process in ACTIVE:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


def instance_path(name):
    row = next(row for row in read_json(ROOT / 'data/milp-s/manifest.json') if row['name'] == name)
    path = ROOT / row['path']
    assert sha(path) == row['sha256'], name
    return path


def solver(name, seed, limits, cutoff):
    import pyscipopt as scip
    from utils import init_params
    model = scip.Model()
    assert abs(model.version() - 8.04) < .001
    init_params(model, {'node_limit': limits['node_limit'], 'time_limit': float(limits['time_limit_seconds'])},
                {'heuristics': False, 'conflict_usesb': False, 'probing_bounds': False,
                 'checksol': False, 'reevalage': 0})
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.readProblem(str(instance_path(name)))
    if cutoff is not None:
        model.setObjlimit(cutoff)
    return model


def statistics(model, branch_name, started):
    return {'solver_status': model.getStatus(), 'nodes': model.getNNodes(),
            'fair_nodes': model.getFairNNodes(branch_name.encode()), 'nodes_left': model.getNNodesLeft(),
            'gap': model.getGap(), 'primal_bound': model.getPrimalbound(), 'dual_bound': model.getDualbound(),
            'solutions': model.getNSols(), 'primal_dual_integral': model.getPrimalDualIntegral(),
            'scip_seconds': model.getSolvingTime(), 'wall_seconds': time.monotonic() - started}


def collect_worker(directory, cfg, task, attempt):
    from env import SCIPCollectBrancher
    from formal_data import TrajectorySink
    import random
    import pyscipopt as scip

    class NullLog:
        def append(self, value):
            pass

    random.seed(task['solver_seed'])
    np.random.seed(task['solver_seed'])
    model = solver(task['instance'], task['solver_seed'], cfg['collection'], None)
    sink = TrajectorySink(attempt / 'trajectory.h5')
    brancher = SCIPCollectBrancher(model, 'random', 'relpscost', task['random_prefix_steps'],
                                  {'var_dim': 25, 'node_dim': 8, 'mip_dim': 53}, False)
    brancher.collect_dict = sink
    brancher.nnodes_list, brancher.nnodesleft_list = NullLog(), NullLog()
    model.includeBranchrule(brancher, 'MambaCollection', 'stream original expert states', 999999, -1, 1)
    try:
        started = time.monotonic()
        model.optimize()
        stats = statistics(model, 'MambaCollection', started)
        allowed = ['optimal', 'timelimit'] + (['nodelimit'] if cfg['profile'] == 'diagnostic' else [])
        assert stats['solver_status'] in allowed, stats
        assert len(sink) == brancher.collect_count
        total = len(sink)
    finally:
        sink.close()
        model.freeProb()
    with h5py.File(attempt / 'trajectory.h5', 'r') as file:
        assert len(file['states']) == total
        eligible = int(np.count_nonzero(file['counts'][:] > 1))
    return {'samples': total, 'eligible_states': eligible, 'solver': stats,
            'cutoff': None, 'primal_heuristics': False,
            'trajectory': str((attempt / 'trajectory.h5').relative_to(directory))}


def evaluation_tasks(cfg):
    return [{'id': f'{name}-s{seed}', 'instance': name, 'solver_seed': seed}
            for name in cfg['evaluation']['instances'] for seed in cfg['evaluation']['solver_seeds']]


def evaluate_worker(directory, cfg, task, attempt):
    import torch
    import pyscipopt as scip
    from env import MambaEvalBrancher
    from network import GroupedMamba, TreeGateNet

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    selection_path = directory / 'training/selected.json'
    selected = read_json(selection_path)
    assert selected['frozen'] is True
    assert selected['config_sha256'] == sha(directory / 'config.json')
    assert selected['dataset_sha256'] == sha(directory / 'dataset/manifest.json')
    assert sha(directory / selected['checkpoint']) == selected['checkpoint_sha256']
    checkpoint = torch.load(directory / selected['checkpoint'], map_location='cpu')
    assert checkpoint['config_sha256'] == selected['config_sha256']
    assert checkpoint['dataset_sha256'] == selected['dataset_sha256']
    embedding, sequence = TreeGateNet(infimum=8).cuda(), GroupedMamba(8, 100).cuda()
    embedding.load_state_dict(checkpoint['embedding'])
    sequence.load_state_dict(checkpoint['sequence'])
    embedding.eval()
    sequence.eval()
    cutoff = read_json(ROOT / 'data/milp-s/cutoffs.json')[task['instance']]['cutoff'] if cfg['evaluation']['cutoff'] else None
    output = {'hardware': {**hardware(), 'gpu': torch.cuda.get_device_name()},
              'selection_sha256': sha(selection_path), 'cutoff': cutoff, 'policies': {}}

    class Expert(scip.Branchrule):
        def branchexeclp(self, allowaddcons):
            return {'result': self.model.executeBranchRule('relpscost', allowaddcons)}

    # Both methods run serially on this SAME CPU/host. A failed pair is retried
    # together, so an old baseline from a different host is never paired silently.
    for policy in ['relpscost', 'mamba']:
        model = solver(task['instance'], task['solver_seed'], cfg['evaluation'], cutoff)
        branch_name = 'MambaFormalEval'
        brancher = Expert() if policy == 'relpscost' else MambaEvalBrancher(
            model, 'cuda:0', embedding, sequence, {'var_dim': 25, 'node_dim': 8, 'mip_dim': 53}, False, 49, True, 1)
        model.includeBranchrule(brancher, branch_name, policy, 999999, -1, 1)
        try:
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                model.optimize()
            torch.cuda.synchronize()
            result = statistics(model, branch_name, started)
            assert result['solver_status'] in ['optimal', 'infeasible', 'timelimit', 'nodelimit']
            result['cutoff_proof_without_incumbent'] = bool(
                cutoff is not None and result['solver_status'] == 'infeasible' and result['solutions'] == 0)
            if policy == 'mamba':
                result['branch_calls'] = brancher.branchexec_count
                result['branch_count'] = brancher.branch_count
            output['policies'][policy] = result
            write_json(attempt / (policy + '.json'), result)
        finally:
            model.freeProb()
    return output


def worker(args):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    directory, cfg, _ = load_run(args.run_dir)
    plan = read_json(directory / 'plan.json') if args.kind == 'collection' else evaluation_tasks(cfg)
    task = next(t for t in plan if t['id'] == args.task_id)
    attempt = Path(args.attempt)
    result = {'status': 'failed', 'task': task, 'hardware': hardware(), 'started_utc': stamp()}
    try:
        with lock(directory / args.kind / task['id'] / '.lock'):
            if task_done(directory, task, args.kind):
                return 0
            result.update(collect_worker(directory, cfg, task, attempt) if args.kind == 'collection'
                          else evaluate_worker(directory, cfg, task, attempt))
            result['status'] = 'passed'
            write_json(attempt / 'result.json', result)
            files = [attempt / 'result.json']
            if args.kind == 'collection':
                files.append(attempt / 'trajectory.h5')
            else:
                files.extend([attempt / 'relpscost.json', attempt / 'mamba.json'])
            marker = {'status': 'passed', 'task': task, 'result': str((attempt / 'result.json').relative_to(directory)),
                      'artifacts': {str(p.relative_to(directory)): sha(p) for p in files}}
            write_json(directory / args.kind / task['id'] / 'done.json', marker)
    except Exception:
        result['error'] = traceback.format_exc()
        write_json(attempt / 'result.json', result)
        print(result['error'], file=sys.stderr, flush=True)
    return 0 if result['status'] == 'passed' else 2


def run_shard(args, kind):
    directory, cfg, _ = load_run(args.run_dir)
    assert 0 <= args.shard_index < args.num_shards
    if kind == 'evaluation':
        assert args.workers == 1, 'One solver at a time per evaluation GPU'
        assert (directory / 'training/selected.json').exists(), 'Train and freeze selection first'
    quota = hardware()['cgroup'].get('cpu.max', 'max 100000').split()
    if quota[0] != 'max':
        assert args.workers <= max(1, int(quota[0]) // int(quota[1])), 'Workers exceed CPU quota'
    tasks = read_json(directory / 'plan.json') if kind == 'collection' else evaluation_tasks(cfg)
    tasks = tasks[args.shard_index::args.num_shards]
    job_stamp = stamp()
    report = {'status': 'running', 'kind': kind, 'shard_index': args.shard_index, 'num_shards': args.num_shards,
              'workers': args.workers, 'started_utc': job_stamp, 'hardware': hardware(), 'results': []}
    job_report = directory / 'job_reports' / f'{kind}-{args.shard_index}-{job_stamp}.json'

    def run_one(task):
        if STOP.is_set():
            return {'id': task['id'], 'status': 'not_started'}
        try:
            if task_done(directory, task, kind):
                return {'id': task['id'], 'status': 'already_committed'}
            attempt = directory / kind / task['id'] / ('attempt-' + stamp())
            attempt.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(Path(__file__)), '_worker', '--run-dir', str(directory),
                       '--kind', kind, '--task-id', task['id'], '--attempt', str(attempt)]
            timeout = cfg['collection' if kind == 'collection' else 'evaluation']['time_limit_seconds']
            timeout = timeout + 300 if kind == 'collection' else 2 * timeout + 600
            with open(attempt / 'process.log', 'w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with ACTIVE_LOCK:
                    ACTIVE.add(process)
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f'Watchdog exceeded {timeout} seconds; partial attempt retained')
                finally:
                    with ACTIVE_LOCK:
                        ACTIVE.discard(process)
            assert code == 0, f'Worker exit={code}; see {attempt / "process.log"}'
            assert task_done(directory, task, kind)
            return {'id': task['id'], 'status': 'committed', 'attempt': str(attempt.relative_to(directory))}
        except Exception:
            return {'id': task['id'], 'status': 'failed', 'error': traceback.format_exc()}

    with lock(directory / 'job_reports' / f'.{kind}-{args.shard_index}-of-{args.num_shards}.lock'):
        write_json(job_report, report)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, task) for task in tasks]
            for future in as_completed(futures):
                item = future.result()
                report['results'].append(item)
                write_json(job_report, report)
                print(json.dumps(item), flush=True)
        report['status'] = 'passed' if all(r['status'] in ['committed', 'already_committed'] for r in report['results']) else 'failed'
        report['finished_utc'] = stamp()
        write_json(job_report, report)
    return 0 if report['status'] == 'passed' else 2


def build_data(args):
    directory, cfg, record = load_run(args.run_dir)
    with lock(directory / 'dataset/.lock'):
        plan = read_json(directory / 'plan.json')
        markers, missing = [], []
        for task in plan:
            marker = task_done(directory, task)
            if marker is None:
                missing.append(task['id'])
            else:
                markers.append((task, marker))
        write_json(directory / 'dataset/completeness.json', {'expected': len(plan), 'committed': len(markers),
                                                            'missing': missing, 'timestamp_utc': stamp()})
        assert not missing, f'{len(missing)} collection tasks missing; no partial formal dataset is allowed'
        trajectories = []
        for task, marker in markers:
            result = read_json(directory / marker['result'])
            path = directory / result['trajectory']
            with h5py.File(path, 'r') as file:
                counts = file['counts'][:]
                assert len(counts) == result['samples']
                indices = np.flatnonzero(counts > 1)
                # Preserve upstream: a whole trajectory of 1 eligible state is
                # omitted from IL; a 1-state tail after a full window is kept.
                windows = (len(indices) + 99) // 100 if len(indices) > 1 else 0
                for first in range(0, len(counts), 128):
                    states, labels = file['states'][first:first + 128], file['actions'][first:first + 128]
                    assert np.isfinite(states).all()
                    assert ((labels >= 0) & (labels < counts[first:first + 128])).all()
            index_path = directory / 'dataset/indices' / (task['id'] + '.npy')
            with atomic_file(index_path) as stream:
                np.save(stream, indices)
            trajectories.append({'id': task['id'], 'split': task['split'], 'path': result['trajectory'],
                                 'sha256': sha(path), 'index': str(index_path.relative_to(directory)),
                                 'index_sha256': sha(index_path), 'raw_states': len(counts),
                                 'eligible_states': len(indices), 'single_candidate_excluded': int((counts == 1).sum()),
                                 'windows': windows, 'solver_status': result['solver']['solver_status']})
        manifest = {'schema': 1, 'profile': cfg['profile'], 'run_sha256': sha(directory / 'run.json'),
                    'plan_sha256': record['plan_sha256'], 'trajectories': trajectories,
                    'splits': {split: {'trajectories': sum(t['split'] == split for t in trajectories),
                                      'cl_states': sum(t['eligible_states'] for t in trajectories if t['split'] == split),
                                      'il_windows': sum(t['windows'] for t in trajectories if t['split'] == split)}
                               for split in ['train', 'valid']}}
        assert all(row['cl_states'] and row['il_windows'] for row in manifest['splits'].values())
        target = directory / 'dataset/manifest.json'
        if target.exists():
            assert read_json(target) == manifest, 'Dataset would change; use a new run'
        else:
            write_json(target, manifest)
        print(json.dumps(manifest['splits'], indent=2), flush=True)
    return 0


def summarize(args):
    directory, cfg, _ = load_run(args.run_dir)
    rows, missing = [], []
    selected_digest = sha(directory / 'training/selected.json')
    for task in evaluation_tasks(cfg):
        marker = task_done(directory, task, 'evaluation')
        if marker is None:
            missing.append(task['id'])
            continue
        row = read_json(directory / marker['result'])
        assert row['selection_sha256'] == selected_digest
        rows.append(row)
    result = {'status': 'incomplete' if missing else 'complete', 'profile': cfg['profile'],
              'expected_pairs': len(evaluation_tasks(cfg)), 'completed_pairs': len(rows), 'missing': missing,
              'selection_sha256': selected_digest, 'rows': rows}
    if not missing:
        result['geometric_means'] = {}
        for policy in ['relpscost', 'mamba']:
            summary = {}
            for metric in ['nodes', 'fair_nodes', 'wall_seconds', 'scip_seconds', 'primal_dual_integral']:
                values = [row['policies'][policy][metric] for row in rows]
                assert all(np.isfinite(values)) and min(values) >= 0
                summary[metric] = 0. if 0 in values else math.exp(sum(math.log(v) for v in values) / len(values))
            summary['statuses'] = {status: sum(row['policies'][policy]['solver_status'] == status for row in rows)
                                   for status in sorted({row['policies'][policy]['solver_status'] for row in rows})}
            result['geometric_means'][policy] = summary
    write_json(directory / 'evaluation/summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)
    return 0 if not missing else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('prepare')
    init.add_argument('--run-dir', required=True)
    init.add_argument('--config', default=str(ROOT / 'configs/formal-milp-s.json'))
    for name in ['collect', 'evaluate']:
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--shard-index', type=int, required=True)
        p.add_argument('--num-shards', type=int, required=True)
        p.add_argument('--workers', type=int, default=1)
    for name in ['build-data', 'summarize', 'status']:
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
    p = sub.add_parser('_worker')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--kind', choices=['collection', 'evaluation'], required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--attempt', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.command == 'prepare':
        print(prepare(args.run_dir, args.config))
        return 0
    if args.command == '_worker':
        return worker(args)
    if args.command in ['collect', 'evaluate']:
        assert args.workers > 0
        return run_shard(args, 'collection' if args.command == 'collect' else 'evaluation')
    if args.command == 'build-data':
        return build_data(args)
    if args.command == 'summarize':
        return summarize(args)
    directory, cfg, _ = load_run(args.run_dir)
    for kind, tasks in [('collection', read_json(directory / 'plan.json')), ('evaluation', evaluation_tasks(cfg))]:
        done = [task['id'] for task in tasks if task_done(directory, task, kind)]
        print(json.dumps({'kind': kind, 'committed': len(done), 'expected': len(tasks),
                          'missing': [task['id'] for task in tasks if task['id'] not in done]}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
