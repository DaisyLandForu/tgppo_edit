"""Prepare measured relpscost normalization statistics for the training code.

The paper uses instance-normalized rewards. Do not substitute invented
baseline values. This command runs real SCIP solves and can take hours.
"""
import argparse
import concurrent.futures
import json
import os
import pickle
from pathlib import Path
from pyscipopt import Model
from project.utils import init_params, settings, scip_limits

ROOT = Path(__file__).resolve().parent

def solve(job):
    path, cutoff, time_limit, seed = job
    model = Model()
    init_params(model, dict(scip_limits, time_limit=time_limit), settings['sandbox'])
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.readProblem(path)
    model.setObjlimit(cutoff)
    model.optimize()
    row = dict(cutoff=cutoff, baseline_nodes=model.getNNodes(),
               baseline_gap=model.getGap(), baseline_integral=model.getPrimalDualIntegral(),
               baseline_status=str(model.getStatus()), baseline_time_limit=time_limit,
               baseline_solve_time=model.getSolvingTime(), baseline_seed=seed,
               baseline_solver='SCIP 6.0.1 / SoPlex 4.0.1 / sandbox / relpscost')
    model.freeProb()
    return Path(path).name.split('.')[0], row

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['train', 'test'], default='train')
    ap.add_argument('--instances', nargs='*', help='Optional subset, e.g. lseu')
    ap.add_argument('--time-limit', type=float, default=3600)
    ap.add_argument('--max-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.time_limit <= 0 or args.max_workers < 1:
        ap.error('time-limit and max-workers must be positive')
    if args.instances and args.output is None:
        ap.error('Specify --output for a subset to keep formal metadata separate')
    output = args.output or ROOT / 'datasets' / (args.split + '_instances_infos.pkl')
    output.parent.mkdir(parents=True, exist_ok=True)
    cutoffs = json.loads((ROOT / 'datasets/cutoffs.json').read_text())
    paths = sorted((ROOT / 'datasets' / (args.split + '_instances')).glob('*.mps.gz'))
    if args.instances:
        names = set(args.instances)
        paths = [p for p in paths if p.name.split('.')[0] in names]
        if len(paths) != len(names):
            ap.error('Some requested instances were not found')
    if not paths:
        ap.error('No instances found')
    rows = {}
    if output.exists():
        with output.open('rb') as f:
            rows = pickle.load(f)
        for row in rows.values():
            if row.get('baseline_time_limit') != args.time_limit or row.get('baseline_seed') != args.seed:
                ap.error('Existing output uses a different protocol; choose another --output')
    jobs = [(str(p), cutoffs[p.name.split('.')[0]]['cutoff'], args.time_limit, args.seed)
            for p in paths if p.name.split('.')[0] not in rows]
    print(f'{len(jobs)} baseline solves; workers={args.max_workers}; limit={args.time_limit}s', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(solve, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            name, row = future.result()
            if row['baseline_status'] not in ['optimal', 'infeasible', 'timelimit']:
                raise RuntimeError(f'Unexpected status for {name}: {row}')
            rows[name] = row
            tmp = output.with_suffix('.tmp')
            with tmp.open('wb') as f:
                pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, output)
            output.with_suffix('.json').write_text(json.dumps(rows, indent=2))
            print(name, row['baseline_status'], row['baseline_nodes'], flush=True)
    print('Saved', output, flush=True)

if __name__ == '__main__':
    main()
