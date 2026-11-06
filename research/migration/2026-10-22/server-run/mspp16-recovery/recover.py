#!/usr/bin/env python3
"""Recover only mspp16 with an explicit resource exception; originals are read-only."""
import argparse
import copy
import csv
import datetime
import fcntl
import json
import math
from pathlib import Path
import sys

LAUNCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUNCH))
import post_training as p

MATRIX = p.MATRIX + [('native', 11)]
RESOURCE = dict(instance='mspp16', address_space_limit_bytes=24*1024**3,
                workers=1, container_memory_min_bytes=32*1024**3,
                cpu_quota_min=8, watchdog_seconds=300., solver_time_limit_seconds=120.,
                actor_device='cpu', cuda_required=False)


def model_name(method, seed):
    return 'native' if method == 'native' else '{}-s{}'.format(method, seed)


def valid_rows(rows, names, seeds, mode):
    expected = {(n, s) for n in names for s in seeds}
    p.require(len(rows) == len(expected) and
              {(r['instance_name'], r['solver_seed']) for r in rows} == expected,
              'Missing or duplicate evaluation tasks')
    p.require(all(r.get('valid') and r.get('task_terminal') and r.get('worker_cpu_complete')
                  and not r.get('cuda_initialized') and r.get('mode') == mode
                  and r.get('callback_seconds', {}).get('critic', 0.) == 0.
                  and r.get('search_extraction_wall_seconds', 0.) == 0.
                  and r.get('time_limit_seconds') == 120. for r in rows),
              'Invalid task, changed solver budget, CUDA, Critic or search features')
    p.require(all(math.isfinite(r[k]) for r in rows for k in
                  ('pdi_horizon_midpoint', 'pdi_horizon_lower', 'pdi_horizon_upper',
                   'worker_cpu_seconds', 'worker_wall_seconds', 'scip_solve_time', 'nnodes')),
              'Nonfinite evaluation measurement')


def frozen_inputs(root, c, run):
    post = run/'post-training-20261023'
    audited = p.load_audit(c, post)
    path = post/'frozen_selection.json'
    if not path.exists():
        raise p.Pending('All six selections must freeze before recovery')
    frozen = p.read(path)
    p.require(frozen['protocol_sha256'] == p.fingerprint(c) and
              frozen['training_audit_sha256'] == p.sha256(post/'training_audit.json') and
              {(r['group'], r['seed']) for r in frozen['models']} == set(p.MATRIX),
              'Frozen model/protocol provenance changed')
    for row in frozen['models']:
        p.require(p.sha256(root/row['selection_path']) == row['selection_sha256'], 'Selection changed')
        for export in row['exports'].values():
            p.require(p.sha256(root/export['path']) == export['sha256'], 'Frozen Actor changed')
    return audited, frozen


def actor_for(root, frozen, method, seed):
    p.require((method, seed) in MATRIX, 'Unknown model/seed')
    if method == 'native':
        return None, 'relpscost-native'
    row = next(r for r in frozen['models'] if (r['group'], r['seed']) == (method, seed))
    return root/row['exports']['best']['path'], row['exports']['best']['sha256']


def reference(root, path):
    return dict(path=str(path.relative_to(root)), sha256=p.sha256(path))


def referenced(root, value):
    path = root/value['path']
    p.require(p.sha256(path) == value['sha256'], 'Recorded source changed: '+str(path))
    return p.read(path)


def original_snapshot(root, c, run, audited, frozen, method, seed):
    actor, key = actor_for(root, frozen, method, seed)
    base = run/'test/official66-best'/key[:16]
    # Existing experiment lock is opened read-only. Never create or edit it.
    lock_path = run/'post-training-20261023'/('.evaluate-{}-s{}.lock'.format(method, seed))
    with lock_path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise p.Pending('Original evaluation is still running: '+model_name(method, seed))
        names = list(audited['datasets']['test66'])
        identity = dict(actor_sha256=key, stack=audited['stack'], protocol=p.fingerprint(c),
                        names=names, instance_hashes={n: audited['datasets']['test66'][n]['sha256'] for n in names})
        p.require(p.read(base/'identity.json') == identity, 'Original test identity differs')
        selected, failed, manifests, jobs = {}, [], [], {}
        for directory in sorted(base.glob('attempt-*')):
            path = directory/'manifest.json'
            if not path.exists():
                continue
            manifest = p.read(path)
            p.require(manifest['identity'] == identity, 'Original attempt identity differs')
            manifests.append(reference(root, path))
            for job in manifest['jobs']:
                result = directory/(job['job_id']+'.json')
                if not result.exists():
                    raise p.Pending('Original task has no result: '+job['job_id'])
                row = p.read(result)
                p.require(row['job_id'] == job['job_id'], 'Original job/result identity differs')
                if row.get('valid') and row.get('task_terminal'):
                    selected[job['job_id']] = (row, reference(root, result))
                else:
                    p.require(job['instance_name'] == 'mspp16', 'Unexpected failed instance; refusing to skip it')
                    failed.append(dict(job_id=job['job_id'], status=row.get('status'),
                                       exitcode=row.get('exitcode'), wall_seconds=row.get('wall_seconds'),
                                       source=reference(root, result)))
                if job['instance_name'] == 'mspp16':
                    expected = p.server.bound_job(root, c, p.read(root/'tgppo/datasets/cutoffs.json'),
                        'mspp16', job['solver_seed'], seed, 'native' if actor is None else 'eval', actor,
                        job_id='test-mspp16-s{}'.format(job['solver_seed']), instances=root/'tgppo/datasets/test_instances')
                    p.require(job == expected, 'Original mspp16 input or action RNG differs')
                    jobs[job['job_id']] = job
        kept_names = [n for n in names if n != 'mspp16']
        p.require(len(kept_names) == 65 and len(selected) == 195, 'Expected exactly 195 valid original tasks')
        valid_rows([r for r, _ in selected.values()], kept_names, c['solver_seeds'], 'native' if actor is None else 'eval')
        p.require(all(r.get('address_space_limit_bytes') == c['worker_address_space_bytes'] for r, _ in selected.values()),
                  'Original worker memory allowance differs from the parent protocol')
        p.require(set(jobs) == {'test-mspp16-s{}'.format(s) for s in c['solver_seeds']} and
                  {r['job_id'] for r in failed} == set(jobs), 'Expected only the three failed mspp16 seeds')
        return dict(method=method, seed=seed, original_directory=str(base.relative_to(root)),
                    identity=reference(root, base/'identity.json'), manifests=manifests,
                    valid_sources=[dict(job_id=j, **source) for j, (_, source) in sorted(selected.items())],
                    failures=failed, retry_jobs=[jobs[j] for j in sorted(jobs)])


def load_policy(root, c, run, work):
    audited, frozen = frozen_inputs(root, c, run)
    path = work/'policy.json'
    if not path.exists():
        raise p.Pending('Run recovery prepare first; no solver has started')
    policy = p.read(path)
    p.require(policy['parent_protocol_sha256'] == p.fingerprint(c) and policy['resource_exception'] == RESOURCE and
              policy['stack'] == audited['stack'] and
              policy['frozen_selection_sha256'] == p.sha256(run/'post-training-20261023/frozen_selection.json') and
              policy['training_audit_sha256'] == p.sha256(run/'post-training-20261023/training_audit.json'),
              'Recovery resource policy or frozen inputs changed')
    p.require({(r['method'], r['seed']) for r in policy['models']} == set(MATRIX), 'Recovery matrix differs')
    for source in policy['protected_files']:
        p.require(p.sha256(root/source['path']) == source['sha256'], 'Original entrypoint changed')
    return policy, audited, frozen


def prepare(root, c, run, work):
    if (work/'policy.json').exists():
        value, _, _ = load_policy(root, c, run, work)
        print('Recovery policy already exists; no original results changed', flush=True)
        return value
    audited, frozen = frozen_inputs(root, c, run)
    p.validate_dataset(root, audited, 'test66')
    with p.lock(work/'.prepare.lock'):
        models = []
        for method, seed in MATRIX:
            models.append(original_snapshot(root, c, run, audited, frozen, method, seed))
            print('PRESERVED {}: 195 valid originals, only three mspp16 failures'.format(model_name(method, seed)), flush=True)
        protected = [LAUNCH/'post_job.sh', LAUNCH/'post_training.py', LAUNCH/'env.sh', LAUNCH/'protocol.json',
                     Path(__file__).resolve(), Path(__file__).with_name('run.sh').resolve()]
        policy = dict(schema='mspp16-resource-recovery-v1', prepared_at_utc=p.now(),
            parent_protocol_sha256=p.fingerprint(c),
            frozen_selection_sha256=p.sha256(run/'post-training-20261023/frozen_selection.json'),
            training_audit_sha256=p.sha256(run/'post-training-20261023/training_audit.json'),
            resource_exception=RESOURCE, stack=audited['stack'],
            protected_files=[reference(root, path) for path in protected], models=models,
            rule='Keep original 65 instances under 8 GiB; all seven methods rerun only mspp16 with the same 24 GiB/one-worker resource exception. Do not rewrite original attempts.')
        p.write_json(work/'policy.json', policy)
    print('RECOVERY_PREPARED: 1365 valid originals retained; exactly 21 mspp16 tasks authorized', flush=True)
    return policy


def original_rows(root, model):
    for item in [model['identity']]+model['manifests']:
        referenced(root, item)
    for item in model['failures']:
        referenced(root, item['source'])
    return [referenced(root, source) for source in model['valid_sources']]


def retry_jobs(model):
    jobs = copy.deepcopy(model['retry_jobs'])
    p.require(len(jobs) == 3 and {j['solver_seed'] for j in jobs} == {0, 1, 2} and
              all(j['instance_name'] == 'mspp16' and j['limits']['time_limit'] == 120. for j in jobs),
              'Only mspp16 with original solver seeds/budget may be retried')
    for job in jobs:
        job['address_space_limit_bytes'] = RESOURCE['address_space_limit_bytes']
    return jobs


def retry(root, c, run, work, method, seed):
    policy, audited, frozen = load_policy(root, c, run, work)
    model = next((m for m in policy['models'] if (m['method'], m['seed']) == (method, seed)), None)
    p.require(model is not None, 'Unknown model/seed')
    info = p.target_cpu(root, c, audited)
    p.require(info['effective_cpu_count'] >= RESOURCE['cpu_quota_min'] and
              (info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= RESOURCE['container_memory_min_bytes']),
              'Recovery needs the matching Gold CPU, at least 8 CPU quota and 32 GiB container memory; GPU is not used')
    p.validate_dataset(root, audited, 'test66')
    valid_rows(original_rows(root, model), [n for n in audited['datasets']['test66'] if n != 'mspp16'],
               c['solver_seeds'], 'native' if method == 'native' else 'eval')
    actor, key = actor_for(root, frozen, method, seed)
    identity = dict(parent_protocol_sha256=p.fingerprint(c), recovery_policy_sha256=p.sha256(work/'policy.json'),
                    actor_sha256=key, stack=info['stack'], resource_exception=RESOURCE)
    base = work/'tasks'/model_name(method, seed)
    with p.lock(base/'.retry.lock'):
        p.write_json(base/('preflight-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), info)
        pairs = p.server.cached_jobs(retry_jobs(model), base, identity, RESOURCE['workers'], RESOURCE['watchdog_seconds'])
        rows = [r for r, _ in pairs]
        valid_rows(rows, ['mspp16'], c['solver_seeds'], 'native' if actor is None else 'eval')
        p.require(all(r.get('address_space_limit_bytes') == RESOURCE['address_space_limit_bytes'] for r in rows),
                  'Recovered worker did not apply the 24 GiB address-space limit')
        if actor is not None:
            p.require(p.sha256(actor) == key, 'Recovery changed the frozen Actor')
        p.write_json(base/'complete.json', dict(finished_at_utc=p.now(), method=method, seed=seed, runs=3,
            recovery_policy_sha256=p.sha256(work/'policy.json'), results=reference(root, base/'selected_results.json'),
            actor_sha256=key, preflight=info))
    print('MSPP16_RECOVERED {}: 3/3 valid, original 195 untouched'.format(model_name(method, seed)), flush=True)
    if all((work/'tasks'/model_name(m['method'], m['seed'])/'complete.json').exists() for m in policy['models']):
        report(root, c, run, work)
    else:
        print('This recovery passed; other models still pending. Full66 report waits for all seven.', flush=True)


def recovered_rows(root, c, work, model):
    base = work/'tasks'/model_name(model['method'], model['seed'])
    receipt = base/'complete.json'
    if not receipt.exists():
        return None
    record = p.read(receipt)
    policy = p.read(work/'policy.json')
    original_identity = referenced(root, model['identity'])
    p.require(record['recovery_policy_sha256'] == p.sha256(work/'policy.json') and
              (record['method'], record['seed']) == (model['method'], model['seed']) and
              record['preflight']['stack'] == policy['stack'] and record['actor_sha256'] == original_identity['actor_sha256'] and
              record['results']['path'] == str((base/'selected_results.json').relative_to(root)), 'Recovery receipt identity differs')
    p.require(p.read(base/'identity.json') == dict(parent_protocol_sha256=policy['parent_protocol_sha256'],
        recovery_policy_sha256=p.sha256(work/'policy.json'), actor_sha256=record['actor_sha256'], stack=policy['stack'],
        resource_exception=RESOURCE), 'Recovery cache identity differs')
    rows = referenced(root, record['results'])
    valid_rows(rows, ['mspp16'], c['solver_seeds'], 'native' if model['method'] == 'native' else 'eval')
    p.require(all(r.get('address_space_limit_bytes') == RESOURCE['address_space_limit_bytes'] for r in rows),
              'Recovered memory allowance differs')
    return rows


def metrics(rows, names):
    def mean(key):
        return sum(sum(r[key] for r in rows if r['instance_name'] == n)/
                   sum(r['instance_name'] == n for r in rows) for n in names)/len(names)
    proofs = sum(r['proof_completed'] and r['scip_solve_time'] <= 120. for r in rows)
    par2 = sum(sum(r['scip_solve_time'] if r['proof_completed'] and r['scip_solve_time'] <= 120. else 240.
                    for r in rows if r['instance_name'] == n)/sum(r['instance_name'] == n for r in rows)
                for n in names)/len(names)
    return dict(instances=len(names), runs=len(rows), mean_pdi=mean('pdi_horizon_midpoint'),
                mean_pdi_lower=mean('pdi_horizon_lower'), mean_pdi_upper=mean('pdi_horizon_upper'),
                mean_par2=par2, mean_nodes=mean('nnodes'), proof_count=proofs, proof_rate=proofs/len(rows),
                valid_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                valid_worker_wall_seconds_sum=sum(r['worker_wall_seconds'] for r in rows),
                actor_callback_seconds=sum(r.get('callback_seconds', {}).get('actor', 0.) for r in rows))


def report(root, c, run, work):
    policy, audited, frozen = load_policy(root, c, run, work)
    recovered = {model_name(m['method'], m['seed']): recovered_rows(root, c, work, m) for m in policy['models']}
    complete = all(rows is not None for rows in recovered.values())
    names = [n for n in audited['datasets']['test66'] if complete or n != 'mspp16']
    selected, by_model = [], {}
    for model in policy['models']:
        key = model_name(model['method'], model['seed'])
        rows = original_rows(root, model)+(recovered[key] if complete else [])
        valid_rows(rows, names, c['solver_seeds'], 'native' if model['method'] == 'native' else 'eval')
        by_model[key] = rows
        selected.append(dict(method=model['method'], training_seed=None if model['method'] == 'native' else model['seed'],
                             **metrics(rows, names)))
    combined = [dict(method='native', **metrics(by_model['native'], names))]
    for method in ('B0', 'R0', 'Rjoint'):
        combined.append(dict(method=method, **metrics(by_model[method+'-s11']+by_model[method+'-s22'], names)))
    baseline = next(m['mean_pdi'] for m in combined if m['method'] == 'B0')
    for row in combined:
        row['pdi_improvement_percent_vs_B0'] = 100*(1-row['mean_pdi']/baseline) if baseline > 0 else None
    snapshot = dict(observed_at_utc=p.now(), complete66=complete, instances=len(names), instance_names=names,
        scope='full66_with_mspp16_resource_exception' if complete else 'partial65_mspp16_resource_failure_not_full66',
        recovery_policy_sha256=p.sha256(work/'policy.json'), frozen_selection_sha256=policy['frozen_selection_sha256'],
        parent_protocol_sha256=p.fingerprint(c), per_training_seed=selected, combined_training_seeds=combined,
        recovered_models=[k for k, rows in recovered.items() if rows is not None],
        original_failures=[dict(model=model_name(m['method'], m['seed']), **failure) for m in policy['models'] for failure in m['failures']],
        cost_scope='Valid worker costs shown; original failures and retry attempts retained separately. GPU allocation, queue, and missing CPU measurements require Crater accounting.',
        interpretation='Instance is the statistical unit; both training seeds are retained. B0 is the common 120s/Adam reference, not the paper-scale 3600s/AdamW reproduction. No Rprod attribution. Stage6 not started.')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = work/'reports'/stamp
    destination.mkdir(parents=True, exist_ok=False)
    p.write_json(destination/'comparison.json', snapshot)
    with (destination/'per-task.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'instance', 'solver_seed', 'pdi', 'pdi_lower', 'pdi_upper',
                         'proof_completed_within_120s', 'solve_time', 'nodes', 'address_space_limit_bytes'])
        for key, rows in by_model.items():
            for r in sorted(rows, key=lambda r: r['job_id']):
                writer.writerow([key, r['instance_name'], r['solver_seed'], r['pdi_horizon_midpoint'],
                    r['pdi_horizon_lower'], r['pdi_horizon_upper'], r['proof_completed'] and r['scip_solve_time'] <= 120.,
                    r['scip_solve_time'], r['nnodes'], r.get('address_space_limit_bytes')])
    lines = ['# '+('66 实例完整结果（mspp16 内存例外）' if complete else '65 共同实例阶段性结果（尚非完整 66）'),
             '', '原始失败日志与有效结果未改写；两个训练种子都纳入组平均。', '',
             '| 方法 | 实例数 | 平均 PDI | 相对 B0 降低 % | 平均 PAR-2 | 证明完成率 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in combined:
        improvement = row['pdi_improvement_percent_vs_B0']
        lines.append('| {} | {} | {:.6f} | {} | {:.3f} | {:.2%} |'.format(row['method'], row['instances'], row['mean_pdi'],
                     '{:.4f}'.format(improvement) if improvement is not None else 'undefined', row['mean_par2'], row['proof_rate']))
    (destination/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('COMPARISON_REPORT='+str(destination), flush=True)
    print(json.dumps(dict(complete66=complete, instances=len(names), recovered_models=snapshot['recovered_models'],
                         combined_training_seeds=combined), indent=2), flush=True)
    return snapshot


def status(root, c, run, work):
    policy, _, _ = load_policy(root, c, run, work)
    output = []
    for model in policy['models']:
        base = work/'tasks'/model_name(model['method'], model['seed'])
        valid = set()
        failed = []
        for path in base.glob('attempt-*/test-mspp16-*.json'):
            row = p.read(path)
            if row.get('valid') and row.get('task_terminal'):
                valid.add(row['job_id'])
            else:
                failed.append(dict(path=str(path), status=row.get('status'), exitcode=row.get('exitcode')))
        output.append(dict(model=model_name(model['method'], model['seed']), original_valid=195,
                           recovered_valid=len(valid), required_recovery=3,
                           completed=recovered_rows(root, c, work, model) is not None, failed_attempts=failed))
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=p.ROOT)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('command', choices=['prepare', 'retry', 'status', 'report'])
    parser.add_argument('method', nargs='?', choices=['B0', 'R0', 'Rjoint', 'native'])
    parser.add_argument('seed', nargs='?', type=int, choices=[11, 22], default=11)
    args = parser.parse_args()
    p.torch.set_num_threads(1)
    root = args.root.resolve()
    c = p.server.plan(root, LAUNCH/'protocol.json')
    run = (args.run_root or root/c['run_directory']).resolve()
    work = run/'test/mspp16-recovery-20261023'
    if args.command == 'prepare':
        prepare(root, c, run, work)
    elif args.command == 'retry':
        p.require(args.method is not None, 'retry requires a model or native')
        retry(root, c, run, work, args.method, args.seed)
    elif args.command == 'report':
        report(root, c, run, work)
    else:
        status(root, c, run, work)


if __name__ == '__main__':
    try:
        main()
    except p.Pending as exc:
        print('PENDING: '+str(exc), file=sys.stderr, flush=True)
        sys.exit(75)
    except (ValueError, RuntimeError) as exc:
        print('FAILED: '+str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
