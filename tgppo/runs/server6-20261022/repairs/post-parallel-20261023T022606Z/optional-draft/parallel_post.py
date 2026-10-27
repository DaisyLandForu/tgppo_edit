#!/usr/bin/env python3
"""Shard the existing development/test solves without changing their protocol."""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import json
from pathlib import Path
import sys

import post_training as post

server = post.server
read = post.read
require = post.require
sha256 = post.sha256
fingerprint = post.fingerprint
write_json = post.write_json
MATRIX = post.MATRIX
METHODS = [('native', 11)] + MATRIX
SHARDS = 6


def partitions(names):
    """Order is fixed before outcomes, including the first reporting subset."""
    require(len(names) == 66 and len(set(names)) == 66, 'Expected all 66 held-out instances')
    ordered = sorted(names, key=lambda n: hashlib.sha256(
        ('tgppo-fixed-test-order-20261023/' + n).encode()).hexdigest())
    return [ordered[i:i+11] for i in range(0, 66, 11)]


def manifest(c, postdir, audited):
    value = dict(schema='parallel-post-v1', protocol_sha256=fingerprint(c),
                 training_audit_sha256=sha256(postdir/'training_audit.json'),
                 matrix=[dict(group=g, seed=s) for g, s in MATRIX],
                 milestones=c['milestones'], solver_seeds=c['solver_seeds'],
                 development=c['development'], workers_per_job=c['workers'],
                 actor_device='cpu', test_shards=partitions(list(audited['datasets']['test66'])),
                 reporting_rule='Report the longest completed prefix of predeclared shards, common to all seven models; never select instances using outcomes')
    destination = postdir/'parallel/manifest.json'
    with post.lock(postdir/'parallel/.manifest.lock'):
        if destination.exists():
            require(read(destination) == value, 'Parallel manifest changed; preserve the existing experiment')
        else:
            write_json(destination, value)
    return value


def preflight(root, c, run, postdir, audited):
    # Print diagnostics before rejecting a wrong node, rather than a bare traceback.
    info = server.preflight(root, c, 'cpu')
    print('NODE_PREFLIGHT='+json.dumps(info, sort_keys=True), flush=True)
    require(info['stack'] == audited['stack'],
            'NODE_MISMATCH: expected '+audited['stack']['cpu_model']+'; allocated '+
            info['stack']['cpu_model']+'. Request a matching Gold 6148 node. '
            'Selecting a container image or requesting a GPU alone does not set the CPU model.')
    require(info['effective_cpu_count'] >= 8, 'Request at least 8 CPU cores')
    require(info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= 32*1024**3,
            'Request at least 32 GiB RAM')
    server.baseline_bundle(root, c, run)
    post.validate_dataset(root, audited, 'development')
    return info


def training_row(audited, group, seed):
    require((group, seed) in MATRIX, 'Outside the six authorized models')
    return next(r for r in audited['jobs'] if (r['group'], r['seed']) == (group, seed))


def valid_deployment(rows, names, seeds):
    expected = {(n, s) for n in names for s in seeds}
    require(len(rows) == len(expected) and
            {(r['instance_name'], r['solver_seed']) for r in rows} == expected,
            'Incomplete or duplicated instance/solver-seed coverage')
    require(all(r.get('valid') and r.get('task_terminal') and r.get('worker_cpu_complete')
                and not r.get('cuda_initialized')
                and r.get('callback_seconds', {}).get('critic', 0.) == 0.
                and r.get('search_extraction_wall_seconds', 0.) == 0. for r in rows),
            'Invalid deployment task, CUDA, Critic, or extra search features')


def development_identity(c, audited, row, episode):
    return dict(checkpoint_sha256=row['checkpoints'][str(episode)]['sha256'],
                stack=audited['stack'], protocol=fingerprint(c))


def development_rows(c, run, audited, group, seed):
    row = training_row(audited, group, seed)
    out = server.group_dir(run, group, seed)
    result = {}
    for episode in c['milestones']:
        base = out/'development'/('episode-{:04d}'.format(episode))
        if not (base/'selected_results.json').exists():
            raise post.Pending('{}-s{} milestone {} not complete'.format(group, seed, episode))
        require(read(base/'identity.json') == development_identity(c, audited, row, episode),
                'Development cache identity differs')
        rows = read(base/'selected_results.json')
        valid_deployment(rows, c['development'], c['solver_seeds'])
        result[episode] = rows
    return result


def finalize_one(root, c, run, postdir, audited, group, seed):
    out = server.group_dir(run, group, seed)
    if (postdir/'frozen_selection.json').exists():
        post.freeze(root, c, run, postdir)
        return
    # This operation reads verified Gold-node evidence; it does not run SCIP.
    # It is therefore also safe in the CPU-mismatched development container.
    with post.lock(out/'.selection.lock'):
        allrows = development_rows(c, run, audited, group, seed)
        row = training_row(audited, group, seed)
        if not (out/'selection_verified.json').exists():
            post.validate_training_files(root, row)
            best, scores = server.selection(allrows, c['development'], c['solver_seeds'])
            value = dict(rule='minimum instance-equal dev mean 120s PDI; exact ties favor earlier episode',
                         best=best, all_checkpoints=scores, selection_set='frozen development only',
                         selection_instances=c['development'], solver_seeds=c['solver_seeds'],
                         training_seed=seed, checkpoint_sha256=row['checkpoints'][str(best['episode'])]['sha256'],
                         protocol_sha256=fingerprint(c), stack=audited['stack'])
            if (out/'selection.json').exists():
                require(read(out/'selection.json') == value, 'Existing selection differs; refusing overwrite')
            else:
                write_json(out/'selection.json', value)
            for kind, episode in [('best', best['episode']), ('final', 500)]:
                target = out/('actor_'+kind+'.pt')
                if not target.exists():
                    server.export_actor(out/('episode-{:04d}.pt'.format(episode)), target)
        selected = post.selected_record(root, c, run, audited, group, seed)
        if not (out/'selection_verified.json').exists():
            write_json(out/'selection_verified.json', dict(verified_at_utc=post.now(), **selected))
        print('MODEL_SELECTED {}-s{} episode={}'.format(group, seed, selected['best']['episode']), flush=True)


def finalize(root, c, run, postdir, audited):
    pending = []
    for group, seed in MATRIX:
        try:
            finalize_one(root, c, run, postdir, audited, group, seed)
        except post.Pending as exc:
            pending.append(str(exc))
    if pending:
        raise post.Pending('; '.join(pending))
    return post.freeze(root, c, run, postdir)


def select_shard(root, c, run, postdir, audited, group, seed, episode):
    require(episode in c['milestones'], 'Unknown milestone')
    row = training_row(audited, group, seed)
    info = preflight(root, c, run, postdir, audited)
    if (postdir/'frozen_selection.json').exists():
        post.freeze(root, c, run, postdir)
        print('Selection is already frozen; verified without rerunning solves', flush=True)
        return
    out = server.group_dir(run, group, seed)
    base = out/'development'/('episode-{:04d}'.format(episode))
    checkpoint = out/('episode-{:04d}.pt'.format(episode))
    # Shared group lock prevents an old six-job selector from using the same cache.
    with post.lock(out/'.selection.lock', shared=True), post.lock(base/'.parallel.lock'):
        require(sha256(checkpoint) == row['checkpoints'][str(episode)]['sha256'], 'Training checkpoint changed')
        cutoffs = read(root/'tgppo/datasets/cutoffs.json')
        jobs = [server.bound_job(root, c, cutoffs, name, solver_seed, seed, 'eval', checkpoint,
                                 job_id='dev-{}-s{}'.format(name, solver_seed))
                for name in c['development'] for solver_seed in c['solver_seeds']]
        pairs = server.cached_jobs(jobs, base, development_identity(c, audited, row, episode),
                                   c['workers'], c['watchdog'])
        valid_deployment([r for r, _ in pairs], c['development'], c['solver_seeds'])
        require(sha256(checkpoint) == row['checkpoints'][str(episode)]['sha256'], 'Evaluation mutated checkpoint')
        write_json(base/'parallel_verified.json', dict(finished_at_utc=post.now(), preflight=info,
                   results_sha256=sha256(base/'selected_results.json'), runs=len(pairs)))
    print('DEVELOPMENT_SHARD_COMPLETE {}-s{} episode={} runs=15'.format(group, seed, episode), flush=True)
    try:
        finalize_one(root, c, run, postdir, audited, group, seed)
        post.freeze(root, c, run, postdir)
    except post.Pending as exc:
        # A completed shard is successful even while siblings/other models run.
        print('SHARD_SUCCESS; selection/freeze still pending: '+str(exc), flush=True)


def evaluation_path(run, method, seed, shard):
    label = 'native' if method == 'native' else '{}-s{}'.format(method, seed)
    return run/'post-training-20261023/parallel/test'/label/('shard-{:02d}'.format(shard))


def evaluate_shard(root, c, run, postdir, audited, plan, method, seed, shard):
    require((method, seed) in METHODS and 0 <= shard < SHARDS, 'Unknown model or test shard')
    # Gate the entire held-out set on all six complete, frozen dev selections.
    frozen = post.freeze(root, c, run, postdir)
    info = preflight(root, c, run, postdir, audited)
    post.validate_dataset(root, audited, 'test66')
    native = method == 'native'
    actor = None
    if not native:
        row = next(r for r in frozen['models'] if (r['group'], r['seed']) == (method, seed))
        actor = root/row['exports']['best']['path']
    names = plan['test_shards'][shard]
    base = evaluation_path(run, method, seed, shard)
    # Distinct label + directory from the old unsharded official66-best results.
    with post.lock(base/'.evaluate.lock'):
        server.evaluate(root, c, run, actor, root/'tgppo/datasets/test_instances', names,
                        'parallel66-shard{:02d}'.format(shard), seed, native)
        key = 'relpscost-native' if native else sha256(actor)
        directory = run/'test'/('parallel66-shard{:02d}'.format(shard))/key[:16]
        rows = read(directory/'selected_results.json')
        valid_deployment(rows, names, c['solver_seeds'])
        write_json(base/'complete.json', dict(finished_at_utc=post.now(), method=method,
                   training_seed=None if native else seed, shard=shard, runs=len(rows),
                   frozen_selection_sha256=sha256(postdir/'frozen_selection.json'),
                   parallel_manifest_sha256=sha256(postdir/'parallel/manifest.json'),
                   selected_results_path=str((directory/'selected_results.json').relative_to(root)),
                   selected_results_sha256=sha256(directory/'selected_results.json'), preflight=info))
        print('TEST_SHARD_COMPLETE {}-s{} shard={} runs={}'.format(method, seed, shard, len(rows)), flush=True)


def receipt_rows(root, c, run, postdir, audited, plan, method, seed, shard):
    receipt = evaluation_path(run, method, seed, shard)/'complete.json'
    if not receipt.exists():
        return None
    item = read(receipt)
    require(item['method'] == method and item['shard'] == shard and
            item['training_seed'] == (None if method == 'native' else seed) and
            item['frozen_selection_sha256'] == sha256(postdir/'frozen_selection.json') and
            item['parallel_manifest_sha256'] == sha256(postdir/'parallel/manifest.json') and
            item['preflight']['stack'] == audited['stack'], 'Evaluation receipt identity differs')
    path = root/item['selected_results_path']
    require(sha256(path) == item['selected_results_sha256'], 'Evaluation results changed')
    rows = read(path)
    valid_deployment(rows, plan['test_shards'][shard], c['solver_seeds'])
    return rows


def completed_prefix(complete):
    prefix = 0
    while prefix < SHARDS and all((m, s, prefix) in complete for m, s in METHODS):
        prefix += 1
    return prefix


def metrics(rows, names, limit):
    require(rows and names, 'Cannot summarize empty results')
    # Instance is the unit: average solver repetitions and both training seeds first.
    def instance_mean(key):
        return sum(sum(r[key] for r in rows if r['instance_name'] == n)/
                   sum(r['instance_name'] == n for r in rows) for n in names)/len(names)
    enriched = [dict(r, par2=(r['scip_solve_time'] if r['proof_completed'] and
                              r['scip_solve_time'] <= limit else 2*limit)) for r in rows]
    proofs = sum(r['proof_completed'] and r['scip_solve_time'] <= limit for r in rows)
    par2 = sum(sum(r['par2'] for r in enriched if r['instance_name'] == n)/
               sum(r['instance_name'] == n for r in enriched) for n in names)/len(names)
    return dict(instances=len(names), runs=len(rows), mean_pdi=instance_mean('pdi_horizon_midpoint'),
                mean_pdi_lower=instance_mean('pdi_horizon_lower'),
                mean_pdi_upper=instance_mean('pdi_horizon_upper'), mean_par2=par2,
                proofs=proofs, proof_rate=proofs/len(rows), mean_nodes=instance_mean('nnodes'),
                selected_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                selected_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows),
                actor_callback_seconds=sum(r.get('callback_seconds', {}).get('actor', 0.) for r in rows))


def status_report(root, c, run, postdir, audited, plan):
    complete = {}
    for method, seed in METHODS:
        for shard in range(SHARDS):
            rows = receipt_rows(root, c, run, postdir, audited, plan, method, seed, shard)
            if rows is not None:
                complete[(method, seed, shard)] = rows
    prefix = completed_prefix(complete)
    snapshot = dict(observed_at_utc=post.now(), selection_frozen=(postdir/'frozen_selection.json').exists(),
                    development_completed=sum((server.group_dir(run, g, s)/'development'/
                        ('episode-{:04d}'.format(e))/'selected_results.json').exists()
                        for g, s in MATRIX for e in c['milestones']),
                    development_required=30, test_shards_completed=len(complete), test_shards_required=42,
                    common_completed_prefix=prefix, reportable_instances=prefix*11,
                    complete_shards=[dict(method=m, seed=s, shard=k) for m, s, k in sorted(complete)])
    if prefix:
        names = sum(plan['test_shards'][:prefix], [])
        results = {(m, s): sum([complete[(m, s, k)] for k in range(prefix)], []) for m, s in METHODS}
        per_seed = [dict(method=m, training_seed=None if m == 'native' else s,
                         **metrics(rows, names, c['time_limit'])) for (m, s), rows in results.items()]
        combined = [dict(method='native', **metrics(results[('native', 11)], names, c['time_limit']))]
        for method in ('B0', 'R0', 'Rjoint'):
            combined.append(dict(method=method, **metrics(results[(method, 11)]+results[(method, 22)], names, c['time_limit'])))
        reference = next(r for r in combined if r['method'] == 'B0')['mean_pdi']
        for item in combined:
            item['pdi_improvement_percent_vs_B0'] = (100*(1-item['mean_pdi']/reference) if reference > 0 else None)
        snapshot.update(instance_names=names, per_training_seed=per_seed, combined_training_seeds=combined,
                        scope='full66' if prefix == SHARDS else 'predeclared partial test; not full66',
                        cost_scope='Selected valid worker costs only; failed attempts, queue and allocated GPU occupancy must also be reported from raw attempts and Crater',
                        baseline_scope='B0: common 120s/Adam reference, not the paper 3600s/AdamW reproduction; no Rprod attribution')
        destination = postdir/'parallel/reports'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ.json')
        write_json(destination, snapshot)
        print('COMPARISON_REPORT='+str(destination), flush=True)
    print(json.dumps(snapshot, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=post.ROOT)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('protocol.json'))
    parser.add_argument('--run-root', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'preflight', 'finalize', 'status'):
        sub.add_parser(command)
    select = sub.add_parser('select')
    select.add_argument('method', choices=['B0', 'R0', 'Rjoint'])
    select.add_argument('seed', type=int, choices=[11, 22])
    select.add_argument('episode', type=int, choices=[100, 200, 300, 400, 500])
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('method', choices=['B0', 'R0', 'Rjoint', 'native'])
    evaluate.add_argument('seed', type=int, choices=[11, 22])
    evaluate.add_argument('shard', type=int, choices=list(range(SHARDS)))
    args = parser.parse_args()
    post.torch.set_num_threads(1)
    root = args.root.resolve()
    c = server.plan(root, args.config)
    run = (args.run_root or root/c['run_directory']).resolve()
    postdir = run/'post-training-20261023'
    audited = post.load_audit(c, postdir)
    plan = manifest(c, postdir, audited)
    if args.command == 'plan':
        print(json.dumps(plan, indent=2), flush=True)
    elif args.command == 'preflight':
        info = preflight(root, c, run, postdir, audited)
        for group, seed in MATRIX:
            out = server.group_dir(run, group, seed)
            row = training_row(audited, group, seed)
            for episode in c['milestones']:
                require(sha256(out/('episode-{:04d}.pt'.format(episode))) ==
                        row['checkpoints'][str(episode)]['sha256'], 'Audited checkpoint changed')
            policy, payload = post.load_actor(out/'episode-0500.pt')
            require(not hasattr(policy, 'critic_network') and not post.torch.cuda.is_initialized(),
                    'Actor deployment must remain CPU-only without Critic')
            del policy, payload
        write_json(postdir/'parallel/preflights'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                   dict(passed=True, observed_at_utc=post.now(), preflight=info, actor_loads=6, audited_milestones=30))
        print('PREFLIGHT_PASSED: Gold 6148, quota, shared paths, baseline, 30 checkpoints and six Actor loads; zero solves', flush=True)
    elif args.command == 'select':
        select_shard(root, c, run, postdir, audited, args.method, args.seed, args.episode)
    elif args.command == 'finalize':
        finalize(root, c, run, postdir, audited)
    elif args.command == 'evaluate':
        evaluate_shard(root, c, run, postdir, audited, plan, args.method, args.seed, args.shard)
    else:
        status_report(root, c, run, postdir, audited, plan)


if __name__ == '__main__':
    try:
        main()
    except post.Pending as exc:
        print('PENDING: '+str(exc), file=sys.stderr, flush=True)
        sys.exit(75)
    except (ValueError, RuntimeError) as exc:
        print('PRECHECK_OR_EXPERIMENT_FAILED: '+str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
