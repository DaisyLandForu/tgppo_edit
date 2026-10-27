#!/usr/bin/env python3
"""Audit completed training, select on development, freeze, then evaluate Actors.

This entrypoint deliberately lives outside the hashed training runtime. It uses
the frozen server implementation without changing its algorithm or protocol.
"""
import argparse
from contextlib import ExitStack, contextmanager
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'tgppo/experiments/stage4/metrics_binding/src'),
                str(ROOT/'tgppo/experiments/server-20261022/code')]
import torch
import server
from project.runtime import read_checkpoint, load_actor
from project.utils.protocol import fingerprint, sha256, write_json

MATRIX = [(g, s) for g in ('B0', 'R0', 'Rjoint') for s in (11, 22)]


class Pending(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def lock(path, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Pending('Another job owns '+str(path))
        yield


def tensor_digest(state, exclude_extra=False):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if exclude_extra and key.startswith(('extra.', 'extra_projection.', 'time_projection.',
                                            'target_mean', 'target_std', 'return_')):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_collection_history(c, seed, order, batches, history):
    require(order == server.schedule(seed, c), 'Training task schedule changed')
    require(len(history) == c['max_episodes']//c['batch_episodes'], 'Missing collection/update batches')
    buffered = 0
    version = 0
    transitions = 0
    skipped = []
    rows = []
    for index, metrics in enumerate(history):
        begin = index*c['batch_episodes']
        batch = batches[begin]
        require(len(batch) == c['batch_episodes'], 'Incomplete collection batch')
        expected_ids = {'e{:05d}-{}-s{}'.format(i, order[i], c['solver_seeds'][(i//len(c['train']))%len(c['solver_seeds'])])
                        for i in range(begin, begin+c['batch_episodes'])}
        require({r['job_id'] for r in batch} == expected_ids, 'Episode coverage or solver seeds differ')
        require(all(r.get('valid') and r.get('task_terminal') and r.get('worker_cpu_complete')
                    and not r.get('cuda_initialized') and r['policy_version'] == version for r in batch),
                'Invalid, incomplete, CUDA, or stale-policy collection')
        count = sum(r['num_transitions'] for r in batch)
        buffered += count
        transitions += count
        require(metrics['episodes_seen'] == begin+c['batch_episodes'] and finite_tree(metrics), 'Malformed update metrics')
        require(metrics['num_samples'] == buffered, 'Collected transitions were lost or duplicated')
        if metrics['updated']:
            require(buffered >= 256 and metrics['num_updates'] > 0, 'Invalid effective PPO update')
            buffered = 0
            version += 1
        else:
            require(buffered < 256, 'Skipped a complete logical batch')
            skipped.append(dict(episode=metrics['episodes_seen'], retained_transitions=buffered))
        require(metrics['policy_version'] == version, 'Policy version mismatch')
        rows += batch
    return dict(episodes=len(rows), transitions=transitions, complete_ppo_updates=version,
                pending_transitions=buffered, skipped_batches=skipped,
                worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                workers_cpu_only=True, all_finite_tasks=True)


def dataset_manifest(root, c):
    downloads = {r['name']: r for r in read(root/'tgppo/datasets/download_manifest.json')}
    names = read(Path(__file__).with_name('test66.json'))
    require(len(names) == 66 and len(set(names)) == 66, 'Test manifest must contain 66 unique instances')
    cutoffs = read(root/'tgppo/datasets/cutoffs.json')
    result = {}
    for split, items in [('train', c['train']), ('development', c['development']), ('test66', names)]:
        directory = 'test_instances' if split == 'test66' else 'train_instances'
        result[split] = {}
        for name in items:
            digest = sha256(root/'tgppo/datasets'/directory/(name+'.mps.gz'))
            require(digest == downloads[name]['sha256'], 'Dataset SHA mismatch: '+name)
            if split != 'test66':
                require(digest == c['instance_hashes'][name], 'Frozen split SHA mismatch: '+name)
            record = cutoffs[name]
            require(record.get('status') == 'optimal' and record.get('source') and
                    isinstance(record.get('cutoff'), (int, float)) and math.isfinite(record['cutoff']),
                    'Missing compatible verified cutoff: '+name)
            result[split][name] = dict(sha256=digest, cutoff=record)
    seen = set()
    for split in ('train', 'development', 'test66'):
        hashes = {r['sha256'] for r in result[split].values()}
        require(not seen & hashes, 'Dataset split content overlap')
        seen |= hashes
    return result


def audit(root, c, run, post):
    destination = post/'training_audit.json'
    if destination.exists():
        previous = load_audit(c, post)
        for row in previous['jobs']:
            validate_training_files(root, row)
        print('Existing training audit remains valid:', destination, flush=True)
        return previous
    with lock(post/'.audit.lock'):
        dataset = dataset_manifest(root, c)
        baseline = read(run/'normalization/baselines.json')
        require(baseline['protocol_sha256'] == fingerprint(c), 'Normalization protocol differs')
        expected_stack = baseline['stack']
        rows = []
        initializations = {}
        for group, seed in MATRIX:
            out = server.group_dir(run, group, seed)
            with lock(out/'.train.lock', shared=True):
                summary = read(out/'summary.json')
                execution = read(out/'preflight.json')
                require(execution['stack'] == expected_stack and 'V100' in execution['gpu'], 'Training hardware/software differs')
                require(summary['status'] == 'training_complete' and summary['episodes_seen'] == 500 and
                        summary['trained_episodes'] == 500 and summary['pending_transitions'] == 0 and
                        summary['protocol_sha256'] == fingerprint(c), 'Training not complete under frozen protocol')
                history = read(out/'updates.json')
                batches = {begin: read(out/'collection'/('e{:05d}'.format(begin))/'selected_results.json')
                           for begin in range(0, 500, 4)}
                collection = check_collection_history(c, seed, read(out/'task_schedule.json'), batches, history)
                require(collection['complete_ppo_updates'] == summary['complete_ppo_updates'] and
                        collection['pending_transitions'] == 0, 'Final update/memory mismatch')
                initial = read(out/'initialization.json')
                initializations[(group, seed)] = initial
                files = {}
                checkpoints = {}
                for label, episode in [('latest', 500)]+[('episode-{:04d}'.format(n), n) for n in c['milestones']]:
                    path = out/(label+'.pt')
                    digest = sha256(path)
                    payload = read_checkpoint(path)
                    progress = payload['progress']
                    require(payload['config'] == server.params(root, c, group, seed, 'cuda'), 'Checkpoint config differs')
                    require(progress['phase'] == 'ready' and progress['next_episode'] == episode and
                            len(progress['history']) == episode//4, 'Milestone is not a stable complete boundary')
                    require(finite_tree({k: payload[k] for k in ('actor','critic','actor_optimizer','critic_optimizer')}),
                            'Nonfinite model or optimizer')
                    require(payload['actor_optimizer']['state'] and payload['critic_optimizer']['state'] and
                            len(payload['rng'].get('cuda', [])) == 1, 'Missing optimizer/CUDA RNG recovery state')
                    actor_digest = tensor_digest(payload['actor'])
                    critic_digest = tensor_digest(payload['critic'], exclude_extra=True)
                    require(actor_digest != initial['actor_fingerprint'] and
                            critic_digest != initial['common_critic_fingerprint'], 'Model parameters did not change')
                    if episode == 500:
                        require(progress['trained_episodes'] == 500 and payload['memory']['num_transitions'] == 0 and
                                payload['policy_version'] == summary['complete_ppo_updates'] and
                                progress['optimizer_steps'] == summary['optimizer_steps'], 'Final committed state differs')
                    if label == 'latest':
                        require(digest == summary['final_checkpoint_sha256'], 'Final checkpoint SHA mismatch')
                    files[str(path.relative_to(root))] = digest
                    checkpoints[str(episode) if label != 'latest' else label] = dict(
                        sha256=digest, actor_digest=actor_digest, policy_version=payload['policy_version'])
                    del payload
                require(checkpoints['latest']['actor_digest'] == checkpoints['500']['actor_digest'], 'Latest and final Actor differ')
                for name in ('summary.json','preflight.json','updates.json','task_schedule.json','initialization.json'):
                    files[str((out/name).relative_to(root))] = sha256(out/name)
                attempts = [read(p) for p in sorted((out/'attempts').glob('*.json'))]
                require(all(a.get('training_finished') and not a.get('error') for a in attempts), 'Interrupted training attempt needs cost review')
                row = dict(group=group, seed=seed, **collection, optimizer_steps=summary['optimizer_steps'],
                           wall_seconds=sum(a['attempt_wall_seconds'] for a in attempts),
                           learner_update_wall_seconds=sum(m['learning_wall_seconds'] for m in history),
                           peak_cuda_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in history),
                           peak_cuda_reserved_bytes=max(m['peak_cuda_reserved_bytes'] for m in history),
                           host=execution['host'], files=files, checkpoints=checkpoints)
                rows.append(row)
                print('Audited {}-s{}: 500 episodes, {} PPO updates, five milestones verified'.format(
                    group, seed, collection['complete_ppo_updates']), flush=True)
        for seed in (11, 22):
            for field in ('actor_fingerprint','public_torch_rng_sha256','public_cuda_rng_sha256'):
                require(len({initializations[(group, seed)][field] for group in c['groups']}) == 1, 'Paired initialization differs')
        result = dict(passed=True, observed_at_utc=now(), protocol_sha256=fingerprint(c),
                      stack=expected_stack, jobs=rows, datasets=dataset, selection_complete=False,
                      additional_compatible_dataset=None,
                      scope='training audit and input verification; development selection and held-out tests remain separate')
        write_json(destination, result)
        return result


def load_audit(c, post):
    path = post/'training_audit.json'
    if not path.exists():
        raise Pending('Run post_job.sh audit first')
    result = read(path)
    require(result.get('passed') and result['protocol_sha256'] == fingerprint(c), 'Audit or frozen runtime changed')
    require({(r['group'], r['seed']) for r in result['jobs']} == set(MATRIX), 'Audit matrix incomplete')
    return result


def validate_training_files(root, row):
    for relative, digest in row['files'].items():
        require(sha256(root/relative) == digest, 'Audited training file changed: '+relative)


def target_cpu(root, c, audited):
    info = server.preflight(root, c, 'cpu')
    print('Node preflight: '+json.dumps(info,sort_keys=True),flush=True)
    require(info['stack'] == audited['stack'], 'CPU/software differs from training: expected '+
            audited['stack']['cpu_model']+', got '+info['stack']['cpu_model']+
            '. Request a Gold 6148 node. Selecting an image or requesting a GPU alone '
            'does not guarantee the CPU model; no development solve has started.')
    require(info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= 16*1024**3, 'Request at least 16 GiB RAM')
    return info


def preflight(root, c, run, post):
    audited = load_audit(c, post)
    info = target_cpu(root,c,audited)
    require(info['effective_cpu_count'] >= 8, 'For this Custom template request at least 8 CPU cores')
    require(info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= 32*1024**3,
            'For this Custom template request at least 32 GiB RAM')
    server.baseline_bundle(root,c,run)
    validate_dataset(root,audited,'development')
    for row in audited['jobs']:
        out = server.group_dir(run,row['group'],row['seed'])
        for episode in c['milestones']:
            require((out/('episode-{:04d}.pt'.format(episode))).is_file(), 'Missing stable milestone')
        checkpoint = out/'episode-0500.pt'
        require(sha256(checkpoint) == row['checkpoints']['500']['sha256'], 'Final stable checkpoint changed')
        policy, payload = load_actor(checkpoint)
        require(not hasattr(policy,'critic_network') and finite_tree(payload['actor']) and
                not torch.cuda.is_initialized(), 'CPU Actor loading check failed')
        del policy,payload
    destination = post/'preflights'/(now().replace(':','').replace('+','_')+'.json')
    write_json(destination,dict(passed=True,observed_at_utc=now(),preflight=info,
        checked_models=6,checked_milestone_paths=30,solves_started=0,
        protocol_sha256=fingerprint(c)))
    print('PREFLIGHT_PASSED: matching Gold 6148/software, 8 CPU/32 GiB, baseline, shared paths, '
          '30 milestone paths and six final Actor loads; zero solves. '+str(destination),flush=True)


def validate_dataset(root, audited, split):
    directory = root/'tgppo/datasets'/('test_instances' if split == 'test66' else 'train_instances')
    cutoffs = read(root/'tgppo/datasets/cutoffs.json')
    for name, record in audited['datasets'][split].items():
        require(sha256(directory/(name+'.mps.gz')) == record['sha256'] and cutoffs[name] == record['cutoff'],
                'Instance/cutoff changed after audit: '+name)


def selected_record(root, c, run, audited, group, seed):
    out = server.group_dir(run, group, seed)
    if not all((out/n).exists() for n in ('selection.json','actor_best.pt','actor_final.pt')):
        raise Pending('Development selection not complete: {}-s{}'.format(group, seed))
    row = next(r for r in audited['jobs'] if (r['group'], r['seed']) == (group, seed))
    selection = read(out/'selection.json')
    require(selection['protocol_sha256'] == fingerprint(c) and selection['stack'] == audited['stack'] and
            selection['training_seed'] == seed and selection['selection_instances'] == c['development'] and
            selection['solver_seeds'] == c['solver_seeds'], 'Selection identity differs')
    allrows = {}
    for episode in c['milestones']:
        base = out/'development'/('episode-{:04d}'.format(episode))
        identity = read(base/'identity.json')
        require(identity == dict(checkpoint_sha256=row['checkpoints'][str(episode)]['sha256'],
                                 stack=audited['stack'], protocol=fingerprint(c)), 'Development cache identity differs')
        allrows[episode] = read(base/'selected_results.json')
        require(all(r.get('task_terminal') and not r.get('cuda_initialized') and
                    r.get('callback_seconds',{}).get('critic',0.) == 0. and
                    r.get('search_extraction_wall_seconds',0.) == 0. for r in allrows[episode]),
                'Development deployment used Critic/search features or incomplete tasks')
    best, scores = server.selection(allrows, c['development'], c['solver_seeds'])
    require(selection['best'] == best and selection['all_checkpoints'] == scores and
            selection['checkpoint_sha256'] == row['checkpoints'][str(best['episode'])]['sha256'], 'Selected checkpoint violates frozen rule')
    exports = {}
    for kind, episode in [('best',best['episode']), ('final',500)]:
        path = out/('actor_'+kind+'.pt')
        policy, payload = load_actor(path)
        require(payload['schema_version'] == 'actor-only-v1' and 'critic' not in payload and
                'actor_optimizer' not in payload and not hasattr(policy,'critic_network') and
                payload['provenance']['checkpoint_sha256'] == row['checkpoints'][str(episode)]['sha256'] and
                tensor_digest(payload['actor']) == row['checkpoints'][str(episode)]['actor_digest'], 'Actor-only export differs')
        exports[kind] = dict(path=str(path.relative_to(root)), sha256=sha256(path), episode=episode)
    return dict(group=group,seed=seed,selection_path=str((out/'selection.json').relative_to(root)),
                selection_sha256=sha256(out/'selection.json'),best=best,exports=exports)


def freeze(root, c, run, post):
    audited = load_audit(c, post)
    destination = post/'frozen_selection.json'
    with lock(post/'.freeze.lock'):
        if destination.exists():
            result = read(destination)
            require(result['protocol_sha256'] == fingerprint(c) and
                    result['training_audit_sha256'] == sha256(post/'training_audit.json'), 'Frozen selection identity changed')
            require({(r['group'],r['seed']) for r in result['models']} == set(MATRIX), 'Frozen selection matrix differs')
            for row in result['models']:
                require(sha256(root/row['selection_path']) == row['selection_sha256'], 'Frozen selection changed')
                require(set(row['exports']) == {'best','final'}, 'Frozen Actor exports incomplete')
                for record in row['exports'].values():
                    require(sha256(root/record['path']) == record['sha256'], 'Frozen Actor changed')
            return result
        with ExitStack() as stack:
            for group, seed in MATRIX:
                stack.enter_context(lock(server.group_dir(run,group,seed)/'.selection.lock',shared=True))
            models = [selected_record(root,c,run,audited,g,s) for g,s in MATRIX]
            result = dict(frozen_at_utc=now(),protocol_sha256=fingerprint(c),
                          training_audit_sha256=sha256(post/'training_audit.json'),
                          rule=c['selection'],models=models)
            write_json(destination,result)
        print('All six development selections frozen:',destination,flush=True)
        return result


def select(root, c, run, post, group, seed):
    audited = load_audit(c, post)
    require((group,seed) in MATRIX, 'Outside authorized six-job matrix')
    target_cpu(root,c,audited)
    validate_dataset(root,audited,'development')
    if (post/'frozen_selection.json').exists():
        freeze(root,c,run,post)
        print('Selection already frozen; keeping existing weights',flush=True)
        return
    out = server.group_dir(run,group,seed)
    with lock(out/'.selection.lock'):
        row = next(r for r in audited['jobs'] if (r['group'],r['seed']) == (group,seed))
        validate_training_files(root,row)
        server.baseline_bundle(root,c,run)
        server.select(root,c,run,group,seed)
        result = selected_record(root,c,run,audited,group,seed)
        write_json(out/'selection_verified.json',dict(verified_at_utc=now(),**result))
        print(json.dumps(result,indent=2),flush=True)
    try:
        freeze(root,c,run,post)
    except Pending as exc:
        print('This selection passed; other selections still pending:',exc,flush=True)


def evaluate(root, c, run, post, method, seed):
    audited = load_audit(c, post)
    target_cpu(root,c,audited)
    frozen = freeze(root,c,run,post)  # Never touch held-out outcomes before all six selections freeze.
    native = method == 'native'
    require(native or (method,seed) in MATRIX,'Outside authorized evaluation matrix')
    actor = None
    if not native:
        row = next(r for r in frozen['models'] if (r['group'],r['seed']) == (method,seed))
        actor = root/row['exports']['best']['path']
    names = list(audited['datasets']['test66'])
    instances = root/'tgppo/datasets/test_instances'
    validate_dataset(root,audited,'test66')
    with lock(post/('.evaluate-{}-s{}.lock'.format(method,seed))):
        server.evaluate(root,c,run,actor,instances,names,'official66-best',seed,native)
        key = 'relpscost-native' if native else sha256(actor)
        directory = run/'test/official66-best'/key[:16]
        rows = read(directory/'selected_results.json')
        require(len(rows) == 198 and {(r['instance_name'],r['solver_seed']) for r in rows} ==
                {(n,s) for n in names for s in c['solver_seeds']} and all(r.get('valid') and
                r.get('task_terminal') and not r.get('cuda_initialized') and
                r.get('callback_seconds',{}).get('critic',0.) == 0. and
                r.get('search_extraction_wall_seconds',0.) == 0. for r in rows), 'Held-out evaluation invalid')
        write_json(post/'evaluation'/('{}-s{}.json'.format(method,seed)),dict(
            finished_at_utc=now(),method=method,training_seed=None if native else seed,runs=198,
            frozen_selection_sha256=sha256(post/'frozen_selection.json'),
            selected_results_path=str((directory/'selected_results.json').relative_to(root)),
            selected_results_sha256=sha256(directory/'selected_results.json'),stack=audited['stack']))


def status(c,run,post):
    result = dict(training_audit=(post/'training_audit.json').exists(),
                  selection_frozen=(post/'frozen_selection.json').exists(),selections=[])
    for group,seed in MATRIX:
        out=server.group_dir(run,group,seed)
        row=dict(group=group,seed=seed,verified=(out/'selection_verified.json').exists())
        row['completed_development_checkpoints']=len(list(out.glob('development/episode-*/selected_results.json')))
        if row['verified']:
            row['best']=read(out/'selection_verified.json')['best']
        result['selections'].append(row)
    result['evaluation_receipts']=[p.name for p in (post/'evaluation').glob('*.json')]
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('protocol.json'))
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('command',choices=['audit','preflight','select','freeze','evaluate','status'])
    parser.add_argument('method',nargs='?',choices=['B0','R0','Rjoint','native'])
    parser.add_argument('seed',nargs='?',type=int,choices=[11,22],default=11)
    args=parser.parse_args();torch.set_num_threads(1)
    root=args.root.resolve();c=server.plan(root,args.config)
    run=(args.run_root or root/c['run_directory']).resolve();post=run/'post-training-20261023'
    post.mkdir(parents=True,exist_ok=True)
    if args.command=='audit':audit(root,c,run,post)
    elif args.command=='preflight':preflight(root,c,run,post)
    elif args.command=='select':select(root,c,run,post,args.method,args.seed)
    elif args.command=='freeze':freeze(root,c,run,post)
    elif args.command=='evaluate':evaluate(root,c,run,post,args.method,args.seed)
    else:status(c,run,post)


if __name__=='__main__':
    try:
        main()
    except Pending as exc:
        print('PENDING: '+str(exc),file=sys.stderr,flush=True)
        sys.exit(75)
    except (ValueError,RuntimeError) as exc:
        print('FAILED: '+str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
