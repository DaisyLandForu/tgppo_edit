"""Paper-scale final training. Explicit commands; no work executes on import."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESEARCH_ROOT', str(PACKAGE.parents[2])))
RUN = ROOT / 'tgppo/runs/paper-trainmode-20261108'
VARIANTS = ('b0', 'mlp', 'mlp_bc')
SEEDS = (11, 22)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def protocol():
    c = read(PACKAGE / 'configs/protocol.json')
    assert c['episodes'] == 500 and c['time_limit'] == 3600 and c['batch_episodes'] == 1
    assert len(c['train']) == 25 and len(c['test']) == 66
    assert not set(c['train']) & set(c['test']) and c['solver_seeds'] == list(range(5))
    if (c['alignment_choice'] != 'paper_text_standard_gae' or
            c['actor_update_mode'] != 'train' or
            c['gae_implementation'] != 'standard_episodic_v1' or
            c['update_organization'] != 'one_complete_episode_then_three_epochs'):
        raise ValueError('Explicit paper-text/standard-GAE protocol required')
    return c


def verify():
    release = read(PACKAGE / 'docs/release.json')
    if not release.get('passed'):
        raise ValueError('Local preparation checks have not passed')
    for relative, digest in release['files'].items():
        if sha(PACKAGE / relative) != digest:
            raise ValueError('Changed frozen package file: ' + relative)
    for relative, digest in release['inputs'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Changed frozen input: ' + relative)
    for relative, digest in read(PACKAGE / 'docs/protected_sources.json').items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Historical source changed: ' + relative)
    return protocol()


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Stopped(RuntimeError):
    pass


@contextlib.contextmanager
def budget(folder, seconds, resume):
    """One cumulative ceiling, including failed attempts and preparation."""
    folder = Path(folder)
    with lock(folder / 'execution.lock', blocking=False):
        path = folder / 'budget.json'
        record = read(path) if path.exists() else dict(attempts=[])
        if record['attempts'] and not resume:
            raise ValueError('Prior execution retained; use explicit --resume')
        spent = sum(a.get('elapsed_seconds', a['allocated_seconds']) for a in record['attempts'])
        launch = float(os.environ.get('PAPER_LAUNCH_EPOCH', time.time()))
        remaining = seconds - spent - max(0., time.time() - launch)
        if remaining <= 60:
            raise Stopped('Cumulative budget exhausted; no automatic extension')
        attempt = dict(started_epoch=launch, allocated_seconds=seconds-spent,
                       host=socket.gethostname(), pid=os.getpid())
        record['attempts'].append(attempt)
        write(path, record)
        def stop(signum, frame):
            raise Stopped('Signal/deadline {}; atomic checkpoint retained'.format(signum))
        previous = {s: signal.signal(s, stop) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.setitimer(signal.ITIMER_REAL, remaining-30)
        try:
            yield launch + seconds-spent
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in previous.items():
                signal.signal(s, handler)
            attempt.update(elapsed_seconds=max(0., time.time()-launch), finished_epoch=time.time())
            record['charged_seconds'] = sum(a.get('elapsed_seconds', a['allocated_seconds']) for a in record['attempts'])
            write(path, record)


def schedule(seed, c):
    import numpy as np
    rng = np.random.RandomState(seed)
    pairs = [(n, s) for n in c['train'] for s in c['solver_seeds']]
    result = []
    while len(result) < c['episodes']:
        result.extend(pairs[i] for i in rng.permutation(len(pairs)))
    return result[:c['episodes']]


def configuration(c, variant, seed, device=None):
    from project.runtime import config_with_protocol
    device = device or c['learner_device']
    cfg = read(ROOT / 'tgppo/configs/paper_hparams.json')
    for key in ('optimizer', 'optimizer_betas', 'optimizer_eps', 'weight_decay',
                'actor_width', 'inducing_points', 'expert_dataset_sha256', 'bc_steps',
                'bc_batch_size', 'bc_lr', 'bc_selection_milestones', 'attention_cell_budget'):
        cfg[key] = c[key]
    for key in ('actor_update_mode', 'gae_implementation', 'update_organization', 'alignment_choice'):
        cfg[key] = c[key]
    cfg.update(learner_device=device, actor_kind='b0' if variant == 'b0' else 'mlp',
               server_training_seed=seed, task_time_limit=c['time_limit'], variant=variant,
               experiment=c['experiment'], experiment_protocol_sha256=fp(c),
               expert_pretraining='bc' if variant == 'mlp_bc' else 'none')
    return config_with_protocol(cfg)


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cached_job(job, folder, identity, c, resume=False):
    from project.runtime import run_jobs
    folder = Path(folder)
    with lock(folder / 'job.lock'):
        path = folder / 'identity.json'
        if path.exists() and read(path) != identity:
            raise ValueError('Cached task identity changed: ' + str(folder))
        write(path, identity)
        attempts = sorted(folder.glob('attempt-*'))
        for a in reversed(attempts):
            f = a / (job['job_id'] + '.json')
            if not f.exists():
                continue
            row = read(f)
            if row.get('valid') and (job['mode'] != 'collect' or (a / (job['job_id']+'.pkl')).exists()):
                return row, a
        if attempts and not resume:
            raise ValueError('Previous failed/incomplete task retained; use --resume: ' + str(folder))
        attempt = folder / ('attempt-{:03d}'.format(len(attempts)))
        write(attempt / 'job.json', job)
        rows = run_jobs([job], attempt, 1, c['watchdog'])
        if len(rows) != 1 or not rows[0].get('valid'):
            raise RuntimeError('Invalid task retained; no automatic retry: ' + str(attempt))
        return rows[0], attempt


def make_job(c, name, solver_seed, seed, mode, job_id, checkpoint=None, baseline=None):
    from project.runtime import make_job as build
    directory = ROOT / ('tgppo/datasets/train_instances' if name in c['train'] else 'tgppo/datasets/test_instances')
    job = build(job_id, directory, name, c['time_limit'], solver_seed, seed,
                read(ROOT / 'tgppo/datasets/cutoffs.json'), mode, checkpoint, baseline)
    # Actor construction, BC, policy length and model label do not consume action RNG.
    from project.runtime import action_seed
    job['sampling_seed'] = action_seed(seed, job_id if mode == 'collect' else 'paper-test-{}-s{}'.format(name, solver_seed))
    job['address_space_limit_bytes'] = c['worker_address_space_bytes']
    return job


def native(c, name, solver_seed, info, resume=False):
    from project.runtime import job_identity
    from project.utils.protocol import validate_baseline
    job = make_job(c, name, solver_seed, 0, 'native', 'native-'+name+'-s'+str(solver_seed))
    ident = job_identity(job)
    signature = dict(identity=ident, stack=info['stack'])
    folder = RUN / 'normalization' / fp(info['stack']) / fp(ident)
    with lock(folder / 'baseline.lock'):
        file = folder / 'baseline.json'
        if file.exists():
            bundle = read(file)
            if bundle['signature'] != signature:
                raise ValueError('Normalization identity mismatch')
        else:
            bundle = None
            for source_root in c.get('normalization_readonly_sources', []):
                source = ROOT / source_root / fp(info['stack']) / fp(ident) / 'baseline.json'
                if not source.exists():
                    continue
                candidate = read(source)
                if (candidate.get('signature') != signature or
                        candidate.get('horizon_seconds') != c['time_limit']):
                    continue
                validate_baseline(candidate, ident)
                bundle = candidate
                write(folder/'reuse_provenance.json', dict(source=str(source), sha256=sha(source),
                    source_is_readonly=True, identity=ident, stack=info['stack']))
                break
            if bundle is None:
                row, _ = cached_job(job, folder / 'task', signature, c, resume)
                bundle = dict(signature=signature, identity=ident, result=row,
                              host=info['host'], horizon_seconds=c['time_limit'])
            write(file, bundle)
        validate_baseline(bundle, ident)
    return bundle


def ready_save(out, agent, cfg, progress, c):
    from project.runtime import save_checkpoint
    from server import export_actor
    save_checkpoint(out / 'latest.pt', agent, cfg, progress)
    n = progress['next_episode']
    if n in c['milestones']:
        cp = out / ('episode-{:04d}.pt'.format(n))
        save_checkpoint(cp, agent, cfg, progress)
        export_actor(cp, out / ('actor-{:04d}.pt'.format(n)))


def initial(c, variant, seed, out, device=None):
    import torch
    from project import runtime as rt
    device = device or c['learner_device']
    cfg = configuration(c, variant, seed, device)
    rt.seed_all(seed)
    agent = rt.build_agent(cfg)
    rng = rt.rng_state()
    before = state_digest(agent.actor_network.state_dict())
    if variant == 'mlp_bc':
        from pretrain import fit
        try:
            fit(agent.actor_network, cfg, out / 'bc', device=device, resume=(out/'bc/latest.pt').exists())
        finally:
            rt.restore_rng(rng)
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
    init = dict(actor_before_bc=before, actor=state_digest(agent.actor_network.state_dict()),
                critic=state_digest(agent.critic_network.state_dict()),
                cpu_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda_rng=([hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()]
                          if device.startswith('cuda') else []),
                schedule_sha256=fp(schedule(seed, c)), configuration_sha256=fp(cfg),
                hardware=read(out/'environment.json'))
    common = {k: init[k] for k in ('critic', 'cpu_rng', 'cuda_rng', 'schedule_sha256')}
    with lock(RUN/'pairing'/('s'+str(seed)+'.lock')):
        paired = RUN/'pairing'/('s'+str(seed)+'.json')
        record = read(paired) if paired.exists() else dict(common=common, initializations={})
        if record['common'] != common:
            raise ValueError('Actual cross-job Critic/RNG/schedule initialization differs')
        for other, entry in record['initializations'].items():
            if variant != 'b0' and other != 'b0' and entry['actor_before_bc'] != before:
                raise ValueError('Actual MLP/MLP+BC pre-BC initialization differs')
        record['initializations'][variant] = init
        write(paired, record)
    write(out / 'initialization.json', init)
    progress = dict(next_episode=0, phase='ready', experiment_protocol_sha256=fp(c))
    ready_save(out, agent, cfg, progress, c)
    return agent, cfg, progress


def train(c, variant, seed, resume, device=None):
    import torch
    import server
    from project import runtime as rt
    device = device or c['learner_device']
    out = RUN / 'training' / ('{}-s{}'.format(variant, seed))
    if (out / 'summary.json').exists() and read(out/'summary.json').get('training_complete'):
        assert rt.read_checkpoint(out/'latest.pt')['progress']['next_episode'] == c['episodes']
        print('Already complete; no repeated training.', flush=True)
        return
    with budget(out, c['job_budget_seconds'], resume) as end:
        info = server.preflight(ROOT, c, device)
        write(out / ('environment-attempt-'+str(len(read(out/'budget.json')['attempts']))+'.json'), info)
        write(out / 'environment.json', info)
        previous = read(out/'state.json') if (out/'state.json').exists() else {'next_episode': 0}
        write(out/'state.json', dict(previous, status='target_device_checks', learner_device=device))
        # Actual target-device bridge is mandatory, synthetic only, before any solve.
        from checks import target_checks
        gate = out / 'target-checks' / fp(dict(stack=info['stack'], gpu=info['gpu'], device=device))[:20]
        target_checks(c, seed, gate, device)
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        if (out/'latest.pt').exists():
            if not resume:
                raise ValueError('Existing checkpoint requires --resume')
            agent, payload = rt.load_agent(out/'latest.pt', device=device)
            cfg, progress = payload['config'], payload['progress']
            if cfg != configuration(c, variant, seed, device):
                raise ValueError('Resume config changed')
        else:
            write(out/'state.json', dict(status='initialization_or_bc', next_episode=0))
            agent, cfg, progress = initial(c, variant, seed, out, device)
        pending = out / 'pending.pt'
        plan = schedule(seed, c)
        write(out / 'schedule.json', [dict(episode=i+1, instance=n, solver_seed=s) for i, (n, s) in enumerate(plan)])
        write(out / 'state.json', dict(status='running', **progress))
        for i in range(progress['next_episode'], c['episodes']):
            # Never shorten a solve just to fit the allocation. Keep a PPO reserve.
            if end-time.time() < c['watchdog'] + 900:
                write(out/'state.json', dict(status='stopped_before_next_complete_episode', **progress))
                return
            if pending.exists():
                pp = rt.read_checkpoint(pending)
                if pp['progress']['next_episode'] == i:
                    if pp['config'] != cfg or pp['progress']['phase'] != 'update_pending':
                        raise ValueError('Wrong pending checkpoint')
                    agent, pp = rt.load_agent(pending, device=device)
                    task_result = pp['progress']['task_result']
                    collection_directory = Path(pp['progress']['collection_directory'])
                else:
                    if pp['progress']['next_episode'] > i:
                        raise ValueError('Pending checkpoint ahead of committed state')
                    pending.unlink()
                    pp = None
            else:
                pp = None
            if pp is None:
                name, ss = plan[i]
                write(out/'state.json', dict(status='normalization', next_episode=i, instance=name, solver_seed=ss))
                base = native(c, name, ss, info, resume)
                if end-time.time() < c['watchdog'] + 900:
                    write(out/'state.json', dict(status='stopped_after_normalization', **progress))
                    return
                jid = 'collect-e{:05d}-{}-s{}'.format(i+1, name, ss)
                job = make_job(c, name, ss, seed, 'collect', jid, out/'latest.pt', base)
                identity = dict(checkpoint_sha256=sha(out/'latest.pt'), task=rt.job_identity(job),
                                stack=info['stack'], sampling_seed=job['sampling_seed'])
                write(out/'state.json', dict(status='collecting', next_episode=i, instance=name, solver_seed=ss))
                task_result, collection_directory = cached_job(job, out/'episodes'/str(i+1), identity, c, resume)
                if len(agent.memory):
                    raise ValueError('Cross-episode buffer is forbidden')
                rt.import_trajectories(agent, [task_result], collection_directory)
                if len(agent.memory.episodes) != 1:
                    raise ValueError('Exactly one complete episode required')
                pp = dict(next_episode=i, phase='update_pending', task_result=task_result,
                          collection_directory=str(collection_directory), experiment_protocol_sha256=fp(c))
                rt.save_checkpoint(pending, agent, cfg, pp)
            rt.seed_all(rt.action_seed(seed, 'paper-ppo-update-e{:05d}'.format(i+1)))
            start = time.monotonic()
            def heartbeat(value):
                write(out/'state.json', dict(status='updating', next_episode=i, **value))
                print(json.dumps(dict(episode=i+1, phase='updating', **value)), flush=True)
            update = agent.learn(microbatch_size=c['microbatch_size'], progress_callback=heartbeat)
            assert len(agent.memory) == 0 and agent.memory.active is None
            update.update(episode=i+1, instance=plan[i][0], solver_seed=plan[i][1],
                          learner_wall_seconds=time.monotonic()-start,
                          phase='complete_episode_update', actor_sha256=state_digest(agent.actor_network.state_dict()))
            progress = dict(next_episode=i+1, phase='ready', experiment_protocol_sha256=fp(c))
            # Commit state first. A kill during learning replays pending, a kill
            # after this save resumes at the following episode with no double step.
            write(out/'updates'/('episode-{:04d}.json'.format(i+1)), update)
            ready_save(out, agent, cfg, progress, c)
            pending.unlink()
            for p in collection_directory.glob('*.pkl'):
                p.unlink()  # Only this newly committed trajectory; old runs untouched.
            write(out/'state.json', dict(status='running', **progress))
            print(json.dumps(update, ensure_ascii=False), flush=True)
        summary = dict(training_complete=True, episodes=c['episodes'], next_episode=progress['next_episode'],
                       model=variant, training_seed=seed, selection='fixed500',
                       checkpoint_sha256=sha(out/'episode-0500.pt'), actor_sha256=sha(out/'actor-0500.pt'))
        write(out/'summary.json', summary)
        write(out/'state.json', dict(status='training_complete', **progress))
        print(json.dumps(summary), flush=True)


def normalize(c, shard, shards, resume):
    import server
    if not 0 <= shard < shards:
        raise ValueError('Invalid normalization shard')
    out = RUN/'normalization-jobs'/('shard-{}-of-{}'.format(shard, shards))
    with budget(out, c['job_budget_seconds'], resume) as end:
        # Optional prefill is sequential; request one CPU rather than four.
        info = server.preflight(ROOT, dict(c, workers=1), 'cpu')
        for i, (name, ss) in enumerate((n, s) for n in c['train'] for s in c['solver_seeds']):
            if i % shards != shard:
                continue
            if end-time.time() < c['watchdog']+60:
                return
            native(c, name, ss, info, resume)
            print(json.dumps(dict(normalized=name, solver_seed=ss)), flush=True)


def eval_instances(c, shard):
    if not 0 <= shard < c['evaluation_shards']:
        raise ValueError('Evaluation shards are0..16')
    if shard == 16:
        return ['mspp16']
    ordinary = sorted((n for n in c['test'] if n != 'mspp16'),
                      key=lambda n: hashlib.sha256(('paper-eval-v1:'+n).encode()).hexdigest())
    return ordinary[shard::16]


def frozen_models(c):
    from project import runtime as rt
    models = []
    for seed in SEEDS:
        for variant in VARIANTS:
            out = RUN/'training'/('{}-s{}'.format(variant, seed))
            summary = read(out/'summary.json')
            if not summary.get('training_complete') or summary['episodes'] != 500:
                raise ValueError('All six500episode jobs must finish before formal evaluation')
            path = out/'actor-0500.pt'
            if sha(path) != summary['actor_sha256']:
                raise ValueError('Final Actor changed')
            payload = rt.read_checkpoint(path, allow_actor_only=True)
            if payload['progress']['next_episode'] != 500 or payload['config'] != configuration(
                    c, variant, seed, payload['config']['learner_device']):
                raise ValueError('Wrong fixed500 Actor/configuration')
            cp = out/'episode-0500.pt'
            if sha(cp) != summary['checkpoint_sha256'] or payload['provenance']['checkpoint_sha256'] != summary['checkpoint_sha256']:
                raise ValueError('Full checkpoint/final Actor provenance mismatch')
            models.append(dict(label=variant+'-s'+str(seed), variant=variant, training_seed=seed,
                               path=str(path), sha256=sha(path)))
    return models


def evaluate(c, shard, resume):
    import server
    from project.runtime import run_jobs
    names = eval_instances(c, shard)
    models = frozen_models(c)  # Fail before creating/billing an evaluation attempt.
    out = RUN/'evaluation'/('shard-'+str(shard))
    with budget(out, c['job_budget_seconds'], resume) as end:
        info = server.preflight(ROOT, c, 'cpu')
        env_file = out/'environment.json'
        if env_file.exists() and read(env_file) != info:
            # Preserve completed results and refuse hardware-mixed partial panels.
            old = read(env_file)
            if old['host'] != info['host'] or old['stack'] != info['stack']:
                raise ValueError('Evaluation resume requires same host/software; preserve partial cohort for review')
        write(env_file, info)
        write(out/'models.json', models)
        for name in names:
            pairs = []
            labels = [None] + models
            for ss in c['solver_seeds']:
                offset = ss % len(labels)
                for model in labels[offset:]+labels[:offset]:
                    seed = model['training_seed'] if model else 0
                    label = model['label'] if model else 'native'
                    jid = 'eval-{}-{}-s{}'.format(label, name, ss)
                    j = make_job(c, name, ss, seed, 'eval' if model else 'native', jid,
                                 model['path'] if model else None)
                    j['address_space_limit_bytes'] = c['evaluation_large_memory_bytes'] if name == 'mspp16' else c['evaluation_memory_bytes']
                    pairs.append(j)
            # All methods for an instance are paired on one CPU host/cohort.
            base = out/name
            identity = dict(models=models, stack=info['stack'], host=info['host'],
                            jobs=pairs, protocol_sha256=fp(c))
            ident_file = base/'identity.json'
            if ident_file.exists() and read(ident_file) != identity:
                raise ValueError('Changed evaluation inputs')
            write(ident_file, identity)
            results = {}
            for attempt in sorted(base.glob('attempt-*')):
                for j in pairs:
                    p = attempt/(j['job_id']+'.json')
                    if p.exists() and read(p).get('valid'):
                        results[j['job_id']] = read(p)
            missing = [j for j in pairs if j['job_id'] not in results]
            if list(base.glob('attempt-*')) and missing and not resume:
                raise ValueError('Incomplete evaluation retained; explicit --resume required')
            workers = 1 if name == 'mspp16' else c['evaluation_workers']
            # Dispatch only one wave at a time so the budget can stop cleanly.
            for start in range(0, len(missing), workers):
                if end-time.time() < c['watchdog']+60:
                    return
                attempt = base/('attempt-{:03d}'.format(len(list(base.glob('attempt-*')))))
                jobs = missing[start:start+workers]
                write(attempt/'jobs.json', jobs)
                rows = run_jobs(jobs, attempt, workers, c['watchdog'])
                if any(not r.get('valid') for r in rows):
                    raise RuntimeError('Invalid evaluation retained; no automatic retry')
                results.update({r['job_id']: r for r in rows})
                write(base/'results.json', [results[j['job_id']] for j in pairs if j['job_id'] in results])
            assert len(results) == 35
        write(out/'complete.json', dict(complete=True, instances=names, tasks=35*len(names)))


def status(c):
    result = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            folder = RUN/'training'/('{}-s{}'.format(variant, seed))
            row = read(folder/'state.json') if (folder/'state.json').exists() else dict(status='not_started', next_episode=0)
            if (folder/'summary.json').exists():
                row['summary'] = read(folder/'summary.json')
            if (folder/'budget.json').exists():
                row['budget'] = read(folder/'budget.json')
            result[variant+'-s'+str(seed)] = row
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'normalize', 'eval', 'status', 'check', 'summarize'])
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--seed', type=int, choices=SEEDS, default=11)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=4)
    args = p.parse_args()
    c = protocol()
    if args.command == 'status':
        print(json.dumps(status(c), ensure_ascii=False, indent=2))
        return
    c = verify()
    if args.command == 'check':
        print('Frozen source, original baseline, data and paper-aligned protocol verified; no solves/training.')
        return
    if args.command == 'summarize':
        from analyze import summarize
        summarize(c)
        return
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        if args.command == 'run':
            if not args.variant:
                p.error('--variant is required')
            train(c, args.variant, args.seed, args.resume, args.device)
        elif args.command == 'normalize':
            normalize(c, args.shard, args.shards, args.resume)
        elif args.command == 'eval':
            evaluate(c, args.shard, args.resume)
    except BaseException as exc:
        if args.command == 'run' and args.variant:
            out = RUN/'training'/('{}-s{}'.format(args.variant, args.seed))
            ledger = read(out/'budget.json') if (out/'budget.json').exists() else {'attempts': []}
            # A duplicate launcher must not overwrite a live owner's state.
            if ledger['attempts'] and ledger['attempts'][-1].get('pid') == os.getpid():
                previous = read(out/'state.json') if (out/'state.json').exists() else {}
                write(out/'state.json', dict(previous, status='stopped' if isinstance(exc, Stopped) else 'failed',
                                             error=repr(exc), stopped_epoch=time.time()))
        raise


if __name__ == '__main__':
    main()
