"""Bounded three-arm experiment. GPU run jobs train; the last job finalizes once."""
import argparse
import contextlib
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import torch
import server
from project import runtime as rt
from project.utils.protocol import fingerprint, sha256, write_json, validate_baseline

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESEARCH_ROOT', server.DEFAULT_ROOT)).resolve()
MATRIX = ROOT/'tgppo/runs/candidate-residual-20261104'
VARIANTS = ('b0', 'score', 'candidate')
SEEDS = (11, 22)
TERMINAL = ('training_complete', 'budget_exhausted', 'failed')


def protocol(variant):
    c = server.plan(ROOT, HERE/'configs'/(variant+'.json'))
    return c, ROOT/c['run_directory']


def check_protocols():
    base, _ = protocol('b0')
    allowed = {'variant', 'candidate_residual_mode', 'run_directory'}
    for v in VARIANTS:
        c, _ = protocol(v)
        if any(c.get(k) != base.get(k) for k in set(c)|set(base) if k not in allowed):
            raise ValueError('Uncontrolled protocol difference: ' + v)
        if (c['groups'] != ['B0'] or c['training_seeds'] != list(SEEDS) or
                c['max_episodes'] != 100 or c['milestones'] != [20, 100] or
                c['time_limit'] != 120 or c['evaluation_solver_seeds'] != [0, 1]):
            raise ValueError('Matrix differs from approved scope')
    if 6*base['gpu_job_budget_seconds']+base['finalization_budget_seconds']+base[
            'local_preparation_compute_cap_seconds'] > base['matrix_wall_budget_seconds']:
        raise ValueError('Parallel job ceilings exceed the aggregate wall budget')


def package_hashes():
    paths = list((HERE/'code').rglob('*.py')) + list((HERE/'configs').glob('*.json'))
    paths += list(HERE.glob('*.sh')) + list((HERE/'tests').glob('*.py'))
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}


def verify_package():
    check_protocols()
    manifest = server.read(HERE/'docs/bundle_manifest.json')
    if manifest['files'] != package_hashes():
        raise ValueError('Package changed after local verification; do not train')
    checks = server.read(HERE/'docs/local_checks.json')
    if not checks['passed'] or checks['bundle_sha256'] != fingerprint(manifest['files']):
        raise ValueError('Local checks do not certify this package')
    for relative, digest in checks['external_inputs'].items():
        if sha256(ROOT/relative) != digest:
            raise ValueError('Frozen external input changed: '+relative)


@contextlib.contextmanager
def file_lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class BudgetExhausted(RuntimeError):
    pass


@contextlib.contextmanager
def deadline(seconds):
    if seconds <= 0:
        raise BudgetExhausted('No remaining compute/deadline budget')
    def expired(signum, frame):
        raise BudgetExhausted('Declared wall budget reached')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def time_left(c, reserve=0):
    return datetime.fromisoformat(c['freeze_deadline']).timestamp()-time.time()-reserve


def status_path(variant, seed):
    return MATRIX/'jobs'/('{}-s{}.json'.format(variant, seed))


def statuses():
    return {v+'-s'+str(s): server.read(status_path(v, s)) if status_path(v, s).exists()
            else {'status': 'not_started'} for v in VARIANTS for s in SEEDS}


def adopt_native(source, c, target, signature):
    """Reuse native rows only by full job identity and measured CPU/software stack."""
    bundle = server.read(source)
    if bundle['stack'] != signature:
        return False
    cutoffs = server.read(ROOT/'tgppo/datasets/cutoffs.json')
    for n in c['train']+c['development']:
        for ss in c['solver_seeds']:
            j = server.bound_job(ROOT, c, cutoffs, n, ss, 11, 'native')
            ident = rt.job_identity(j)
            validate_baseline(bundle['baselines'][fingerprint(ident)], ident)
    write_json(target/'normalization/baselines.json', dict(bundle,
        protocol_sha256=fingerprint(c), source=str(source), source_sha256=sha256(source)))
    return True


def prepare(variant, c, run, info):
    # Shared filesystem lock avoids six duplicate native-normalization sweeps.
    with file_lock(MATRIX/'normalization.lock'):
        platform = MATRIX/'matrix_environment.json'
        expected = dict(stack=info['stack'], gpu=info['gpu'], workers=c['workers'])
        if platform.exists() and server.read(platform) != expected:
            raise ValueError('Matrix CPU/software/GPU differs; keep all arms on matching resources')
        if not platform.exists():
            write_json(platform, expected)
        canonical, _ = protocol('b0')
        shared = MATRIX/'shared'
        destination = shared/'normalization/baselines.json'
        if not destination.exists():
            candidates = [ROOT/'tgppo/runs/server6-20261022/normalization/baselines.json',
                ROOT/'tgppo/runs/transformer-ablation-20261027/transformer/normalization/baselines.json']
            reused = False
            for source in candidates:
                if not source.exists():
                    continue
                try:
                    reused = adopt_native(source, canonical, shared, info['stack'])
                except (KeyError, ValueError):
                    reused = False
                if reused:
                    break
            if not reused:
                server.baseline(ROOT, canonical, shared)
        server.baseline_bundle(ROOT, canonical, shared)
        attached = run/'normalization/baselines.json'
        if not attached.exists():
            adopt_native(destination, c, run, info['stack'])
        server.baseline_bundle(ROOT, c, run)
    # One cheap synthetic bridge per architecture, without extra SCIP episodes.
    with file_lock(run/'gpu-bridge/bridge.lock'):
        gate = run/'gpu-bridge/verification.json'
        if gate.exists():
            server.require_target_evidence(ROOT, c, run, info)
            return
        from engineering import bridge
        cfg = server.params(ROOT, c, 'B0', 11, 'cuda')
        result = bridge(cfg, 'cuda', run/'gpu-bridge/synthetic')
        write_json(gate, dict(result, parent_protocol_sha256=fingerprint(c),
            stack=info['stack'], gpu=info['gpu'], variant=variant))


def run(variant, seed, resume=False):
    verify_package()
    c, directory = protocol(variant)
    target = status_path(variant, seed)
    with file_lock(target.with_suffix('.lock'), blocking=False):
        old = server.read(target) if target.exists() else {}
        if old.get('status') == 'training_complete':
            print('Already complete:', variant, seed)
        else:
            if old and not resume:
                raise FileExistsError('Use --resume for retained attempt; budget is not reset')
            attempts = old.get('attempts', [])
            # If the scheduler killed a wrapper without its finally block, charge
            # the whole remaining reservation rather than silently reset usage.
            spent = sum(a.get('wall_seconds', c['gpu_job_budget_seconds']) for a in attempts)
            start_epoch = float(os.environ.get('CR_LAUNCH_EPOCH', time.time()))
            entry = dict(id=uuid.uuid4().hex, started_epoch=start_epoch)
            attempts.append(entry)
            record = dict(status='running', variant=variant, seed=seed, attempts=attempts,
                protocol_sha256=fingerprint(c), budget_seconds=c['gpu_job_budget_seconds'])
            write_json(target, record)
            try:
                remaining = min(c['gpu_job_budget_seconds']-spent-(time.time()-start_epoch),
                    time_left(c, c['finalization_budget_seconds']))
                with deadline(remaining):
                    info = server.preflight(ROOT, c, 'cuda')
                    prepare(variant, c, directory, info)
                    out = server.group_dir(directory, 'B0', seed)
                    server.train(ROOT, c, directory, 'B0', seed,
                        resume=(out/'latest.pt').exists() and resume, device='cuda')
                    for episode in (20, 100):
                        cp = out/('episode-{:04d}.pt'.format(episode))
                        if cp.exists():
                            server.export_actor(cp, out/('actor-{:04d}.pt'.format(episode)))
                    record['status'] = 'training_complete'
            except BudgetExhausted as exc:
                record.update(status='budget_exhausted', error=str(exc))
            except BaseException as exc:
                record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            finally:
                entry['wall_seconds'] = time.time()-start_epoch
                record['charged_wall_seconds'] = spent+entry['wall_seconds']
                write_json(target, record)
            print(json.dumps(record, indent=2), flush=True)
    # Last terminal job automatically evaluates the common endpoint once.
    maybe_finalize()
    if server.read(target)['status'] == 'failed':
        raise RuntimeError('Job failed; retained logs at ' + str(target))


def choose_endpoint(records):
    if not all(r['status'] in TERMINAL for r in records.values()):
        return None
    for ep in (100, 20):
        if ep == 100 and not all(r['status'] == 'training_complete' for r in records.values()):
            continue
        if all((server.group_dir(protocol(v)[1], 'B0', s)/
                ('episode-{:04d}.pt'.format(ep))).exists() for v in VARIANTS for s in SEEDS):
            return ep
    return 0


def maybe_finalize():
    records = statuses()
    endpoint = choose_endpoint(records)
    if endpoint is None:
        print('Other GPU jobs pending; finalization will run after all six terminate.', flush=True)
        return
    try:
        with file_lock(MATRIX/'finalization.lock', blocking=False):
            finalize(endpoint)
    except BlockingIOError:
        print('Another job is finalizing the matrix.', flush=True)


def evaluate(endpoint):
    base, _ = protocol('b0')
    signature = server.stack(ROOT)
    output = MATRIX/'evaluation'/('episode-{:04d}'.format(endpoint))
    output.mkdir(parents=True, exist_ok=True)
    frozen = []
    initializations = {}
    for v in VARIANTS:
        c, run_directory = protocol(v)
        for s in SEEDS:
            out = server.group_dir(run_directory, 'B0', s)
            cp = out/('episode-{:04d}.pt'.format(endpoint))
            payload = rt.read_checkpoint(cp)
            if (payload['progress']['phase'] != 'ready' or
                    payload['progress']['next_episode'] != endpoint or
                    payload['config']['candidate_residual_mode'] != c['candidate_residual_mode'] or
                    payload['config']['server_protocol_sha256'] != fingerprint(c)):
                raise ValueError('Checkpoint does not match common frozen endpoint')
            if any(not torch.isfinite(t).all() for net in ('actor', 'critic') for t in payload[net].values()):
                raise ValueError('Nonfinite checkpoint retained: '+str(cp))
            initializations[v, s] = server.read(out/'initialization.json')
            actor = out/('actor-{:04d}.pt'.format(endpoint))
            if not actor.exists():
                server.export_actor(cp, actor)
            frozen.append(dict(variant=v, seed=s, checkpoint=str(cp), checkpoint_sha256=sha256(cp),
                actor=str(actor), actor_sha256=sha256(actor),
                complete_ppo_blocks=payload['update_counter'],
                logical_minibatch_updates=payload['progress']['optimizer_steps'],
                pending_transitions=payload['memory']['num_transitions'],
                trained_episodes=payload['progress']['trained_episodes']))
    for s in SEEDS:
        for key in ('common_actor_fingerprint', 'common_critic_fingerprint',
                    'public_torch_rng_sha256', 'public_cuda_rng_sha256'):
            if len({initializations[v, s][key] for v in VARIANTS}) != 1:
                raise ValueError('Unpaired initial weights/RNG: '+str((s, key)))
    manifest = dict(episode=endpoint, models=frozen, stack=signature,
        development=base['development'], solver_seeds=base['evaluation_solver_seeds'],
        scope='existing development exploration; not unseen generalization')
    mf = output/'frozen_models.json'
    if mf.exists() and server.read(mf) != manifest:
        raise ValueError('Frozen model/evaluation identity changed')
    write_json(mf, manifest)
    cutoffs = server.read(ROOT/'tgppo/datasets/cutoffs.json')
    native_jobs = [server.bound_job(ROOT, base, cutoffs, n, ss, 11, 'native',
        job_id='dev-{}-s{}'.format(n, ss))
        for n in base['development'] for ss in base['evaluation_solver_seeds']]
    native_file = output/'native_rows.json'
    if not native_file.exists():
        shared = MATRIX/'shared/normalization/baselines.json'
        bundle = server.read(shared) if shared.exists() else {}
        if bundle.get('stack') == signature:
            native = []
            for j in native_jobs:
                ident = rt.job_identity(j)
                r = bundle['baselines'][fingerprint(ident)]
                validate_baseline(r, ident)
                native.append(dict(r['result'], job_id=j['job_id']))
        else:
            native = [r for r, _ in server.cached_jobs(native_jobs, output/'native',
                dict(stack=signature, names=base['development'], protocol=fingerprint(base)),
                base['workers'], base['watchdog'])]
        write_json(native_file, native)
    for m in frozen:
        c, _ = protocol(m['variant'])
        jobs = []
        for n in c['development']:
            for ss in c['evaluation_solver_seeds']:
                j = server.bound_job(ROOT, c, cutoffs, n, ss, m['seed'], 'eval', m['actor'],
                    job_id='dev-{}-s{}'.format(n, ss))
                # Pair RNG independently of variant and checkpoint path.
                j['sampling_seed'] = rt.action_seed(m['seed'], j['job_id'])
                jobs.append(j)
        pairs = server.cached_jobs(jobs, output/('{}-s{}'.format(m['variant'], m['seed'])),
            dict(actor_sha256=m['actor_sha256'], stack=signature, protocol=fingerprint(c)),
            c['workers'], c['watchdog'])
        if any(r.get('callback_seconds', {}).get('critic', 0.) != 0 or r.get('fallback_action_count', 0)
                or r.get('cuda_initialized') for r, _ in pairs):
            raise ValueError('Evaluation violated Actor-only CPU/no-fallback deployment')
    if any(sha256(m['checkpoint']) != m['checkpoint_sha256'] or
            sha256(m['actor']) != m['actor_sha256'] for m in frozen):
        raise ValueError('Evaluation mutated a frozen model')
    write_json(output/'complete.json', dict(complete=True, episode=endpoint, model_tasks=60,
        native_reference_tasks=10, native_may_be_reused=True, stack=signature))


def finalize(endpoint):
    verify_package()
    c, _ = protocol('b0')
    p = MATRIX/'finalization.json'
    state = server.read(p) if p.exists() else {'attempts': []}
    if state.get('complete'):
        print('Already finalized:', MATRIX/'results/REPORT.md')
        return
    spent = sum(a.get('wall_seconds', c['finalization_budget_seconds']) for a in state['attempts'])
    entry = dict(started_epoch=time.time())
    state['attempts'].append(entry)
    state.update(complete=False, episode=endpoint)
    write_json(p, state)
    try:
        with deadline(min(c['finalization_budget_seconds']-spent, time_left(c))):
            if endpoint:
                evaluate(endpoint)
            from reporting import report
            report(ROOT, MATRIX, endpoint, statuses())
            state['complete'] = bool(endpoint)
    except BaseException as exc:
        state.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        entry['wall_seconds'] = time.time()-entry['started_epoch']
        state['charged_wall_seconds'] = spent+entry['wall_seconds']
        write_json(p, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', 'finalize', 'report', 'check'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(statuses(), indent=2)); return
    if args.command == 'check':
        verify_package(); print('Frozen package and local checks verified. No training or solve.'); return
    if args.command == 'run':
        if args.variant is None or args.seed is None:
            parser.error('run requires --variant and --seed')
        run(args.variant, args.seed, args.resume)
    elif args.command == 'finalize':
        maybe_finalize()
    elif args.command == 'report':
        from reporting import report
        endpoint = choose_endpoint(statuses())
        report(ROOT, MATRIX, endpoint or 0, statuses())


if __name__ == '__main__':
    main()
