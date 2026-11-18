"""Independent recovery of frozen experiments; no training on import/check/status."""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
PACKAGES = dict(aligned='paper-aligned-20261106', trainmode='paper-trainmode-20261108')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def boot(experiment):
    upstream = PACKAGE.parent/PACKAGES[experiment]
    sys.path.insert(0, str(upstream/'code'))
    import paper
    return paper


def verify(paper):
    release = json.loads((PACKAGE/'release.json').read_text())
    if not release.get('passed'):
        raise ValueError('Recovery checks have not passed')
    for relative, digest in release['files'].items():
        if sha(PACKAGE/relative) != digest:
            raise ValueError('Changed recovery file: '+relative)
    if sha(paper.PACKAGE/'docs/release.json') != release['upstream_releases'][paper.PACKAGE.name]:
        raise ValueError('Upstream release changed')
    return paper.verify()


def cohort_cached_job(paper, original):
    def cached(job, folder, identity, c, resume=False):
        folder = Path(folder)
        previous = folder/'identity.json'
        if job['mode'] != 'collect' or not previous.exists() or paper.read(previous) == identity:
            return original(job, folder, identity, c, resume)
        old = paper.read(previous)
        if {k:v for k,v in old.items() if k != 'stack'} != {k:v for k,v in identity.items() if k != 'stack'}:
            raise ValueError('Changed policy/task/action RNG; CPU migration alone is allowed')
        a, b = old.get('stack', {}), identity.get('stack', {})
        if (not a.get('cpu_model') or not b.get('cpu_model') or
                {k:v for k,v in a.items() if k != 'cpu_model'} !=
                {k:v for k,v in b.items() if k != 'cpu_model'}):
            raise ValueError('Changed software/architecture; CPU migration alone is allowed')
        if not resume:
            raise ValueError('CPU migration requires resume')
        destination = folder/'recovery-cohorts'/paper.fp(identity)
        provenance = dict(source_identity=str(previous), source_sha256=sha(previous),
            old_stack=a, new_stack=b, identity=identity, source_preserved=True)
        marker = destination/'migration.json'
        if marker.exists() and paper.read(marker) != provenance:
            raise ValueError('Changed migration provenance')
        paper.write(marker, provenance)
        return original(job, destination, identity, c, resume)
    return cached


def budget_seconds(record):
    return sum(a.get('elapsed_seconds', a['allocated_seconds']) for a in record.get('attempts', []))


def unlimited_budget(paper):
    """Explicit resource override; preserve history and record actual elapsed time."""
    @contextlib.contextmanager
    def budget(folder, seconds, resume):
        folder = Path(folder)
        with paper.lock(folder/'execution.lock', blocking=False):
            path = folder/'budget.json'
            record = paper.read(path) if path.exists() else dict(attempts=[])
            if record['attempts'] and not resume:
                raise ValueError('Existing execution requires resume')
            launch = float(os.environ.get('PAPER_LAUNCH_EPOCH', time.time()))
            attempt = dict(started_epoch=launch, host=socket.gethostname(), pid=os.getpid(),
                allocated_seconds=seconds, wall_limit_seconds=None,
                allocated_seconds_is_legacy_accounting_fallback=True,
                resource_override='explicit --no-total-wall-limit', recovery_sha256=sha(__file__))
            record['attempts'].append(attempt)
            record['wall_limit_seconds'] = None
            paper.write(path, record)
            def stop(signum, frame):
                raise paper.Stopped('Signal {}; checkpoint retained'.format(signum))
            handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                yield float('inf')  # Original full-episode reserve remains unmodified.
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
                attempt.update(elapsed_seconds=max(0., time.time()-launch), finished_epoch=time.time())
                record['charged_seconds'] = budget_seconds(record)
                paper.write(path, record)
    return budget


def guarded_budget(paper, original, execution):
    """Write failures only while holding this job's lock, never by PID alone."""
    @contextlib.contextmanager
    def budget(folder, seconds, resume):
        with original(folder, seconds, resume) as end:
            folder = Path(folder)
            attempt = len(paper.read(folder/'budget.json')['attempts'])
            paper.write(folder/('recovery-attempt-{}.json'.format(attempt)), dict(execution,
                host=socket.gethostname(), pid=os.getpid(), entry_sha256=sha(__file__)))
            try:
                yield end
            except BaseException as exc:
                previous = paper.read(folder/'state.json') if (folder/'state.json').exists() else {}
                paper.write(folder/'state.json', dict(previous,
                    status='stopped' if isinstance(exc, paper.Stopped) else 'failed',
                    error=repr(exc), stopped_epoch=time.time()))
                raise
    return budget


def migrate_payload(payload, destination_collection=None):
    """Change only execution device and, for pending, its cleanup directory."""
    result = copy.deepcopy(payload)
    result['config']['learner_device'] = 'cpu'
    # Same JSON fingerprint as the frozen checkpoint implementation.
    result['config_sha256'] = hashlib.sha256(json.dumps(result['config'], sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    if destination_collection is not None:
        result['progress']['collection_directory'] = str(destination_collection)
    return result


def cpu_fork(paper, source_run, recovery_run, c, variant, seed):
    import torch
    from project import runtime as rt
    label = '{}-s{}'.format(variant, seed)
    source, out = source_run/'training'/label, recovery_run/'training'/label
    with paper.lock(recovery_run/'import-locks'/(label+'.lock'), blocking=False):
        marker = out/'cpu_continuation.json'
        if marker.exists():
            info = paper.read(marker)
            if info['source_run'] != str(source) or info['upstream_release_sha256'] != sha(paper.PACKAGE/'docs/release.json'):
                raise ValueError('CPU continuation provenance mismatch')
            if not (out/'latest.pt').exists():
                raise ValueError('CPU continuation checkpoint is missing')
            return out
        if out.exists():
            raise ValueError('Uncommitted CPU import retained for inspection: '+str(out))
        # Reject a live parent, including a parent on a different cluster node.
        with paper.lock(source/'execution.lock', blocking=False):
            source_budget = paper.read(source/'budget.json')
            if any('elapsed_seconds' not in a for a in source_budget['attempts']):
                raise ValueError('Parent has an unfinished budget entry; inspect it before CPU migration')
            source_sha = sha(source/'latest.pt')
            payload = rt.read_checkpoint(source/'latest.pt')
            device = payload['config']['learner_device']
            expected = (paper.configuration(c, variant, seed, device))
            if payload['config'] != expected:
                raise ValueError('Parent configuration differs beyond execution device')
            latest = migrate_payload(payload)
            latest['config_sha256'] = rt.fingerprint(latest['config'])
            staging = out.with_name(out.name+'.import-'+uuid.uuid4().hex)
            staging.mkdir(parents=True)
            torch.save(latest, staging/'latest.pt')
            pending_info = None
            if (source/'pending.pt').exists():
                pending_sha = sha(source/'pending.pt')
                pending = rt.read_checkpoint(source/'pending.pt')
                n, m = payload['progress']['next_episode'], pending['progress']['next_episode']
                if m > n or pending['config'] != payload['config']:
                    raise ValueError('Parent pending trajectory/configuration is inconsistent')
                if m == n:
                    if pending['progress']['phase'] != 'update_pending':
                        raise ValueError('Parent pending has wrong phase')
                    cleanup = out/'imported-pending'/('episode-'+str(n+1))
                    (staging/'imported-pending'/('episode-'+str(n+1))).mkdir(parents=True)
                    converted = migrate_payload(pending, cleanup)
                    converted['config_sha256'] = rt.fingerprint(converted['config'])
                    torch.save(converted, staging/'pending.pt')
                    pending_info = dict(sha256=pending_sha,
                        source_collection=pending['progress']['collection_directory'],
                        source_trajectory_preserved=True, phase='update_pending')
            if sha(source/'latest.pt') != source_sha:
                raise ValueError('Parent checkpoint changed during import')
            paper.write(staging/'budget.json', source_budget)
            paper.write(staging/'state.json', dict(status='cpu_continuation_ready', **latest['progress']))
            paper.write(staging/'cpu_continuation.json', dict(source_run=str(source),
                source_latest_sha256=source_sha, source_config_sha256=payload['config_sha256'],
                destination_initial_sha256=sha(staging/'latest.pt'), pending=pending_info,
                source_budget_sha256=sha(source/'budget.json'), source_charged_seconds=budget_seconds(source_budget),
                source_preserved=True, learner_device_before=device, learner_device_after='cpu',
                weights_and_adamw_preserved=True, algorithm_protocol_unchanged=True,
                bitwise_equivalent_to_gpu=False, no_new_solver_runs_during_import=True,
                upstream_release_sha256=sha(paper.PACKAGE/'docs/release.json')))
            staging.replace(out)
    return out


def readonly_native(paper, original, source_run):
    def native(c, name, solver_seed, info, resume=False):
        from project import runtime as rt
        from project.utils.protocol import validate_baseline
        job = paper.make_job(c, name, solver_seed, 0, 'native', 'native-'+name+'-s'+str(solver_seed))
        ident = rt.job_identity(job)
        suffix = Path(paper.fp(info['stack']))/paper.fp(ident)
        source = source_run/'normalization'/suffix/'baseline.json'
        folder = paper.RUN/'normalization'/suffix
        if source.exists():
            with paper.lock(folder/'baseline.lock'):
                target = folder/'baseline.json'
                if not target.exists():
                    bundle = paper.read(source)
                    if bundle.get('signature') != dict(identity=ident, stack=info['stack']) or bundle.get('horizon_seconds') != c['time_limit']:
                        raise ValueError('Parent normalization identity mismatch')
                    validate_baseline(bundle, ident)
                    paper.write(target, bundle)
                    paper.write(folder/'reuse_provenance.json', dict(source=str(source),
                        source_sha256=sha(source), readonly=True))
        return original(c, name, solver_seed, info, resume)
    return native


def install_cpu(paper, source_run, recovery_run):
    import server
    import checks
    configuration, preflight, target_checks = paper.configuration, server.preflight, checks.target_checks
    paper.RUN = recovery_run
    paper.configuration = lambda c, v, s, device=None: configuration(c, v, s, 'cpu')
    server.preflight = lambda root, c, device: preflight(root, c, 'cpu')
    checks.target_checks = lambda c, seed, directory, device: target_checks(c, seed, directory, 'cpu')
    paper.native = readonly_native(paper, paper.native, source_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'check', 'status'])
    parser.add_argument('--experiment', choices=PACKAGES, required=True)
    parser.add_argument('--variant', choices=['b0', 'mlp', 'mlp_bc'])
    parser.add_argument('--seed', type=int, choices=[11,22], default=11)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-total-wall-limit', action='store_true')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    paper = boot(args.experiment)
    c = verify(paper)
    if args.command == 'check':
        print('Recovery and frozen source verified; no training, GPU or solver started.')
        return
    if not args.variant:
        parser.error('--variant is required for run/status')
    source_run = paper.RUN
    source = source_run/'training'/('{}-s{}'.format(args.variant,args.seed))
    recovery_run = paper.ROOT/'tgppo/runs/paper-recovery-20261109'/('cpu-'+args.experiment)
    out = (recovery_run/'training'/source.name) if args.device == 'cpu' else source
    if args.command == 'status':
        result = dict(source=str(source), output=str(out))
        for filename in ['state.json','budget.json','cpu_continuation.json']:
            if (out/filename).exists():
                result[filename] = paper.read(out/filename)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not (out/'latest.pt').exists() and not (source/'latest.pt').exists():
        raise ValueError('Resume-only entry requires an existing checkpoint')
    ledger_path = (out if out.exists() else source)/'budget.json'
    ledger = paper.read(ledger_path)
    if not args.no_total_wall_limit and c['job_budget_seconds']-budget_seconds(ledger) < c['watchdog']+900:
        raise ValueError('Script cumulative48h ceiling leaves insufficient full-episode reserve. '
                         'Use explicit --no-total-wall-limit to continue without a total wall ceiling; history is retained.')
    if args.device == 'cpu':
        cpu_fork(paper, source_run, recovery_run, c, args.variant, args.seed)
        install_cpu(paper, source_run, recovery_run)
    paper.cached_job = cohort_cached_job(paper, paper.cached_job)
    execution = dict(entry='paper-recovery-20261109', experiment=args.experiment,
        variant=args.variant, seed=args.seed, source=str(source), output=str(out),
        device=args.device, resume=True, total_wall_limit_disabled=args.no_total_wall_limit,
        solver_limit_seconds=c['time_limit'], frozen_source_unchanged=True,
        cpu_gpu_bitwise_equivalence=False)
    base_budget = unlimited_budget(paper) if args.no_total_wall_limit else paper.budget
    paper.budget = guarded_budget(paper, base_budget, execution)
    print(json.dumps(execution,ensure_ascii=False),flush=True)
    # Invoke the frozen training function directly so a duplicate launcher does
    # not enter upstream's PID-only exception handler (container PIDs repeat).
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.experiment == 'trainmode':
        paper.train(c,args.variant,args.seed,True,args.device)
    else:
        paper.train(c,args.variant,args.seed,True)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('Another job owns this condition. Keep its execution.lock; do not submit a duplicate.', file=sys.stderr)
        sys.exit(2)
