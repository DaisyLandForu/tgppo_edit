"""Resume six explicit latest branches; resource changes only, no new PPO protocol."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESEARCH_ROOT', str(PACKAGE.parents[2])))
RUN = ROOT/'tgppo/runs/paper-device-resume-20261116'
EXPERIMENTS = PACKAGE.parent
sys.path.insert(0, str(EXPERIMENTS/'paper-recovery-20261109/code'))
import recover


def condition(variant, seed):
    experiment = 'trainmode' if variant == 'b0' else 'aligned'
    original = ROOT/'tgppo/runs'/recover.PACKAGES[experiment]
    old_cpu = ROOT/'tgppo/runs/paper-recovery-20261109'/('cpu-'+experiment)
    source_run = original if variant == 'b0' else old_cpu
    label = '{}-s{}'.format(variant, seed)
    return dict(experiment=experiment, device='cuda' if seed == 11 else 'cpu',
        source_run=source_run, source=source_run/'training'/label,
        guards=sorted({original/'training'/label, old_cpu/'training'/label}),
        output=RUN/('gpu' if seed == 11 else 'cpu')/'training'/label)


def convert(payload, device, cleanup=None):
    result = copy.deepcopy(payload)
    result['config']['learner_device'] = device
    result['config_sha256'] = hashlib.sha256(json.dumps(result['config'],
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    if cleanup is not None:
        result['progress']['collection_directory'] = str(cleanup)
    return result


@contextlib.contextmanager
def source_locks(paper, spec):
    # Hold both old device branches while executing the new continuation. Locks
    # protect files/conditions; they do not parallelize optimizer updates.
    with contextlib.ExitStack() as stack:
        for folder in spec['guards']:
            if folder.exists():
                stack.enter_context(paper.lock(folder/'execution.lock', blocking=False))
        yield


def fork_locked(paper, c, spec, variant, seed):
    """Caller owns source condition locks and the import lock; no solver/CUDA."""
    import torch
    from project import runtime as rt
    source, out = spec['source'], spec['output']
    marker = out/'device_continuation.json'
    source_sha = paper.sha(source/'latest.pt')
    release_sha = paper.sha(paper.PACKAGE/'docs/release.json')
    if marker.exists():
        info = paper.read(marker)
        if (info['source'] != str(source) or info['source_latest_sha256'] != source_sha
                or info['upstream_release_sha256'] != release_sha
                or info['source_budget_sha256'] != paper.sha(source/'budget.json')
                or info['device_after'] != spec['device']):
            raise ValueError('Source/protocol changed after handoff; reconcile branches, do not overwrite continuation')
        if not (out/'latest.pt').exists():
            raise ValueError('Continuation checkpoint missing')
        return info
    if out.exists():
        raise ValueError('Uncommitted destination exists; inspect before retry: '+str(out))
    ledger = paper.read(source/'budget.json')
    if any('elapsed_seconds' not in a for a in ledger.get('attempts', [])):
        raise ValueError('Source budget has an unfinished attempt; inspect before migration')
    latest = rt.read_checkpoint(source/'latest.pt')
    if latest['config'] != paper.configuration(c, variant, seed, latest['config']['learner_device']):
        raise ValueError('Unexpected source algorithm configuration')
    if latest['progress']['phase'] != 'ready' or latest['memory']['num_transitions']:
        raise ValueError('Committed checkpoint must be a ready episode boundary')
    converted = convert(latest, spec['device'])
    staging = out.with_name(out.name+'.import-'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    torch.save(converted, staging/'latest.pt')
    pending_info = None
    pending_file = source/'pending.pt'
    if pending_file.exists():
        pending = rt.read_checkpoint(pending_file)
        n, m = latest['progress']['next_episode'], pending['progress']['next_episode']
        if pending['config'] != latest['config'] or m > n:
            raise ValueError('Invalid pending cursor/configuration')
        if m == n:
            if pending['progress']['phase'] != 'update_pending':
                raise ValueError('Invalid pending phase')
            cleanup_relative = Path('imported-pending')/('episode-'+str(n+1))
            (staging/cleanup_relative).mkdir(parents=True)
            torch.save(convert(pending, spec['device'], out/cleanup_relative), staging/'pending.pt')
            pending_info = dict(source_sha256=paper.sha(pending_file), episode=n+1,
                transitions=pending['memory']['num_transitions'],
                source_collection=pending['progress']['collection_directory'],
                source_trajectory_preserved=True)
    if paper.sha(source/'latest.pt') != source_sha:
        raise ValueError('Source changed during migration')
    info = dict(source=str(source), source_latest_sha256=source_sha,
        source_budget_sha256=paper.sha(source/'budget.json'),
        upstream_release_sha256=release_sha, experiment=spec['experiment'],
        device_before=latest['config']['learner_device'], device_after=spec['device'],
        next_episode=latest['progress']['next_episode'], pending=pending_info,
        destination_initial_sha256=paper.sha(staging/'latest.pt'),
        weights_optimizers_memory_existing_rng_preserved=True,
        cross_device_bitwise_equivalence=False,
        cuda_rng_present='cuda' in latest['rng'], source_preserved=True,
        created_epoch=time.time(), algorithm_protocol_unchanged=True)
    paper.write(staging/'budget.json', ledger)
    paper.write(staging/'state.json', dict(status='device_continuation_ready', **converted['progress']))
    paper.write(staging/'device_continuation.json', info)
    staging.replace(out)
    return info


def install(paper, spec, args):
    import torch
    import server
    import checks
    from project.agent.agent import Agent
    configuration, preflight, target_checks = paper.configuration, server.preflight, checks.target_checks
    device = spec['device']
    paper.RUN = spec['output'].parents[1]
    paper.configuration = lambda c, v, s, device=None: configuration(c, v, s, spec['device'])

    def resource_preflight(root, c, ignored_device):
        checked = dict(c, cpu_allocation_per_training_job=args.cpus,
            workers=args.cpus, ram_gib_per_training_job=args.ram_gib)
        info = preflight(root, checked, device)
        limit = info.get('memory_limit_bytes')
        if limit is not None and limit < args.ram_gib*1024**3:
            raise RuntimeError('Requested RAM minimum {}GiB exceeds allocation'.format(args.ram_gib))
        info['execution_resource_override'] = dict(cpus=args.cpus, ram_gib=args.ram_gib,
            learner_threads=args.learner_threads, collector_threads=1,
            episode_collectors=1, algorithm_config_unchanged=True)
        return info

    server.preflight = resource_preflight
    checks.target_checks = lambda c, seed, folder, ignored_device: target_checks(c, seed, folder, device)
    # Reuse only matching CPU/software/solver identities, copying cache bundles.
    native = paper.native
    for source_run in dict.fromkeys([spec['source_run'],
            ROOT/'tgppo/runs'/recover.PACKAGES[spec['experiment']],
            ROOT/'tgppo/runs/paper-recovery-20261109'/('cpu-'+spec['experiment'])]):
        native = recover.readonly_native(paper, native, source_run)
    paper.native = native
    paper.cached_job = recover.cohort_cached_job(paper, paper.cached_job)
    original_learn = Agent.learn

    def threaded_learn(agent, *positional, **keywords):
        # seed_all resets to one before EVERY update. Change only the synchronous
        # learner, after that reset, and restore one before subsequent sampling.
        before = torch.get_num_threads()
        torch.set_num_threads(args.learner_threads)
        try:
            return original_learn(agent, *positional, **keywords)
        finally:
            torch.set_num_threads(before)
    Agent.learn = threaded_learn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'status', 'run'])
    parser.add_argument('--variant', choices=['b0', 'mlp', 'mlp_bc'], required=True)
    parser.add_argument('--seed', type=int, choices=[11, 22], required=True)
    parser.add_argument('--cpus', type=int, choices=[4, 8], default=4)
    parser.add_argument('--ram-gib', type=int, choices=[32, 48, 64], default=48)
    parser.add_argument('--learner-threads', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-total-wall-limit', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        release = json.loads((PACKAGE/'release.json').read_text())
        if not release.get('passed'):
            raise ValueError('Preparation checks have not passed')
        for relative, digest in release['files'].items():
            if recover.sha(PACKAGE/relative) != digest:
                raise ValueError('Changed continuation entry: '+relative)
    spec = condition(args.variant, args.seed)
    if args.learner_threads > args.cpus:
        parser.error('learner threads cannot exceed allocated CPUs')
    if spec['device'] == 'cuda' and args.learner_threads != 1:
        parser.error('GPU learners retain one CPU thread; select --learner-threads 1')
    # prepare/status never access CUDA, even for a CUDA destination checkpoint.
    if args.command != 'run' or spec['device'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    paper = recover.boot(spec['experiment'])
    c = recover.verify(paper)
    if args.command == 'status':
        data = {k:str(v) for k,v in spec.items() if k != 'guards'}
        for filename in ['state.json', 'device_continuation.json']:
            path = spec['output']/filename
            if path.exists(): data[filename] = paper.read(path)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if args.command == 'run' and not (args.resume and args.no_total_wall_limit):
        parser.error('run requires explicit --resume --no-total-wall-limit; retain historical budget')
    with source_locks(paper, spec):
        with paper.lock(RUN/'import-locks'/(spec['output'].name+'.lock'), blocking=False):
            imported = fork_locked(paper, c, spec, args.variant, args.seed)
        print(json.dumps(dict(command=args.command, output=str(spec['output']), handoff=imported), ensure_ascii=False), flush=True)
        if args.command == 'prepare':
            return
        install(paper, spec, args)
        execution = dict(entry=PACKAGE.name, source=str(spec['source']), output=str(spec['output']),
            variant=args.variant, seed=args.seed, device=spec['device'], cpus=args.cpus,
            ram_gib=args.ram_gib, learner_threads=args.learner_threads,
            resume=True, total_wall_limit_disabled=True, solver_limit_seconds=c['time_limit'],
            original_algorithm_configuration_preserved=True, cross_device_bitwise_equivalence=False)
        paper.budget = recover.guarded_budget(paper, recover.unlimited_budget(paper), execution)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(EXPERIMENTS/'paper-resource-resume-20261109/code'))
        from resource_resume import MemoryMonitor, verify as verify_resource_wrapper
        verify_resource_wrapper()
        memory_log = RUN/'memory'/('{}-{}-{}.json'.format(spec['output'].name, int(time.time()), os.getpid()))
        print(json.dumps(execution), flush=True)
        with MemoryMonitor(memory_log):
            if spec['experiment'] == 'trainmode':
                paper.train(c, args.variant, args.seed, True, spec['device'])
            else:
                paper.train(c, args.variant, args.seed, True)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('This condition is locked by another job. Keep locks; do not submit duplicates.', file=sys.stderr)
        sys.exit(2)
