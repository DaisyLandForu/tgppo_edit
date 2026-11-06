"""Explicit prepare/train/eval entry points. No experiment runs on import."""
import argparse
import json
from pathlib import Path
from project.runtime import (config_with_protocol, seed_all, build_agent, save_checkpoint,
    load_agent, make_job, job_identity, run_jobs, import_trajectories, cpu_seconds)
from project.utils.protocol import validate_baseline, fingerprint, write_json


def main(command=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if command is None:
        parser.add_argument('command', choices=['prepare', 'train', 'eval'])
    parser.add_argument('--instance-dir', required=True, type=Path)
    parser.add_argument('--instances', nargs='+', required=True)
    parser.add_argument('--cutoffs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--time-limit', type=float, required=True)
    parser.add_argument('--solver-seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--training-seed', type=int, default=11)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--watchdog', type=float, default=180.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--jobs-per-batch', type=int, default=4)
    args = parser.parse_args()
    command = command or args.command
    if args.time_limit < 1 or args.watchdog <= args.time_limit or args.batches < 1:
        parser.error('Use time >= 1, watchdog > time, batches >= 1')
    if args.jobs_per_batch < 1 or len(set(args.instances)) != len(args.instances):
        parser.error('Positive batch jobs and unique instances required')
    if command == 'train' and args.baseline is None:
        parser.error('Training requires measured matching --baseline')
    if command == 'eval' and args.checkpoint is None:
        parser.error('Evaluation requires --checkpoint')
    if args.output.exists():
        parser.error('Use a new --output directory (prevents stale results)')
    args.output.mkdir(parents=True)
    cutoffs = json.loads(args.cutoffs.read_text())
    pairs = [(name, seed) for name in args.instances for seed in args.solver_seeds]
    def job(job_id, pair, mode, checkpoint=None, baseline=None):
        return make_job(job_id, args.instance_dir, pair[0], args.time_limit, pair[1],
                        args.training_seed, cutoffs, mode, checkpoint, baseline)
    write_json(args.output / 'invocation.json', {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(args).items()})
    if command in ('prepare', 'eval'):
        mode = 'native' if command == 'prepare' else 'eval'
        jobs = [job('{}-{:03d}-s{}'.format(mode, i, pair[1]), pair, mode, args.checkpoint)
                for i, pair in enumerate(pairs)]
        rows = run_jobs(jobs, args.output / 'jobs', args.workers, args.watchdog)
        if command == 'prepare':
            baselines = {}
            for j, row in zip(sorted(jobs, key=lambda j: j['job_id']), rows):
                identity = job_identity(j)
                baseline = dict(identity=identity, result=row)
                if row.get('valid'):
                    validate_baseline(baseline, identity)
                baselines[fingerprint(identity)] = baseline
            write_json(args.output / 'baselines.json', baselines)
        write_json(args.output / 'summary.json', dict(command=command, jobs=len(rows),
            valid=sum(bool(r.get('valid')) for r in rows), failures=sum(not r.get('valid') for r in rows)))
        if any(not r.get('valid') for r in rows):
            raise SystemExit('Failed jobs retained in results.json')
        return
    config = config_with_protocol(json.loads(args.config.read_text()) if args.config else {})
    seed_all(args.training_seed)
    resume_identity = dict(instances=args.instances, solver_seeds=args.solver_seeds,
                           training_seed=args.training_seed, time_limit=args.time_limit,
                           jobs_per_batch=args.jobs_per_batch, cutoffs_sha256=fingerprint(cutoffs))
    next_batch, accumulated_worker_cpu = 0, 0.
    if args.checkpoint:
        agent, payload = load_agent(args.checkpoint)
        if payload['config'] != config or payload['progress']['run_identity'] != resume_identity:
            raise ValueError('Resume configuration or data protocol mismatch')
        next_batch = payload['progress']['next_batch']
        accumulated_worker_cpu = payload['progress']['worker_cpu_seconds']
    else:
        agent = build_agent(config)
    baselines = json.loads(args.baseline.read_text())
    # Validate all required pairs before launching any training worker.
    for i, pair in enumerate(pairs):
        identity = job_identity(job('preflight-' + str(i), pair, 'collect'))
        validate_baseline(baselines.get(fingerprint(identity)), identity)
    write_json(args.output / 'config.json', config)
    history = []
    learner_start_cpu = cpu_seconds()
    start_batch = next_batch
    for batch in range(start_batch, start_batch + args.batches):
        progress = dict(next_batch=batch, run_identity=resume_identity,
                        worker_cpu_seconds=accumulated_worker_cpu)
        checkpoint = args.output / ('before-batch-{:03d}.pt'.format(batch))
        save_checkpoint(checkpoint, agent, config, progress)
        jobs = []
        for i in range(args.jobs_per_batch):
            pair = pairs[(batch * args.jobs_per_batch + i) % len(pairs)]
            j = job('collect-b{:03d}-j{:03d}'.format(batch, i), pair, 'collect', checkpoint)
            j['baseline'] = baselines[fingerprint(job_identity(j))]
            jobs.append(j)
        batchdir = args.output / ('batch-{:03d}'.format(batch))
        rows = run_jobs(jobs, batchdir, args.workers, args.watchdog)
        accumulated_worker_cpu += sum(r.get('worker_cpu_seconds') or 0. for r in rows)
        import_trajectories(agent, rows, batchdir)
        before = {k: v.detach().clone() for k, v in agent.actor_network.state_dict().items()}
        metrics = agent.learn()
        metrics['actor_parameter_l2_change'] = sum((v.detach() - before[k]).double().square().sum().item()
            for k, v in agent.actor_network.state_dict().items()) ** .5
        metrics.update(batch=batch, worker_cpu_seconds_cumulative=accumulated_worker_cpu,
                       learner_cpu_seconds_this_invocation=cpu_seconds() - learner_start_cpu)
        history.append(metrics)
        write_json(args.output / 'updates.json', history)
        save_checkpoint(args.output / 'latest.pt', agent, config,
            dict(next_batch=batch + 1, run_identity=resume_identity,
                 worker_cpu_seconds=accumulated_worker_cpu))
        print(json.dumps(metrics, sort_keys=True), flush=True)
    write_json(args.output / 'summary.json', dict(command='train',
        batches_this_invocation=len(history), final_policy_version=agent.policy_version,
        worker_cpu_seconds_cumulative=accumulated_worker_cpu,
        learner_process_cpu_seconds_this_invocation=cpu_seconds(),
        cpu_accounting='worker RUSAGE_SELF + learner RUSAGE_SELF; no RUSAGE_CHILDREN',
        updated_batches_this_invocation=sum(bool(m['updated']) for m in history)))


if __name__ == '__main__':
    main()
