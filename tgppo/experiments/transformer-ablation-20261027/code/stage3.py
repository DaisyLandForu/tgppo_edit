"""Stage-3 only: frozen native baseline, budgeted B0 pilot, development evaluation."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import traceback
import time
import torch
from project.runtime import (make_job, job_identity, run_jobs, config_with_protocol,
    seed_all, build_agent, save_checkpoint, load_agent, import_trajectories, cpu_seconds)
from project.utils.protocol import fingerprint, validate_baseline, sha256, write_json

ROOT = Path('/root/research')
DEFAULT_RUN = ROOT / 'tgppo/runs/stage3'
RUN = DEFAULT_RUN
DOC = ROOT / 'docs/stage3'
PILOT_ORDER = ['lseu', 'stein27', 'misc03', 'pp08aCUTS', 'neos20', 'l152lav',
               'sp150x300d', 'neos21', 'air04', 'air05']


def frozen_config():
    protocol = json.loads((ROOT / 'docs/stage1/protocol_draft.json').read_text())
    return dict(stage=3, authorization='通过，进入阶段3', stage2_accepted=True,
        time_limit=120., watchdog=180., max_workers=4, threads_per_worker=1,
        native_cpu_cap_seconds=9000., pilot_cpu_cap_seconds=1800., pilot_max_episodes=10,
        training_seed=11, solver_seeds=[0, 1, 2], pilot_order=PILOT_ORDER,
        train=protocol['data']['train'], development=protocol['data']['development'],
        instance_hashes=protocol['data']['hashes'],
        pilot_schedule_basis='declared before 120s results; small verified cases first, then larger cases',
        evaluation_checkpoint='final valid budget/episode-limited pilot checkpoint; no development selection',
        pdi_definition='SCIP percent-seconds at T=120; preserve exact extension or overshoot enclosure',
        measurement_version='native-pdi-snapshot-horizon-v2',
        metric_binding='isolated PySCIPOpt1.4.9 + read-only getIntegralSnapshot; same SCIP shared library',
        pdi_uncertainty_rule='report intervals; no ordering claim when paired intervals overlap',
        cpu_affinity=sorted(os.sched_getaffinity(0))[:4])


def prepare(config):
    out = RUN / 'native'
    if out.exists():
        raise FileExistsError(str(out))
    cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    names = config['train'] + config['development']
    # Frozen hashes and sources are checked before any formal solve.
    for name in names:
        path = ROOT / 'tgppo/datasets/train_instances' / (name + '.mps.gz')
        if sha256(path) != config['instance_hashes'][name]:
            raise ValueError('Frozen instance changed: ' + name)
    jobs = [make_job('native-{:02d}-{}-s{}'.format(i, name, seed),
        ROOT / 'tgppo/datasets/train_instances', name, 120., seed, 11, cutoffs, 'native')
        for i, name in enumerate(names) for seed in config['solver_seeds']]
    rows = run_jobs(jobs, out / 'jobs', config['max_workers'], config['watchdog'])
    audit = json.loads((DOC / 'measurement-audit-restart.json').read_text())
    prior_cpu_upper = ((audit['completed_worker_cpu_seconds'] + audit['cpu_reserve_seconds']
        + sum(r['cpu_seconds'] for r in audit['active_children_cpu_snapshots'])) if RUN == DEFAULT_RUN else 0.)
    measured_cpu = sum(row.get('worker_cpu_seconds') or 0. for row in rows) + cpu_seconds() + prior_cpu_upper
    baselines = {}
    for job, row in zip(sorted(jobs, key=lambda j: j['job_id']), rows):
        identity = job_identity(job)
        baseline = dict(identity=identity, result=row)
        if row.get('valid'):
            validate_baseline(baseline, identity)
        baselines[fingerprint(identity)] = baseline
    write_json(out / 'baselines.json', baselines)
    write_json(out / 'summary.json', dict(jobs=len(rows), valid=sum(bool(r.get('valid')) for r in rows),
        worker_cpu_seconds=sum(r.get('worker_cpu_seconds') or 0. for r in rows),
        parent_cpu_seconds=cpu_seconds(), total_cpu_seconds=measured_cpu,
        audit_restart_cpu_upper_seconds=prior_cpu_upper,
        cpu_cap_seconds=config['native_cpu_cap_seconds'],
        within_budget=measured_cpu <= config['native_cpu_cap_seconds']))
    if any(not r.get('valid') for r in rows) or measured_cpu > config['native_cpu_cap_seconds']:
        raise RuntimeError('Native failures or budget issue; all records retained')


class PilotCPULimit(RuntimeError):
    pass


def _cpu_limit(signum, frame):
    signal.signal(signal.SIGXCPU, signal.SIG_IGN)
    raise PilotCPULimit('Pilot combined worker + learner CPU budget reached')


def pilot(config):
    out = RUN / 'pilot'
    if out.exists():
        raise FileExistsError(str(out))
    out.mkdir(parents=True)
    learner_as_limit = 32 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (learner_as_limit, resource.getrlimit(resource.RLIMIT_AS)[1]))
    write_json(out / 'resource_limits.json', dict(worker_address_space_limit_bytes=8*1024**3,
        learner_address_space_limit_bytes=learner_as_limit,
        candidate_truncation=False, action_fallback=False))
    baselines = json.loads((RUN / 'native/baselines.json').read_text())
    cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    params = config_with_protocol(json.loads((ROOT / 'tgppo/configs/paper_hparams.json').read_text()))
    seed_all(config['training_seed'])
    agent = build_agent(params)
    write_json(out / 'config.json', params)
    worker_cpu, completed, history = 0., 0, []
    collected = []
    initial_weights = {k: v.detach().clone() for k, v in agent.actor_network.state_dict().items()}
    save_checkpoint(out / 'initial.pt', agent, params,
                    dict(next_episode=0, worker_cpu_seconds=0., stage=3))
    shutil.copyfile(out / 'initial.pt', out / 'latest.pt')
    stop_reason = 'episode_limit'
    original_limits = resource.getrlimit(resource.RLIMIT_CPU)
    original_handler = signal.getsignal(signal.SIGXCPU)
    try:
        while completed < config['pilot_max_episodes']:
            count = min(4, config['pilot_max_episodes'] - completed)
            remaining = config['pilot_cpu_cap_seconds'] - worker_cpu - cpu_seconds()
            learning_reserve = (max(10., 1.25 * history[-1]['learning_cpu_seconds'] * count / 4.)
                                if history else 10.)
            if remaining < count * config['watchdog'] + learning_reserve:
                stop_reason = 'insufficient_remaining_budget_for_next_complete_batch'
                break
            batch = len(history)
            before = out / ('before-batch-{:02d}.pt'.format(batch))
            save_checkpoint(before, agent, params, dict(next_episode=completed,
                worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds(), stage=3))
            jobs = []
            for i in range(count):
                index = completed + i
                name = config['pilot_order'][index]
                j = make_job('b0-e{:02d}-{}-s0'.format(index, name),
                    ROOT / 'tgppo/datasets/train_instances', name, 120., 0,
                    config['training_seed'], cutoffs, 'collect', before)
                j['address_space_limit_bytes'] = 8 * 1024**3
                identity = job_identity(j)
                j['baseline'] = baselines[fingerprint(identity)]
                validate_baseline(j['baseline'], identity)
                jobs.append(j)
            directory = out / ('batch-{:02d}'.format(batch))
            rows = run_jobs(jobs, directory, min(4, count), config['watchdog'])
            worker_cpu += sum(r.get('worker_cpu_seconds') or 0. for r in rows)
            completed += len(rows)
            collected.extend(rows)
            write_json(out / 'collection_results.json', collected)
            if any(not r.get('valid') for r in rows):
                stop_reason = 'invalid_collection_batch_retained_no_update'
                break
            import_trajectories(agent, rows, directory)
            # Preserve every collected episode even if this update exhausts budget.
            save_checkpoint(out / ('collected-batch-{:02d}.pt'.format(batch)), agent, params,
                dict(next_episode=completed, worker_cpu_seconds=worker_cpu, stage=3))
            deadline = int(config['pilot_cpu_cap_seconds'] - worker_cpu - 5.)
            if deadline <= cpu_seconds():
                stop_reason = 'budget_reached_before_learning'
                break
            signal.signal(signal.SIGXCPU, _cpu_limit)
            resource.setrlimit(resource.RLIMIT_CPU, (deadline, original_limits[1]))
            begin = cpu_seconds()
            write_json(out / 'learning-progress.json', dict(batch=batch, state='starting',
                samples=len(agent.memory), learner_cpu_seconds=begin, worker_cpu_seconds=worker_cpu))
            def log_progress(progress):
                write_json(out / 'learning-progress.json', dict(progress, batch=batch, state='updating',
                    learner_cpu_seconds=cpu_seconds(), worker_cpu_seconds=worker_cpu,
                    learning_cpu_seconds=cpu_seconds()-begin))
            try:
                metrics = agent.learn(progress_callback=log_progress)
            except PilotCPULimit as exc:
                stop_reason = 'cpu_budget_partial_update_discarded_last_valid_checkpoint_retained'
                write_json(out / 'interrupted_update.json', dict(error=str(exc), batch=batch,
                    worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds()))
                break
            except Exception:
                stop_reason = 'learning_error_last_valid_checkpoint_retained'
                write_json(out / 'learning_error.json', dict(error=traceback.format_exc(), batch=batch,
                    worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds()))
                break
            finally:
                resource.setrlimit(resource.RLIMIT_CPU, original_limits)
                signal.signal(signal.SIGXCPU, original_handler)
            metrics.update(batch=batch, episodes_completed=completed, learning_cpu_seconds=cpu_seconds()-begin,
                           worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds())
            metrics['actor_l2_change_from_initial'] = sum(
                (v.detach()-initial_weights[k]).double().square().sum().item()
                for k,v in agent.actor_network.state_dict().items()) ** .5
            history.append(metrics)
            write_json(out / 'updates.json', history)
            save_checkpoint(out / 'latest.pt', agent, params, dict(next_episode=completed,
                worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds(), stage=3))
            print(json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, original_limits)
        signal.signal(signal.SIGXCPU, original_handler)
    latest, payload = load_agent(out / 'latest.pt')
    write_json(out / 'summary.json', dict(stop_reason=stop_reason, episodes_attempted=completed,
        completed_updates=latest.update_counter, policy_version=latest.policy_version,
        checkpoint_episodes=payload['progress']['next_episode'],
        worker_cpu_seconds=worker_cpu, learner_cpu_seconds=cpu_seconds(),
        total_cpu_seconds=worker_cpu+cpu_seconds(), cpu_cap_seconds=config['pilot_cpu_cap_seconds'],
        within_budget=worker_cpu+cpu_seconds() <= config['pilot_cpu_cap_seconds'],
        peak_learner_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if latest.update_counter < 1:
        raise RuntimeError('No effective B0 update; records retained')


def evaluate(config):
    cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    checkpoint = RUN / ('pilot-repaired/latest.pt' if (RUN / 'pilot-repaired/summary.json').exists()
                        else 'pilot/latest.pt')
    write_json(RUN / 'selected_checkpoint.json', dict(checkpoint=str(checkpoint),
        selection='last complete valid update after resource repair, before any development results'))
    before = sha256(checkpoint)
    jobs = [make_job('b0-dev-{:02d}-{}-s{}'.format(i, name, seed),
        ROOT / 'tgppo/datasets/train_instances', name, 120., seed, 11, cutoffs,
        'eval', checkpoint) for i,name in enumerate(config['development']) for seed in config['solver_seeds']]
    for job in jobs:
        job['address_space_limit_bytes'] = 8 * 1024**3
    rows = run_jobs(jobs, RUN / 'evaluation/jobs', 4, config['watchdog'])
    if sha256(checkpoint) != before:
        raise RuntimeError('Evaluation changed checkpoint')
    write_json(RUN / 'evaluation/summary.json', dict(jobs=len(rows),
        valid=sum(bool(r.get('valid')) for r in rows),
        failures=sum(not r.get('valid') for r in rows), checkpoint_sha256=before,
        critic_seconds=sum(r.get('callback_seconds', {}).get('critic', 0.) for r in rows),
        worker_cpu_seconds=sum(r.get('worker_cpu_seconds') or 0. for r in rows),
        parent_cpu_seconds=cpu_seconds()))
    if any(not r.get('valid') for r in rows):
        raise RuntimeError('Development evaluation failures retained')


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'pilot', 'evaluate'])
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN,
                        help='New directory for a separate reproduction; never overwrite existing jobs')
    args = parser.parse_args()
    RUN = args.run_root.resolve()
    config = frozen_config()
    DOC.mkdir(parents=True, exist_ok=True)
    path = DOC / 'execution_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Frozen stage3 execution configuration changed')
    write_json(path, config)
    write_json(RUN / 'execution_config.json', config)
    os.sched_setaffinity(0, config['cpu_affinity'])
    if args.command == 'prepare':
        prepare(config)
    elif args.command == 'pilot':
        pilot(config)
    elif args.command == 'evaluate':
        evaluate(config)


if __name__ == '__main__':
    main()
