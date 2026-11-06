"""Stage-2 experiment plumbing; original networks/reward and explicit finite tasks."""
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import resource
import time
import traceback
import numpy as np
import torch
from project.agent.agent import Agent, ActorPolicy
from project.policy import Actor, Critic
from project.environment import Environment
from project.reward import RewardH3
from project.utils.settings import state_dims, settings
from project.utils.protocol import (PROTOCOL_VERSION, baseline_identity, fingerprint,
                                    validate_baseline, write_json)

DEFAULT_CONFIG = dict(hidden_dim=256, num_layers=5, num_heads=8, dropout=.05,
    actor_lr=.00024, critic_lr=.00012, policy_clip=.16, entropy_weight=.003,
    gamma=.97, gae_lambda=.92, batch_size=256, n_epochs=3,
    reward_function='reward_h3')
ARCH_KEYS = ('hidden_dim', 'num_layers', 'num_heads', 'dropout')
PPO_KEYS = ('policy_clip', 'entropy_weight', 'gamma', 'gae_lambda', 'batch_size', 'n_epochs')


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def action_seed(training_seed, job_id):
    return int(hashlib.sha256((str(training_seed) + ':' + job_id).encode()).hexdigest()[:8], 16)


def config_with_protocol(config):
    result = dict(DEFAULT_CONFIG, **config)
    if result['reward_function'] != 'reward_h3':
        raise ValueError('Stage2 freezes upstream H3')
    result.update(state_dims=state_dims.copy(), protocol_version=PROTOCOL_VERSION,
                  effective_actor_dropout=0., critic_training_dropout=result['dropout'],
                  upstream_commit='4f90ccddd7999779c93ff2425ca8dd056113a523')
    return result


def build_agent(config, training=True, device=None):
    device = device or (config.get("learner_device", "cpu") if training else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA learner requested but no CUDA device is available")
    arch = dict(config['state_dims'], **{k: config[k] for k in ARCH_KEYS})
    actor, critic = Actor(**arch), Critic(**arch)
    agent_type = Agent
    if 'online_group' in config:
        from project.online import OnlineAgent
        from project.policy.calibrated_critic import CalibratedCritic
        # Preserve public model initialization and the subsequent PPO RNG stream.
        # Search-only modules have a separate deterministic initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['extra_initialization_seed'])
            critic = CalibratedCritic(critic, group=config['online_group'], loss='mse',
                mean=config['target_mean'], std=config['target_std'])
        agent_type = OnlineAgent
    actor, critic = actor.to(device), critic.to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config['actor_lr']) if training else None
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config['critic_lr']) if training else None
    result = agent_type(actor, actor_opt, critic, critic_opt, device=device,
                 state_dims=config['state_dims'], **{k: config[k] for k in PPO_KEYS})
    if getattr(result, 'online_context', False):
        result.task_time_limit = config.get('task_time_limit', 120.)
    return result


def rng_state():
    result = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_initialized():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, include_cuda=True):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if include_cuda and 'cuda' in state:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG restoration requested without CUDA')
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA visible-device count changed during resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, agent, config, progress=None):
    if agent.actor_optimizer is None or agent.critic_optimizer is None:
        raise ValueError('A resumable checkpoint requires both optimizers')
    payload = dict(schema_version=3 if getattr(agent, 'online_context', False) else 2, protocol_version=PROTOCOL_VERSION, config=copy.deepcopy(config),
        config_sha256=fingerprint(config), actor=agent.actor_network.state_dict(),
        critic=agent.critic_network.state_dict(), actor_optimizer=agent.actor_optimizer.state_dict(),
        critic_optimizer=agent.critic_optimizer.state_dict(), policy_version=agent.policy_version,
        update_counter=agent.update_counter, memory=agent.memory.export_dict(), rng=rng_state(),
        progress=progress or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def read_checkpoint(path, allow_actor_only=False):
    payload = torch.load(path, map_location='cpu')  # Trusted, locally produced checkpoints only.
    if allow_actor_only and payload.get('schema_version') == 'actor-only-v1':
        if fingerprint(payload['config']) != payload['config_sha256']:
            raise ValueError('Actor configuration hash mismatch')
        return payload
    if payload.get('schema_version') not in (2, 3) or payload.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Checkpoint requires explicit architecture/protocol; no default guessing')
    if (payload['schema_version'] == 3) != ('online_group' in payload['config']):
        raise ValueError('Online checkpoint discriminator mismatch')
    if fingerprint(payload['config']) != payload['config_sha256']:
        raise ValueError('Checkpoint configuration hash mismatch')
    return payload


def load_agent(path, training=True, restore_random_state=True, device=None):
    payload = read_checkpoint(path)
    agent = build_agent(payload['config'], training, device=device)
    agent.actor_network.load_state_dict(payload['actor'])
    agent.critic_network.load_state_dict(payload['critic'])
    agent.policy_version, agent.update_counter = payload['policy_version'], payload['update_counter']
    if training:
        agent.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        agent.memory.import_dict(payload['memory'])
    if restore_random_state:
        restore_rng(payload['rng'], include_cuda=training and str(agent.device).startswith('cuda'))
    return agent, payload


def load_actor(path):
    payload = read_checkpoint(path, allow_actor_only=True)
    config = payload['config']
    arch = dict(config['state_dims'], **{k: config[k] for k in ARCH_KEYS})
    actor = Actor(**arch)  # This path never constructs a Critic or optimizer.
    actor.load_state_dict(payload['actor'])
    return ActorPolicy(actor), payload


def make_job(job_id, instance_dir, name, time_limit, solver_seed, training_seed,
             cutoffs, mode, checkpoint=None, baseline=None, setting='sandbox'):
    record = cutoffs[name]
    if record.get('status') != 'optimal' or not record.get('source'):
        raise ValueError('Cutoff lacks verified optimal source: ' + name)
    provenance = dict(verified=True, source=record['source'], status=record['status'],
                      record_sha256=fingerprint(record))
    return dict(job_id=job_id, instance=str(Path(instance_dir) / (name + '.mps.gz')),
        instance_name=name, limits=dict(time_limit=float(time_limit), node_limit=-1),
        params=settings[setting].copy(), solver_seed=int(solver_seed), training_seed=int(training_seed),
        sampling_seed=action_seed(training_seed, job_id), cutoff=record['cutoff'],
        cutoff_provenance=provenance, mode=mode, checkpoint=str(checkpoint) if checkpoint else None,
        baseline=baseline)


def job_identity(job):
    return baseline_identity(job['instance'], job['solver_seed'], job['limits'], job['params'],
                             job['cutoff'], job['cutoff_provenance'])


def _worker(job, result_path, trajectory_path):
    start = time.perf_counter()
    result = dict(job_id=job['job_id'], mode=job['mode'], instance_name=job.get('instance_name'),
                  solver_seed=job.get('solver_seed'), sampling_seed=job.get('sampling_seed'))
    try:
        # Fault injection is only used by scheduler tests, never by experiment jobs.
        if job.get('_test_failure'):
            raise RuntimeError('Injected worker failure')
        if job.get('_test_sleep'):
            time.sleep(job['_test_sleep'])
        seed_all(job['training_seed'])
        if job.get('address_space_limit_bytes'):
            limit = int(job['address_space_limit_bytes'])
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            result['address_space_limit_bytes'] = limit
        policy = None
        reward = None
        if job['mode'] == 'collect':
            policy, payload = load_agent(job['checkpoint'], training=False, restore_random_state=False)
            reward = RewardH3()
        elif job['mode'] == 'eval':
            policy, payload = load_actor(job['checkpoint'])
        # Initialization must not consume the action RNG stream.
        seed_all(job['sampling_seed'])
        search_recorder = None
        online_frontier = (job['mode'] == 'collect' and getattr(policy, 'online_context', False)
                           and policy.critic_network.group != 'R0')
        if job.get('collect_search_features') or online_frontier:
            if job['mode'] != 'collect':
                raise ValueError('Search features forbidden during deployment')
            from project.search_features import SearchRecorder
            search_recorder = SearchRecorder(job['limits']['time_limit'])
        env = Environment('cpu', policy, state_dims, job['limits'], job['params'],
                          job['solver_seed'], reward, mode=job['mode'],
                          search_recorder=search_recorder)
        env.reset(job['instance'], job['cutoff'], job.get('baseline'),
                  job['cutoff_provenance'], episode_id=job['job_id'])
        _, info, _ = env.run_episode()
        result.update(info)
        if job['mode'] == 'native':
            result['identity'] = env.identity
        if job['mode'] == 'collect' and info['valid']:
            result['policy_version'] = policy.policy_version
            trajectory = policy.memory.export_dict()
            if search_recorder is not None:
                trajectory['search_features'] = search_recorder.export(trajectory)
                result['search_extraction_wall_seconds'] = search_recorder.seconds
            result['num_transitions'] = trajectory['num_transitions']
            result['num_terminal_transitions'] = sum(r['done'] for e in trajectory['episodes']
                                                    for r in e['transitions'])
            with open(trajectory_path, 'wb') as stream:
                pickle.dump(trajectory, stream, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        result.update(valid=False, task_terminal=False, status='worker_error', error=traceback.format_exc())
    result.update(worker_wall_seconds=time.perf_counter() - start, worker_cpu_seconds=cpu_seconds(),
                  worker_cpu_complete=True, cuda_initialized=torch.cuda.is_initialized(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), reward_version='upstream_h3_t{:g}'.format(job.get('limits', {}).get('time_limit', 0)))
    write_json(result_path, result)


def _live_cpu(pid):
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
        return (float(fields[11]) + float(fields[12])) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError):
        return None


def _run_jobs(jobs, output_dir, max_workers=1, watchdog_seconds=180., _active=None):
    """Each task owns one killable process. Returned order depends only on job IDs."""
    if max_workers < 1 or watchdog_seconds <= 0:
        raise ValueError('Positive workers and watchdog required')
    jobs = sorted(jobs, key=lambda j: j['job_id'])
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job_id = job['job_id']
        if not job_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Unsafe job ID')
        if any((output_dir / (job_id + suffix)).exists() for suffix in ('.json', '.pkl')):
            raise FileExistsError('Refuse stale worker output: ' + job_id)
    context = mp.get_context('spawn')
    pending, rows = list(jobs), []
    active = _active if _active is not None else {}
    while pending or active:
        while pending and len(active) < max_workers:
            job = pending.pop(0)
            if not job['job_id'].replace('-', '').replace('_', '').isalnum():
                raise ValueError('Unsafe job ID')
            result_path = output_dir / (job['job_id'] + '.json')
            trajectory_path = output_dir / (job['job_id'] + '.pkl')
            if result_path.exists() or trajectory_path.exists():
                raise FileExistsError('Refuse stale worker output: ' + str(result_path))
            process = context.Process(target=_worker, args=(job, result_path, trajectory_path))
            process.start()
            active[job['job_id']] = (job, process, time.monotonic(), result_path, trajectory_path)
        for job_id, (job, process, start, result_path, trajectory_path) in list(active.items()):
            timed_out = process.is_alive() and time.monotonic() - start > watchdog_seconds
            live_cpu = _live_cpu(process.pid) if timed_out else None
            if timed_out:
                process.terminate()
                process.join(timeout=1.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.)
            if timed_out or not process.is_alive():
                process.join(timeout=1.)
                if timed_out or process.exitcode != 0 or not result_path.exists():
                    row = dict(job_id=job_id, instance_name=job.get('instance_name'),
                        mode=job['mode'], valid=False, task_terminal=False,
                        status='watchdog_timeout' if timed_out else 'worker_exit_error',
                        exitcode=process.exitcode, worker_cpu_seconds=live_cpu,
                        worker_cpu_complete=False, wall_seconds=time.monotonic() - start)
                    write_json(result_path, row)
                    if trajectory_path.exists():
                        trajectory_path.unlink()  # Never import a killed worker's partial output.
                else:
                    row = json.loads(result_path.read_text())
                rows.append(row)
                process.close()
                del active[job_id]
        if active:
            time.sleep(.02)
    rows.sort(key=lambda r: r['job_id'])
    write_json(output_dir / 'results.json', rows)
    return rows


def run_jobs(jobs, output_dir, max_workers=1, watchdog_seconds=180.):
    active = {}
    try:
        return _run_jobs(jobs, output_dir, max_workers, watchdog_seconds, active)
    finally:
        # Parent exceptions/interruptions must not leave expensive solves running.
        for _, process, _, _, _ in active.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.)
            process.close()


def import_trajectories(agent, rows, directory):
    # Fail closed on any incomplete job; don't filter failed tasks out of a batch.
    if any(not row.get('valid') for row in rows):
        raise RuntimeError('Invalid collection jobs retained; refusing this PPO batch')
    for row in sorted(rows, key=lambda r: r['job_id']):
        if row['policy_version'] != agent.policy_version:
            raise ValueError('Stale worker policy version')
        with open(Path(directory) / (row['job_id'] + '.pkl'), 'rb') as stream:
            payload = pickle.load(stream)
        agent.memory.import_dict(payload)
