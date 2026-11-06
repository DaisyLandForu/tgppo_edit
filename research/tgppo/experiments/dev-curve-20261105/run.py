"""Evaluate frozen MLP BC initialization / PPO20 / PPO100, on development only."""
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

from common import BASE, BUDGET, DOCS, EPISODES, NAMES, OUT, ROOT, SEEDS, read, sha, specs, write


class BudgetReached(RuntimeError):
    pass


def main():
    import torch
    import numpy as np
    import launch
    import server
    from project import runtime as rt
    import cpu_refinalize_cached as support

    started = read(OUT / 'bootstrap.json')['started_epoch']
    state = dict(status='preparing', pid=os.getpid(), host=os.uname().nodename,
        started_epoch=started, budget_seconds=BUDGET, requested_tasks=70, workers=4,
        training_steps=0, gpu=False, data_split='development')
    write(OUT / 'status.json', state)

    def expire(*args):
        raise BudgetReached('40-minute execution cap, reserving time for cleanup/report')

    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, BUDGET - 40 - (time.time() - started)))
    models = {}
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        launch.verify()
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task')) == 1
        config = launch.protocol('mlp_bc')[0]
        assert tuple(config['development']) == NAMES and config['time_limit'] == 120
        info = server.preflight(ROOT, config, 'cpu')
        assert info['effective_cpu_count'] >= 4
        cores = info['allowed_cpus'][:4]
        os.sched_setaffinity(0, cores)
        cache = Path(tempfile.mkdtemp(prefix='tgppo-dev-curve-', dir='/tmp'))
        assert cache.stat().st_dev != ROOT.stat().st_dev, 'Use node-local /tmp outside shared Ceph'
        (cache / 'actors').mkdir()
        (cache / 'instances').mkdir()
        (OUT / 'frozen').mkdir()
        for seed in SEEDS:
            source = BASE / 'mlp_bc/training' / ('B0-s' + str(seed))
            bc = read(source / 'pretrain/result.json')
            initial = read(source / 'initialization.json')
            original = source / 'pretrain/actor.pt'
            assert bc['completed_steps'] == 1000 and bc['selected_step'] == 100
            assert sha(original) == bc['actor_sha256'] == initial['bc_actor_sha256']
            ppo = rt.read_checkpoint(source / 'actor-0100.pt', allow_actor_only=True)
            assert ppo['config']['server_training_seed'] == seed and ppo['config']['actor_kind'] == 'mlp'
            payload = torch.load(original, map_location='cpu')
            assert payload['schema'] == 'light-bc-v1' and payload['signature'] == bc['signature']
            for episode in EPISODES:
                label = 'ep{}-s{}'.format(episode, seed)
                origin = original if episode == 0 else source / ('actor-{:04d}.pt'.format(episode))
                frozen = OUT / 'frozen' / (label + '.pt')
                if episode == 0:
                    converted = dict(schema_version='actor-only-v1', config=ppo['config'],
                        config_sha256=ppo['config_sha256'], actor=payload['actor'],
                        provenance=dict(bc_source_sha256=sha(original), bc_selected_step=100, ppo_episodes=0))
                    torch.save(converted, frozen)
                    policy, loaded = rt.load_actor(frozen)
                    assert set(loaded['actor']) == set(payload['actor'])
                    assert all(torch.equal(loaded['actor'][key], tensor) for key, tensor in payload['actor'].items())
                    assert server.digest_actor(policy.actor_network) == initial['actor_fingerprint']
                else:
                    exported = rt.read_checkpoint(origin, allow_actor_only=True)
                    checkpoint = rt.read_checkpoint(source / ('episode-{:04d}.pt'.format(episode)))
                    assert checkpoint['progress']['next_episode'] == episode
                    assert checkpoint['config'] == exported['config'] == ppo['config']
                    assert set(checkpoint['actor']) == set(exported['actor'])
                    assert all(torch.equal(checkpoint['actor'][key], tensor) for key, tensor in exported['actor'].items())
                    del checkpoint, exported
                    shutil.copyfile(origin, frozen)
                frozen.chmod(0o444)
                local = cache / 'actors' / frozen.name
                shutil.copyfile(frozen, local)
                local.chmod(0o444)
                assert sha(local) == sha(frozen)
                models[label] = dict(episode=episode, seed=seed, source=str(origin), source_sha256=sha(origin),
                    frozen_actor=str(frozen), actor_sha256=sha(frozen), cached_actor=str(local),
                    selected_bc_step=100, initialization_sha256=sha(source / 'initialization.json'))
        selected = []
        for name in NAMES:
            source = ROOT / 'tgppo/datasets/train_instances' / (name + '.mps.gz')
            target = cache / 'instances' / source.name
            shutil.copyfile(source, target)
            assert sha(target) == sha(source) == config['instance_hashes'][name]
            selected.append(dict(instance=name, path=str(source), sha256=sha(source)))
        support.torch, support.np, support.rt = torch, np, rt
        paths = [m['cached_actor'] for m in models.values()]
        expected = support.signatures(paths)
        assert len(os.listdir('/proc/self/task')) == 1 and not torch.cuda.is_initialized()
        original_context = rt.mp.get_context
        context = original_context('fork')
        receive, send = context.Pipe(duplex=False)
        child = context.Process(target=support.probe_child, args=(send, paths))
        child.start()
        send.close()
        try:
            if not receive.poll(60):
                raise RuntimeError('Fork verification timed out; no optimization started')
            observed = receive.recv()
            child.join(timeout=5)
            assert observed.get('values') == expected and observed['threads'] == 1 and not observed['cuda_initialized']
        finally:
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
            receive.close()
        write(OUT / 'cpu_fork_verification.json', dict(passed=True, parent_threads=1, child=observed,
            actual_policy_probabilities_actions_bitwise_equal=True, solver_probe_constructed_not_optimized=True))
        rt.mp.get_context = lambda method=None: original_context('fork' if method == 'spawn' else method)
        cutoffs = read(ROOT / 'tgppo/datasets/cutoffs.json')
        tasks = []
        for item in specs():
            model = models.get(item['label'])
            seed = item['training_seed'] if model else 11
            job = server.bound_job(ROOT, config, cutoffs, item['instance'], item['solver_seed'], seed,
                'eval' if model else 'native', model['cached_actor'] if model else None,
                job_id=item['job_id'], instances=cache / 'instances')
            job['sampling_seed'] = rt.action_seed(seed, item['canonical_sampling_job_id'])
            tasks.append(dict(item, job=job))
        write(OUT / 'manifest.json', dict(models=models, tasks=tasks, selected=selected,
            stack=info['stack'], host=info['host'], cpu_affinity=cores, workers=4, threads_per_worker=1,
            worker_start_method='fork', scope='five existing development instances; not held-out testing',
            canonical_action_rng='dev-{instance}-s{solver_seed}; native uses native-',
            native_reference='fresh10 on same host as all60 model tasks',
            runner_sha256=sha(__file__), plan_sha256=sha(DOCS / 'PLAN.md')))
        raw = Path(tempfile.mkdtemp(prefix='tgppo-dev-curve-raw-', dir='/tmp'))
        (OUT / 'raw').symlink_to(raw, target_is_directory=True)
        write(OUT / 'LOCAL_RAW_OUTPUT.json', dict(local_directory=str(raw), shared_link=str(OUT / 'raw')))
        state.update(status='running', stack=info['stack'], setup_seconds=time.time() - started)
        write(OUT / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        rows = rt.run_jobs([t['job'] for t in tasks], OUT / 'raw', max_workers=4, watchdog_seconds=180)
        write(OUT / 'evaluation_rows.json', rows)
        for row in rows:
            assert row['valid'] and row['worker_cpu_complete'] and not row['cuda_initialized']
            if row['mode'] == 'eval':
                assert row['callback_seconds']['critic'] == 0 and row['fallback_action_count'] == 0
        for model in models.values():
            assert sha(model['source']) == model['source_sha256']
            assert sha(model['frozen_actor']) == sha(model['cached_actor']) == model['actor_sha256']
        assert len(rows) == 70 and not torch.cuda.is_initialized()
        state.update(status='complete', valid_tasks=70, weights_unchanged=True,
            summed_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows))
    except BudgetReached as exc:
        state.update(status='budget_exhausted', error=str(exc))
        raise
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state.update(engine_wall_seconds=time.time() - started,
            raw_result_files=len(list((OUT / 'raw').glob('t*.json'))))
        write(OUT / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
