"""Frozen BC / paired PPO / supervised PPO, all development tasks on one CPU."""
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from common import BASE, BUDGET, NAMES, OUT, ROOT, TOP, model_specs, read, sha, specs, write


def main():
    import numpy as np
    import torch
    import launch
    import server
    import cpu_refinalize_cached as support
    from project import runtime as rt
    started = read(OUT / 'bootstrap.json')['started_epoch']
    state = dict(status='preparing', pid=os.getpid(), host=os.uname().nodename, requested_tasks=110,
        training_steps=0, gpu=False, started_epoch=started, budget_seconds=BUDGET)
    write(OUT / 'status.json', state)
    def expired(*args):
        raise TimeoutError('50-minute cumulative evaluation cap; cleanup/report reserve')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, BUDGET - 40 - (time.time() - started)))
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        launch.verify()
        if not all(x.get('status') == 'training_complete' for x in launch.statuses().values()):
            raise RuntimeError('All4 training jobs must finish before CPU evaluation; no waiting allocation')
        pairings = {str(s): read(BASE / 'pairing' / ('s' + str(s) + '.json')) for s in (11, 22)}
        for s, pair in pairings.items():
            assert set(pair) == {'control', 'expert'}
            assert pair['control']['initialization'] == pair['expert']['initialization']
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task')) == 1
        configs = {v: launch.protocol(v)[0] for v in ('control', 'expert')}
        config = configs['control']
        assert tuple(config['development']) == NAMES and config['time_limit'] == 120
        info = server.preflight(ROOT, config, 'cpu')
        launch.check_software(info)
        assert info['effective_cpu_count'] >= 4
        cores = info['allowed_cpus'][:4]
        os.sched_setaffinity(0, cores)
        cache = Path(tempfile.mkdtemp(prefix='tgppo-supervised-eval-', dir='/tmp'))
        assert cache.stat().st_dev != ROOT.stat().st_dev
        (cache / 'actors').mkdir()
        (cache / 'instances').mkdir()
        (OUT / 'frozen').mkdir()
        models = {}
        for item in model_specs():
            label, seed, episode = item['label'], item['training_seed'], item['episode']
            variant = 'control' if item['variant'] == 'bc' else item['variant']
            source = BASE / variant / 'training' / ('B0-s' + str(seed))
            actor = source / ('actor-{:04d}.pt'.format(episode))
            checkpoint = source / ('episode-{:04d}.pt'.format(episode))
            exported = rt.read_checkpoint(actor, allow_actor_only=True)
            full = rt.read_checkpoint(checkpoint)
            assert full['config'] == exported['config'] == server.params(ROOT, configs[variant], 'B0', seed, 'cuda')
            assert full['progress']['phase'] == 'ready' and full['progress']['next_episode'] == episode
            assert launch.state_digest(full['actor']) == launch.state_digest(exported['actor'])
            if episode == 0:
                other = rt.read_checkpoint(BASE / 'expert/training' / ('B0-s' + str(seed)) / 'episode-0000.pt')
                assert launch.state_digest(full['actor']) == launch.state_digest(other['actor'])
                assert launch.state_digest(full['critic']) == launch.state_digest(other['critic'])
                del other
            frozen = OUT / 'frozen' / (label + '.pt')
            shutil.copyfile(actor, frozen)
            frozen.chmod(0o444)
            local = cache / 'actors' / frozen.name
            shutil.copyfile(frozen, local)
            local.chmod(0o444)
            assert sha(local) == sha(frozen) == sha(actor)
            models[label] = dict(item, source=str(actor), source_sha256=sha(actor),
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                frozen_actor=str(frozen), cached_actor=str(local), actor_sha256=sha(frozen),
                update_counter=full['update_counter'], optimizer_steps=full['progress']['optimizer_steps'])
            del full, exported
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
                raise RuntimeError('CPU fork check timed out; no optimization started')
            observed = receive.recv()
            child.join(timeout=5)
            assert observed.get('values') == expected and observed['threads'] == 1 and not observed['cuda_initialized']
        finally:
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
            receive.close()
        write(OUT / 'cpu_fork_verification.json', dict(passed=True, child=observed, parent_threads=1))
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
            pairings=pairings, stack=info['stack'], host=info['host'], cpu_affinity=cores,
            workers=4, threads_per_worker=1, native_reference='fresh10 on same host as100 models',
            runner_sha256=sha(__file__), protocol_sha256=sha(TOP / 'docs/PROTOCOL.md')))
        raw = Path(tempfile.mkdtemp(prefix='tgppo-supervised-raw-', dir='/tmp'))
        (OUT / 'raw').symlink_to(raw, target_is_directory=True)
        state.update(status='running', stack=info['stack'], setup_seconds=time.time() - started)
        write(OUT / 'status.json', state)
        rows = rt.run_jobs([t['job'] for t in tasks], OUT / 'raw', max_workers=4, watchdog_seconds=180)
        write(OUT / 'evaluation_rows.json', rows)
        assert len(rows) == 110
        for row in rows:
            assert row['valid'] and row['worker_cpu_complete'] and not row['cuda_initialized']
            if row['mode'] == 'eval':
                assert row['callback_seconds']['critic'] == row['fallback_action_count'] == 0
        for model in models.values():
            assert sha(model['source']) == model['source_sha256']
            assert sha(model['checkpoint']) == model['checkpoint_sha256']
            assert sha(model['frozen_actor']) == sha(model['cached_actor']) == model['actor_sha256']
        state.update(status='complete', valid_tasks=110, weights_unchanged=True,
            summed_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows))
    except BaseException as exc:
        state.update(status='failed_or_budget_exhausted', error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['engine_wall_seconds'] = time.time() - started
        write(OUT / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
