"""One full66 test shard, CPU-only, no training or selection."""
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from common import (BUDGET, ORIGINAL, PACKAGE, PLAN, ROOT, VARIANTS, apply_resource_policy,
    execution_batches, output, read, resources, selected, sha, specs, task_count, task_resources, write)


def evaluate_batches(rt, tasks, out, cores, state):
    rows = []
    try:
        for batch in execution_batches(tasks):
            os.sched_setaffinity(0, cores[:batch['workers']])
            state.update(status='running', phase=batch['phase'], active_workers=batch['workers'], completed_tasks=len(rows))
            write(out / 'status.json', state)
            # run_jobs joins/cleans every child before returning; phases cannot overlap.
            rows.extend(rt.run_jobs(batch['jobs'], out / 'raw', max_workers=batch['workers'],
                watchdog_seconds=batch['watchdog_seconds']))
            write(out / 'evaluation_rows.json', rows)
    finally:
        os.sched_setaffinity(0, cores)
    return rows


def main():
    import numpy as np
    import torch
    import server
    import cpu_refinalize_cached as support
    from project import runtime as rt
    shard = int(os.environ['TGPPO_TEST66_SHARD'])
    policy = resources(shard)
    workers, requested = policy['workers'], task_count(shard)
    out = output(shard)
    started = read(out / 'bootstrap.json')['started_epoch']
    state = dict(status='preparing', shard=shard, pid=os.getpid(), host=os.uname().nodename,
        started_epoch=started, budget_seconds=BUDGET, requested_tasks=requested, training_steps=0,
        gpu=False, data_split='complete66 original test instances; historical exposure retained')
    write(out / 'status.json', state)
    def expired(*args):
        raise TimeoutError('240-minute per-shard execution cap; cleanup/report reserve')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, BUDGET - 45 - (time.time() - started)))
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task')) == 1
        configs = {v: server.plan(ROOT, ORIGINAL / 'configs' / (v + '.json')) for v in VARIANTS}
        # Resource-only preflight override; frozen solver/model configs stay unchanged.
        info = server.preflight(ROOT, dict(configs['b0'], workers=workers), 'cpu')
        prior = read(ROOT / 'tgppo/runs/light-actor-20261105/cross-instance-8/manifest.json')
        assert {k: v for k, v in info['stack'].items() if k != 'cpu_model'} == {
            k: v for k, v in prior['stack'].items() if k != 'cpu_model'}
        assert info['effective_cpu_count'] >= workers
        assert info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= policy['ram_gib'] * 1024**3
        cores = info['allowed_cpus'][:workers]
        os.sched_setaffinity(0, cores)
        cache = Path(tempfile.mkdtemp(prefix='tgppo-test66-', dir='/tmp'))
        assert cache.stat().st_dev != ROOT.stat().st_dev
        (cache / 'actors').mkdir()
        (cache / 'instances').mkdir()
        (out / 'frozen').mkdir()
        models = {}
        for item in PLAN['frozen_models']:
            label, seed, variant = item['label'], item['training_seed'], item['variant']
            source = Path(item['actor'])
            assert sha(source) == item['actor_sha256'] == prior['models'][label]['actor_sha256']
            payload = rt.read_checkpoint(source, allow_actor_only=True)
            assert payload['schema_version'] == 'actor-only-v1'
            assert payload['config']['server_training_seed'] == seed
            assert payload['config']['actor_kind'] == ('b0' if variant.startswith('b0') else 'mlp')
            assert payload['config']['expert_pretraining'] == ('bc' if variant.endswith('_bc') else 'none')
            assert payload['provenance']['progress']['next_episode'] == 100
            assert payload['provenance']['checkpoint_sha256'] == prior['models'][label]['checkpoint_sha256']
            frozen = out / 'frozen' / (label + '.pt')
            shutil.copyfile(source, frozen)
            frozen.chmod(0o444)
            local = cache / 'actors' / frozen.name
            shutil.copyfile(frozen, local)
            local.chmod(0o444)
            assert sha(local) == sha(frozen) == item['actor_sha256']
            models[label] = dict(item, source=str(source), frozen_actor=str(frozen), cached_actor=str(local))
        for item in selected(shard):
            assert item['instance'] not in configs['b0']['train'] + configs['b0']['development']
            assert sha(item['path']) == item['sha256']
            target = cache / 'instances' / (item['instance'] + '.mps.gz')
            shutil.copyfile(item['path'], target)
            assert sha(target) == item['sha256']
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
                raise RuntimeError('CPU probability/action verification timeout; no optimization started')
            observed = receive.recv()
            child.join(timeout=5)
            assert observed.get('values') == expected and observed['threads'] == 1 and not observed['cuda_initialized']
        finally:
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
            receive.close()
        write(out / 'cpu_fork_verification.json', dict(passed=True, parent_threads=1, child=observed,
            actual_all8_probabilities_and_actions_bitwise_equal=True, solver_probe_not_optimized=True))
        rt.mp.get_context = lambda method=None: original_context('fork' if method == 'spawn' else method)
        cutoffs = read(ROOT / 'tgppo/datasets/cutoffs.json')
        tasks = []
        for item in specs(shard):
            model = models.get(item['label'])
            variant, seed = (model['variant'], model['training_seed']) if model else ('b0', 11)
            job = server.bound_job(ROOT, configs[variant], cutoffs, item['instance'], item['solver_seed'], seed,
                'eval' if model else 'native', model['cached_actor'] if model else None,
                job_id=item['job_id'], instances=cache / 'instances')
            job['sampling_seed'] = rt.action_seed(seed, item['canonical_sampling_job_id'])
            apply_resource_policy(job, shard)
            tasks.append(dict(item, job=job))
        manifest = dict(shard=shard, models=models, tasks=tasks, selected=selected(shard), resources=info,
            stack=info['stack'], host=info['host'], cpu_affinity=cores, workers=workers, threads_per_worker=1,
            resource_policy=policy,
            execution_schedule=[dict(phase=b['phase'],workers=b['workers'],watchdog_seconds=b['watchdog_seconds'],
                address_space_limit_bytes=b['address_space_limit_bytes'],job_ids=[j['job_id'] for j in b['jobs']])
                for b in execution_batches(tasks)],
            native_reference='fresh{} in same interleaved{}-task shard; no training normalization'.format(len(selected(shard))*2, requested),
            worker_start_method='fork', runner_sha256=sha(__file__), protocol_sha256=sha(PACKAGE / 'docs/PROTOCOL.md'),
            selection_sha256=sha(PACKAGE / 'selection.json'), cache=str(cache))
        write(out / 'manifest.json', manifest)
        raw = Path(tempfile.mkdtemp(prefix='tgppo-test66-raw-', dir='/tmp'))
        (out / 'raw').symlink_to(raw, target_is_directory=True)
        state.update(status='running', setup_seconds=time.time() - started, stack=info['stack'])
        write(out / 'status.json', state)
        print(json.dumps(dict(shard=shard, tasks=requested, cpu=info['stack']['cpu_model'], cap_seconds=BUDGET)), flush=True)
        rows = evaluate_batches(rt, tasks, out, cores, state)
        write(out / 'evaluation_rows.json', rows)
        for row in rows:
            assert row['valid'] and row['worker_cpu_complete'] and not row['cuda_initialized']
            row_policy = task_resources(row['instance_name'])
            assert row['cpu_affinity'] == cores[:row_policy['workers']] and row['time_limit_seconds'] == 120
            assert row['address_space_limit_bytes'] == row_policy['address_space_limit_bytes']
            if row['mode'] == 'eval':
                assert row['callback_seconds']['critic'] == row['fallback_action_count'] == 0
        assert len(rows) == requested
        for m in models.values():
            assert sha(m['source']) == sha(m['frozen_actor']) == sha(m['cached_actor']) == m['actor_sha256']
        for item in selected(shard):
            assert sha(item['path']) == item['sha256']
        state.update(status='complete', valid_tasks=requested, completed_tasks=requested, active_workers=0, weights_unchanged=True,
            summed_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows))
    except BaseException as exc:
        state.update(status='failed_or_budget_exhausted', error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['engine_wall_seconds'] = time.time() - started
        write(out / 'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
