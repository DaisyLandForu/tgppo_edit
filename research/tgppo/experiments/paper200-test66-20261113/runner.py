"""Four resumable CPU Custom shards, fixed-200 weights, full66/solver0."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import uuid

from common import (BASE, PACKAGE, ROOT, RUN, fingerprint, plan, read, scan,
                    selected, sha, specs, task_resources, verify, write, TRAINING_SEEDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'check', 'summarize', 'status'))
    parser.add_argument('--shard', type=int, choices=range(4))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    verify()
    if args.command in ('summarize', 'status'):
        from analyze import summarize
        summarize()
        return
    if args.command == 'check':
        import probe
        paths = [str(ROOT / m['frozen_actor']) for m in plan()['models']]
        result = probe.check(paths)
        write(PACKAGE / 'ACTOR_CPU_CHECK.json', result)
        print(result, flush=True)
        return
    if args.shard is None:
        parser.error('run requires --shard 0..3')
    out = RUN / ('shard-' + str(args.shard))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.lock').open('a') as execution_lock:
        # A second submission cannot corrupt the first one's state or results.
        fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = list((out / 'attempts').glob('*'))
        if prior and not args.resume:
            raise ValueError('Prior attempt retained; use --resume, never delete results')
        evaluate(args.shard, out)


def evaluate(shard, out):
    start = time.time()
    state = dict(status='preparing', shard=shard, started_epoch=start, pid=os.getpid(),
                 training_steps=0, gpu=False, total_wall_limit_seconds=None)
    cache = None
    cores = None
    attempt = None
    previous_signals = {}
    def stop(signum, frame):
        raise RuntimeError('Signal {}; completed shared-storage results retained'.format(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_signals[sig] = signal.signal(sig, stop)
    write(out / 'status.json', state)
    try:
        import torch
        import server
        import probe
        from project import runtime as rt
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        info = server.preflight(ROOT, dict(read(BASE / 'configs/protocol.json'), workers=4), 'cpu')
        assert info['memory_limit_bytes'] is None or info['memory_limit_bytes'] >= 32*1024**3
        cores = info['allowed_cpus'][:4]
        os.sched_setaffinity(0, cores)
        environment = dict(stack=info['stack'], host=info['host'], cores=cores, worker_start_method='fork',
                           plan_sha256=sha(PACKAGE / 'plan.json'))
        cid = fingerprint(environment)[:24]
        panels, cohorts = scan(shard)
        pending = [s for s in specs(shard) if s['instance'] not in panels]
        cached_rows = cohorts.get(cid, {}).get('rows', {})
        pending = [s for s in pending if s['job_id'] not in cached_rows]
        if not pending:
            state.update(status='complete', complete_instances=len(panels), valid_tasks=5*len(panels))
            return
        stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        attempt = out / 'attempts' / stamp
        attempt.mkdir(parents=True)
        write(attempt / 'environment.json', info)
        cache = Path(tempfile.mkdtemp(prefix='tgppo-paper200-', dir='/tmp'))
        (cache / 'actors').mkdir()
        (cache / 'instances').mkdir()
        models = {}
        for m in plan()['models']:
            path = cache / 'actors' / (m['label'] + '.pt')
            shutil.copyfile(ROOT / m['frozen_actor'], path)
            assert sha(path) == m['actor_sha256']
            path.chmod(0o444)
            models[m['label']] = path
        instance_map = {i['name']: i for i in selected(shard)}
        for name in {s['instance'] for s in pending}:
            i = instance_map[name]
            path = cache / 'instances' / (name + '.mps.gz')
            shutil.copyfile(ROOT / i['path'], path)
            assert sha(path) == i['sha256']
        write(attempt / 'cpu_fork_check.json', probe.check(list(models.values())))
        original_context = rt.mp.get_context
        rt.mp.get_context = lambda method=None: original_context('fork' if method == 'spawn' else method)
        cutoffs = read(ROOT / 'tgppo/datasets/cutoffs.json')
        tasks = []
        for spec in pending:
            label, name = spec['label'], spec['instance']
            job = rt.make_job(spec['job_id'], cache / 'instances', name, 3600, 0,
                TRAINING_SEEDS[label], cutoffs, 'native' if label == 'native' else 'eval',
                checkpoint=models.get(label))
            job['sampling_seed'] = rt.action_seed(TRAINING_SEEDS[label], spec['canonical_sampling_job_id'])
            job['address_space_limit_bytes'] = task_resources(name)['address_space_limit_bytes']
            tasks.append(dict(job_id=spec['job_id'], spec=spec, job=job))
        write(attempt / 'manifest.json', dict(plan_sha256=sha(PACKAGE / 'plan.json'), cohort_id=cid,
              environment=environment, tasks=tasks, setup_seconds=time.time()-start))
        state.update(status='running', attempt=str(attempt), cohort_id=cid, pending=len(tasks),
                     already_complete_instances=len(panels), environment=environment)
        write(out / 'status.json', state)
        for phase in ('ordinary', 'large_serial'):
            batch = [t['job'] for t in tasks if task_resources(t['spec']['instance'])['phase'] == phase]
            if not batch:
                continue
            policy = task_resources(batch[0]['instance_name'])
            os.sched_setaffinity(0, cores[:policy['workers']])
            state.update(phase=phase, active_workers=policy['workers'])
            write(out / 'status.json', state)
            print('shard={} phase={} tasks={} workers={} horizon=3600 watchdog=4200'.format(
                shard, phase, len(batch), policy['workers']), flush=True)
            # Raw worker JSON is immediately atomic on shared storage, not /tmp.
            rows = rt.run_jobs(batch, attempt / 'raw', policy['workers'], policy['watchdog_seconds'])
            write(attempt / (phase + '-rows.json'), rows)
        panels, _ = scan(shard)
        complete = len(panels) == len(selected(shard))
        state.update(status='complete' if complete else 'incomplete_requires_explicit_resume',
                     complete_instances=len(panels), valid_tasks=len(panels)*5, active_workers=0)
        if not complete:
            raise RuntimeError('Invalid/missing rows retained; inspect progress.json before explicit --resume')
        verify()
    except BaseException as exc:
        state.update(status='stopped_or_failed', error=repr(exc), active_workers=0)
        raise
    finally:
        if cores:
            os.sched_setaffinity(0, cores)
        if cache:
            shutil.rmtree(cache, ignore_errors=True)
        state.update(elapsed_seconds=time.time()-start, finished_epoch=time.time())
        write(out / 'status.json', state)
        if attempt:
            write(attempt / 'execution.json', state)
        from analyze import summarize
        try:
            summarize()
        except Exception as exc:
            print('Summarize failed; raw records retained: ' + repr(exc), flush=True)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
