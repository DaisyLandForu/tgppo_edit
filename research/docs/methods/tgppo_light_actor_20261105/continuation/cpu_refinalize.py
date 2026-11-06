"""Authorized CPU remeasurement; frozen evaluator, fresh10 native, total90min.

All changes are process-local evaluation plumbing. Frozen sources/weights and
the old Gold6148 attempt remain untouched. At most one retry of invalid tasks
per model/native batch, retaining valid rows and all failure/cost records.
"""
import fcntl
import json
import os
from pathlib import Path
import socket
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ORIGINAL=ROOT/'tgppo/runs/light-actor-20261105'
OUT=ORIGINAL/'cpu-recovery-20261105'


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    auth=read(OUT/'authorization.json')
    assert auth['authorized'] and auth['additional_wall_seconds']==5400 and auth['tasks']==170
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(OUT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'results/complete.json').exists():return
    budget_path=OUT/'evaluation/budget.json';budget=read(budget_path)
    setup=budget['attempts'][-1]
    assert setup['stage']=='initial_setup' and 'elapsed_seconds' not in setup
    started=setup['started_epoch']
    state=dict(status='preparing',pid=os.getpid(),host=socket.gethostname(),started_epoch=started,
        additional_budget_seconds=5400,expected_model_tasks=160,expected_native_tasks=10,
        training_steps=0,driver=str(Path(__file__).resolve()),old_evaluation_preserved=True)
    write(OUT/'status.json',state)
    try:
        import torch
        import launch
        import server
        from project.utils.protocol import fingerprint,write_json
        from project import runtime as rt
        torch.set_num_threads(1);launch.verify()
        launch.MATRIX=OUT
        launch.job_path=lambda variant,seed: ORIGINAL/'jobs'/('{}-s{}.json'.format(variant,seed))
        original_export=launch.ensure_export
        def read_only_export(checkpoint,actor,payload=None):
            if not Path(actor).exists():raise RuntimeError('Source Actor export missing; do not write the frozen run')
            return original_export(checkpoint,actor,payload)
        launch.ensure_export=read_only_export
        original_cached=launch._CACHED
        def cached(jobs,folder,identity,workers,watchdog):
            try:return original_cached(jobs,folder,identity,workers,watchdog)
            except RuntimeError as exc:
                if 'Invalid jobs retained' not in str(exc) or len(list(Path(folder).glob('attempt-*')))>=2:
                    raise
                print('Retrying invalid/missing tasks once; valid records preserved: '+str(folder),flush=True)
                return original_cached(jobs,folder,identity,workers,watchdog)
        launch._CACHED=cached
        def fresh_development_native(c,cohort,info):
            # No training normalization is consumed during Actor-only evaluation.
            cutoffs=server.read(ROOT/'tgppo/datasets/cutoffs.json')
            jobs=[server.bound_job(ROOT,c,cutoffs,n,ss,11,'native',job_id='native-'+n+'-s'+str(ss))
                for n in c['development'] for ss in c['evaluation_solver_seeds']]
            pairs=cached(jobs,cohort/'fresh-native',dict(stack=info['stack'],protocol=fingerprint(c),
                kind='fresh-development-native-10'),c['workers'],c['watchdog'])
            by_id={r['job_id']:r for r,_ in pairs};values={}
            for job in jobs:
                identity=rt.job_identity(job)
                values[fingerprint(identity)]=dict(identity=identity,result=by_id[job['job_id']])
            write_json(cohort/'normalization/baselines.json',dict(protocol_sha256=fingerprint(c),
                stack=info['stack'],baselines=values,native_reference='fresh development10 on this host',
                note='evaluation-only; deliberately not a75-task training normalization bundle'))
        launch.normalize=fresh_development_native
        setup['elapsed_seconds']=time.time()-started
        budget['charged_seconds']=launch.charged_seconds(budget['attempts']);write(budget_path,budget)
        if budget['charged_seconds']>=5380:raise RuntimeError('Setup consumed the execution budget')
        state.update(status='evaluating',setup_seconds=setup['elapsed_seconds']);write(OUT/'status.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        with launch.lock(OUT/'finalize.lock',blocking=False):
            launch.finalize(100,(11,22))
        done=read(OUT/'results/complete.json')
        done.update(native_reference='fresh10, same host as all160 model evaluations',
            recovery_authorization=auth,old_gold_evaluation=str(ORIGINAL/'evaluation'),
            evaluation_driver_sha256=launch.sha256(__file__))
        write(OUT/'results/complete.json',done)
        state.update(status='complete',model_tasks=160,native_tasks=10,cuda_initialized=torch.cuda.is_initialized())
        assert not state['cuda_initialized']
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        latest=read(budget_path)
        if 'elapsed_seconds' not in latest['attempts'][-1] and latest['attempts'][-1].get('stage')=='initial_setup':
            latest['attempts'][-1]['elapsed_seconds']=time.time()-started
            latest['charged_seconds']=sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in latest['attempts'])
            write(budget_path,latest)
        state.update(charged_wall_seconds=time.time()-started,
            valid_raw_task_files=sum(1 for p in (OUT/'evaluation').rglob('*.json')
                if p.name.startswith(('dev-','native-')) and read(p).get('valid')))
        write(OUT/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
