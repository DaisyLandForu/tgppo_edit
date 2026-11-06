"""Frozen-Actor retest: 28 sequential CPU tasks, aggregate wall cap 600 s."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time

LAUNCHED=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/'tgppo/runs/candidate-residual-20261104'
OUT=PARENT/'cpu-retest-seeds23'


class BudgetReached(RuntimeError):
    pass


def main():
    import experiment
    import server
    import torch
    from project import runtime as rt
    from project.utils.protocol import sha256,write_json

    OUT.mkdir(parents=True,exist_ok=True)
    state_path=OUT/'status.json'
    records=[];prior=0.
    if state_path.exists():
        state=server.read(state_path)
        if '--resume' not in sys.argv or state['status']!='failed':
            raise RuntimeError('Retest already attempted; explicit failed-attempt resume only, preserving budget.')
        prior=state['charged_wall_seconds']
        attempts=OUT/'attempts';attempts.mkdir(exist_ok=True)
        write_json(attempts/('attempt-{:03d}.json'.format(len(list(attempts.glob('attempt-*.json'))))),state)
        records=server.read(OUT/'completed.json') if (OUT/'completed.json').exists() else []
        state.update(status='running',resumed_from_error=state.pop('error',None))
    else:
        state=dict(status='running',budget_seconds=600,started_unix=time.time(),
                   requested_tasks=28,completed_tasks=0,failed_tasks=0,training_steps=0)
    write_json(state_path,state)
    def expire(signum,frame):
        raise BudgetReached('Aggregate 600-second wall budget reached; cleanup reserve 3 seconds')
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,597-prior-(time.monotonic()-LAUNCHED)))
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        core=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{core})
        torch.set_num_threads(1)
        experiment.verify_package()
        frozen=server.read(PARENT/'evaluation/episode-0100/frozen_models.json')
        configurations={v:experiment.protocol(v)[0] for v in experiment.VARIANTS}
        signature=server.stack(ROOT)
        assert {k:v for k,v in signature.items() if k!='cpu_model'}=={
            k:v for k,v in frozen['stack'].items() if k!='cpu_model'}
        models={}
        for m in frozen['models']:
            assert sha256(m['actor'])==m['actor_sha256']
            assert sha256(m['checkpoint'])==m['checkpoint_sha256']
            label='{}-s{}'.format(m['variant'],m['seed'])
            copy=OUT/'frozen'/(label+'.pt');copy.parent.mkdir(exist_ok=True)
            if not copy.exists():shutil.copyfile(m['actor'],copy)
            assert sha256(copy)==m['actor_sha256']
            models[label]=dict(m,frozen_actor=str(copy))
        cutoffs=server.read(ROOT/'tgppo/datasets/cutoffs.json')
        labels=['native','b0-s11','score-s11','candidate-s11','b0-s22','score-s22','candidate-s22']
        tasks=[]
        # Fixed rotation spreads variants across execution order without selecting winners.
        for block,(name,ss) in enumerate([('dcmulti',2),('vpm2',2),('vpm2',3),('dcmulti',3)]):
            assert sha256(ROOT/'tgppo/datasets/train_instances'/(name+'.mps.gz'))==configurations['b0']['instance_hashes'][name]
            order=labels[2*block:]+labels[:2*block]
            for label in order:
                model=models.get(label);variant=model['variant'] if model else 'native'
                seed=model['seed'] if model else 11
                job=server.bound_job(ROOT,configurations[variant if model else 'b0'],cutoffs,
                    name,ss,seed,'eval' if model else 'native',model['frozen_actor'] if model else None,
                    job_id='dev-{}-s{}'.format(name,ss))
                tasks.append(dict(index=len(tasks),label=label,variant=variant,training_seed=seed if model else None,job=job))
        manifest=dict(models=models,tasks=tasks,time_limit=120,solver_seeds=[2,3],instances=['dcmulti','vpm2'],
            workers=1,cpu_affinity=[core],wall_budget_seconds=600,stack=signature,host=socket.gethostname(),
            original_evaluation_stack=frozen['stack'],parent_frozen_manifest_sha256=sha256(PARENT/'evaluation/episode-0100/frozen_models.json'),
            runner_sha256=sha256(__file__),scope='paired positive/negative development-instance retest; no new training or model selection')
        if (OUT/'manifest.json').exists():
            original=server.read(OUT/'manifest.json')
            assert {k:v for k,v in original.items() if k!='runner_sha256'}=={
                k:v for k,v in manifest.items() if k!='runner_sha256'}
            write_json(OUT/'runner_amendment.json',dict(original_runner_sha256=original['runner_sha256'],
                corrected_runner_sha256=manifest['runner_sha256'],reason='Native rows do not have Actor callback fields; preserve first result and charged time.'))
        else:write_json(OUT/'manifest.json',manifest)
        for task in tasks:
            if task['index'] in {r['index'] for r in records}:continue
            remaining=597-prior-(time.monotonic()-LAUNCHED)
            if remaining<=2:raise BudgetReached('No remaining task budget')
            directory=OUT/'raw'/('t{:02d}-{}'.format(task['index'],task['label']))
            started=time.monotonic()
            row=rt.run_jobs([task['job']],directory,max_workers=1,
                watchdog_seconds=min(180,remaining))[0]
            record=dict(index=task['index'],label=task['label'],variant=task['variant'],
                training_seed=task['training_seed'],result=str(directory/(task['job']['job_id']+'.json')),
                elapsed_seconds=time.monotonic()-started)
            records.append(record)
            write_json(OUT/'completed.json',records)
            state['completed_tasks']=sum(server.read(r['result']).get('valid',False) for r in records)
            state['charged_wall_seconds']=prior+time.monotonic()-LAUNCHED
            write_json(state_path,state)
            if not row.get('valid'):
                state['failed_tasks']+=1
                raise RuntimeError('Invalid task preserved: '+record['result'])
            assert not row['cuda_initialized'] and row['cpu_affinity']==[core]
            if task['variant']!='native':
                assert row['callback_seconds']['critic']==0 and row['fallback_action_count']==0
            assert row['sampling_seed']==task['job']['sampling_seed'] and row['time_limit_seconds']==120
            print('{}/28 {} {} solver{} PDI={:.5f} nodes={} solve={:.3f}s elapsed={:.1f}s'.format(
                len(records),task['label'],row['instance_name'],row['solver_seed'],row['pdi_horizon_midpoint'],
                row['nnodes'],row['scip_solve_time'],prior+time.monotonic()-LAUNCHED),flush=True)
        for model in models.values():
            assert sha256(model['actor'])==sha256(model['frozen_actor'])==model['actor_sha256']
            assert sha256(model['checkpoint'])==model['checkpoint_sha256']
        assert not torch.cuda.is_initialized()
        state.update(status='complete',models_unchanged=True,cuda_initialized=False)
    except BudgetReached as exc:
        state.update(status='budget_exhausted',error=str(exc))
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['charged_wall_seconds']=prior+time.monotonic()-LAUNCHED
        state['uncompleted_tasks']=28-len(records)
        write_json(state_path,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
