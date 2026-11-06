"""112 frozen-policy eval tasks, four single-thread CPU workers, 90-minute cap."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import time

LAUNCHED=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/'tgppo/runs/candidate-residual-20261104'
OUT=PARENT/'cross-instance-8'

class BudgetReached(RuntimeError):pass

def main():
    import experiment
    import server
    import torch
    from project import runtime as rt
    from project.utils.protocol import sha256,write_json
    cohort=server.read(OUT/'cohort.json')
    status_path=OUT/'status.json'
    if status_path.exists():raise RuntimeError('Attempt already exists; preserve results and budget, no automatic rerun.')
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),budget_seconds=5400,
        requested_tasks=112,workers=4,completed_tasks=0,training_steps=0)
    write_json(status_path,state)
    def expire(signum,frame):raise BudgetReached('90-minute wall cap;20-second cleanup reserve')
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,5380-(time.monotonic()-LAUNCHED)))
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        limits=server.resource_limits();assert limits['effective_cpu_count']>=4
        cores=limits['allowed_cpus'][:4];os.sched_setaffinity(0,cores);torch.set_num_threads(1)
        experiment.verify_package()
        original=server.read(PARENT/'evaluation/episode-0100/frozen_models.json')
        signature=server.stack(ROOT)
        assert {k:v for k,v in signature.items() if k!='cpu_model'}=={
            k:v for k,v in original['stack'].items() if k!='cpu_model'}
        configs={v:experiment.protocol(v)[0] for v in experiment.VARIANTS}
        models={}
        for m in original['models']:
            assert sha256(m['actor'])==m['actor_sha256'] and sha256(m['checkpoint'])==m['checkpoint_sha256']
            label='{}-s{}'.format(m['variant'],m['seed'])
            copied=OUT/'frozen'/(label+'.pt');copied.parent.mkdir(exist_ok=True)
            shutil.copyfile(m['actor'],copied)
            assert sha256(copied)==m['actor_sha256']
            models[label]=dict(m,frozen_actor=str(copied))
        cutoffs=server.read(ROOT/'tgppo/datasets/cutoffs.json')
        labels=['native','b0-s11','score-s11','candidate-s11','b0-s22','score-s22','candidate-s22']
        tasks=[]
        for index,item in enumerate(cohort['selected']):
            name=item['instance'];assert sha256(item['path'])==item['sha256']
            for ss in cohort['solver_seeds']:
                rotation=(2*index+ss)%len(labels);order=labels[rotation:]+labels[:rotation]
                for label in order:
                    m=models.get(label);variant=m['variant'] if m else 'native';seed=m['seed'] if m else 11
                    jid='t{:03d}-{}-{}-s{}'.format(len(tasks),label,name,ss)
                    job=server.bound_job(ROOT,configs[variant if m else 'b0'],cutoffs,name,ss,seed,
                        'eval' if m else 'native',m['frozen_actor'] if m else None,job_id=jid,
                        instances=ROOT/'tgppo/datasets/test_instances')
                    job['sampling_seed']=rt.action_seed(seed,'test-{}-s{}'.format(name,ss))
                    tasks.append(dict(index=len(tasks),label=label,variant=variant,
                        training_seed=seed if m else None,stratum=item['stratum'],job=job))
        assert len(tasks)==112
        manifest=dict(models=models,tasks=tasks,stack=signature,host=socket.gethostname(),cpu_affinity=cores,
            resources=limits,workers=4,threads_per_worker=1,cohort_sha256=sha256(OUT/'cohort.json'),
            runner_sha256=sha256(__file__),parent_models_sha256=sha256(PARENT/'evaluation/episode-0100/frozen_models.json'))
        write_json(OUT/'manifest.json',manifest)
        print(json.dumps(dict(started=True,instances=[r['instance'] for r in cohort['selected']],
            tasks=len(tasks),workers=4,cpu=signature['cpu_model'],budget_seconds=5400)),flush=True)
        rows=rt.run_jobs([r['job'] for r in tasks],OUT/'raw',max_workers=4,watchdog_seconds=180)
        write_json(OUT/'evaluation_rows.json',rows)
        valid=[r for r in rows if r.get('valid')]
        for r in valid:
            assert not r['cuda_initialized'] and r['cpu_affinity']==cores and r['time_limit_seconds']==120
            if r['mode']=='eval':assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
        for m in models.values():
            assert sha256(m['actor'])==sha256(m['frozen_actor'])==m['actor_sha256']
            assert sha256(m['checkpoint'])==m['checkpoint_sha256']
        assert not torch.cuda.is_initialized()
        state.update(status='complete' if len(valid)==112 else 'incomplete',completed_tasks=len(valid),
            invalid_tasks=len(rows)-len(valid),models_unchanged=True,cuda_initialized=False,
            summed_worker_wall_seconds=sum(r.get('worker_wall_seconds',r.get('wall_seconds',0)) for r in rows))
    except BudgetReached as exc:state.update(status='budget_exhausted',error=str(exc))
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['charged_wall_seconds']=time.monotonic()-LAUNCHED
        state['raw_result_files']=len(list((OUT/'raw').glob('t*.json')))
        write_json(status_path,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
