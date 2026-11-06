"""Conditional frozen-policy 2x2: 144 CPU tasks, at most 90 minutes, no retries."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import tempfile
import tarfile
import time
from datetime import datetime

LAUNCHED=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/light-actor-20261105'
OUT=RUN/'cross-instance-8'


class BudgetReached(RuntimeError):pass


def main():
    # Read-only gate first, before importing the solver runtime.
    from decide import read,sha,panel,csv_write
    decision=read(HERE/'decision.json')
    assert decision['status']=='passed' and decision['cross_instance_allowed']
    assert decision['rule_sha256']==sha(HERE/'PLAN.md')
    assert decision['decision_script_sha256']==sha(HERE/'decide.py')
    assert all(sha(ROOT/p)==digest for p,digest in decision['source_sha256'].items())
    status_path=OUT/'status.json'
    if status_path.exists():raise RuntimeError('Existing attempt: preserve budget/results, no automatic retry')
    OUT.mkdir(parents=True,exist_ok=True)
    bootstrap=read(HERE/'cross_bootstrap.json') if (HERE/'cross_bootstrap.json').exists() else {}
    started=bootstrap.get('started_epoch',time.time())
    state=dict(status='preparing',pid=os.getpid(),host=socket.gethostname(),started_epoch=started,
        budget_seconds=5400,requested_tasks=144,workers=4,training_steps=0,cuda_initialized=False)
    def write(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    write(status_path,state)
    def expire(signum,frame):raise BudgetReached('Cumulative CPU execution cap reached; cleanup reserve20s')
    signal.signal(signal.SIGALRM,expire)
    deadline_epoch=datetime.fromisoformat('2026-11-05T22:00:00+08:00').timestamp()
    remaining=min(5400-(time.time()-started),deadline_epoch-time.time())
    if remaining<=20:raise RuntimeError('No execution/cleanup time remains before the deadline')
    signal.setitimer(signal.ITIMER_REAL,remaining-20)
    models={};tasks=[];rows=[];raw_cache=None
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch
        import numpy as np
        import launch
        import server
        from project import runtime as rt
        launch.verify();torch.set_num_threads(1)
        kind=decision['chosen']['kind'];variants=('b0','b0_bc',kind,kind+'_bc')
        configs={v:launch.protocol(v)[0] for v in variants}
        info=server.preflight(ROOT,configs['b0'],'cpu');cores=info['allowed_cpus'][:4]
        assert info['effective_cpu_count']>=4
        os.sched_setaffinity(0,cores)
        cache=Path(tempfile.mkdtemp(prefix='tgppo-cross-',dir='/tmp'));(cache/'instances').mkdir();(cache/'actors').mkdir()
        assert cache.stat().st_dev!=ROOT.stat().st_dev
        done=read(Path(decision['results_directory'])/'complete.json');cohort_path=Path(done['cohort'])
        assert {k:v for k,v in info['stack'].items() if k!='cpu_model'}=={
            k:v for k,v in done['stack'].items() if k!='cpu_model'}
        full=read(cohort_path/'models-seeds-11-22.json')
        frozen=OUT/'frozen';frozen.mkdir(exist_ok=True)
        for m in full:
            if m['variant'] not in variants:continue
            actor=RUN/m['variant']/'training'/('B0-s'+str(m['seed']))/'actor-0100.pt'
            assert sha(actor)==m['actor_sha256'] and sha(m['checkpoint'])==m['checkpoint_sha256']
            label='{}-s{}'.format(m['variant'],m['seed']);copy=frozen/(label+'.pt')
            if copy.exists():raise RuntimeError('Uncommitted frozen copy exists; inspect instead of overwrite')
            shutil.copyfile(actor,copy);copy.chmod(0o444)
            assert sha(copy)==m['actor_sha256']
            cached=cache/'actors'/copy.name;shutil.copyfile(copy,cached);cached.chmod(0o444)
            assert sha(cached)==m['actor_sha256']
            models[label]=dict(variant=m['variant'],seed=m['seed'],actor=str(actor),frozen_actor=str(copy),
                cached_actor=str(cached),actor_sha256=m['actor_sha256'],checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'])
        assert len(models)==8
        old=ROOT/'tgppo/runs/candidate-residual-20261104/cross-instance-8/cohort.json'
        selected=read(old)['selected']
        assert [x['instance'] for x in selected]==['bell5','bell3a','neos-1215259','neos-612125','sp98ir','neos-512201','n3seq24','bc1']
        assert not {x['instance'] for x in selected}&set(configs['b0']['train']+configs['b0']['development'])
        cutoffs=read(ROOT/'tgppo/datasets/cutoffs.json')
        # Reuse the validated CPU-only process approach; never fork a CUDA/threaded learner.
        import cpu_refinalize_cached as support
        support.torch=torch;support.np=np;support.rt=rt
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task'))==1
        paths=[models[label]['cached_actor'] for label in ('b0-s11','b0_bc-s22',kind+'-s11',kind+'_bc-s22')]
        expected=support.signatures(paths);assert len(os.listdir('/proc/self/task'))==1
        original_context=rt.mp.get_context;context=original_context('fork');receive,send=context.Pipe(duplex=False)
        child=context.Process(target=support.probe_child,args=(send,paths));child.start();send.close()
        try:
            if not receive.poll(60):raise RuntimeError('Cross CPU fork verification timed out; no solves launched')
            observed=receive.recv();child.join(timeout=5)
            assert observed.get('values')==expected and observed['threads']==1 and not observed['cuda_initialized']
        finally:
            if child.is_alive():child.kill();child.join(timeout=1)
            receive.close()
        write(OUT/'cpu_fork_verification.json',dict(passed=True,child=observed))
        rt.mp.get_context=lambda method=None: original_context('fork' if method=='spawn' else method)
        labels=['native']+['{}-s{}'.format(v,s) for s in (11,22) for v in variants]
        for index,item in enumerate(selected):
            name=item['instance'];assert sha(item['path'])==item['sha256']
            local_instance=cache/'instances'/(name+'.mps.gz');shutil.copyfile(item['path'],local_instance)
            assert sha(local_instance)==item['sha256']
            for ss in (0,1):
                rotation=(2*index+ss)%len(labels);order=labels[rotation:]+labels[:rotation]
                for label in order:
                    m=models.get(label);variant=m['variant'] if m else 'native';seed=m['seed'] if m else 11
                    jid='t{:03d}-{}-{}-s{}'.format(len(tasks),label,name,ss)
                    job=server.bound_job(ROOT,configs[variant if m else 'b0'],cutoffs,name,ss,seed,
                        'eval' if m else 'native',m['cached_actor'] if m else None,job_id=jid,
                        instances=cache/'instances')
                    job['sampling_seed']=rt.action_seed(seed,'test-{}-s{}'.format(name,ss))
                    tasks.append(dict(index=len(tasks),variant=variant,training_seed=seed if m else None,
                        stratum=item['stratum'],job=job))
        assert len(tasks)==144
        manifest=dict(variants=list(variants),models=models,tasks=tasks,selected=selected,stack=info['stack'],
            host=socket.gethostname(),cpu_affinity=cores,resources=info,workers=4,threads_per_worker=1,
            source_cohort_sha256=sha(old),decision_sha256=sha(HERE/'decision.json'),rule_sha256=sha(HERE/'PLAN.md'),
            runner_sha256=sha(__file__),native_reference='16 fresh native tasks, same host and interleaved schedule',
            worker_start_method='fork',cache=str(cache),cpu_fork_verification=sha(OUT/'cpu_fork_verification.json'),
            scope='previously explored cross-instance cohort; not a new untouched test set')
        write(OUT/'manifest.json',manifest)
        raw_cache=Path(tempfile.mkdtemp(prefix='tgppo-cross-raw-',dir='/tmp'))
        raw_link=OUT/'raw'
        if raw_link.exists() or raw_link.is_symlink():
            raise RuntimeError('Existing raw outputs: inspect rather than overwrite')
        raw_link.symlink_to(raw_cache,target_is_directory=True)
        write(OUT/'LOCAL_RAW_OUTPUT.json',dict(local_directory=str(raw_cache),shared_link=str(raw_link)))
        state.update(status='running',stack=info['stack'],variants=list(variants));write(status_path,state)
        print(json.dumps(dict(tasks=144,workers=4,cpu=info['stack']['cpu_model'],budget_seconds=5400),ensure_ascii=False),flush=True)
        rows=rt.run_jobs([t['job'] for t in tasks],OUT/'raw',max_workers=4,watchdog_seconds=180)
        write(OUT/'evaluation_rows.json',rows)
        valid=[r for r in rows if r.get('valid')]
        for r in valid:
            assert not r['cuda_initialized'] and r['cpu_affinity']==cores and r['time_limit_seconds']==120
            if r['mode']=='eval':assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
        for m in models.values():
            assert sha(m['actor'])==sha(m['frozen_actor'])==sha(m['cached_actor'])==m['actor_sha256']
            assert sha(m['checkpoint'])==m['checkpoint_sha256']
        assert not torch.cuda.is_initialized()
        state.update(status='complete' if len(valid)==144 else 'incomplete',valid_tasks=len(valid),
            invalid_tasks=len(rows)-len(valid),weights_unchanged=True,
            summed_worker_wall_seconds=sum(r.get('worker_wall_seconds',0) for r in rows))
    except BudgetReached as exc:state.update(status='budget_exhausted',error=str(exc))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if raw_cache is not None:
            files=sorted(raw_cache.glob('t*.json'))
            temporary=OUT/'raw_results.tmp'
            with tarfile.open(temporary,'w') as archive:
                for path in files:archive.add(path,arcname=path.name,recursive=False)
            temporary.replace(OUT/'raw_results.tar')
        state['charged_wall_seconds']=time.time()-started
        state['raw_result_files']=len(list((OUT/'raw').glob('t*.json')))
        write(status_path,state);print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
