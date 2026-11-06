"""Authorized20-task CPU ablation; existing weights, cap1200s, no retries."""
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'tgppo/runs/light-actor-20261105'
OUT=BASE/'bc-only-ablation-20'

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)

class BudgetReached(RuntimeError):pass

def main():
    import torch
    import numpy as np
    import launch
    import server
    from project import runtime as rt
    from project.utils.protocol import fingerprint
    import cpu_refinalize_cached as support
    started=read(OUT/'bootstrap.json')['started_epoch']
    if (OUT/'status.json').exists():raise RuntimeError('Existing attempt; preserve results and budget, no retry')
    state=dict(status='preparing',pid=os.getpid(),host=os.uname().nodename,started_epoch=started,
        budget_seconds=1200,requested_tasks=20,workers=4,training_steps=0,gpu=False)
    write(OUT/'status.json',state)
    def expire(*args):raise BudgetReached('1200-second cap;20s cleanup reserve')
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,1200-(time.time()-started)-20))
    raw=None;models={};tasks=[]
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        torch.set_num_threads(1);launch.verify()
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task'))==1
        config=launch.protocol('mlp_bc')[0];info=server.preflight(ROOT,config,'cpu')
        assert info['effective_cpu_count']>=4
        cores=info['allowed_cpus'][:4];os.sched_setaffinity(0,cores)
        cache=Path(tempfile.mkdtemp(prefix='tgppo-bc-ablation-',dir='/tmp'))
        assert cache.stat().st_dev!=ROOT.stat().st_dev
        (cache/'actors').mkdir();(cache/'instances').mkdir();(OUT/'frozen').mkdir()
        prior=read(BASE/'cross-instance-8/manifest.json')
        for seed in (11,22):
            source=BASE/'mlp_bc/training'/('B0-s'+str(seed))
            bc=read(source/'pretrain/result.json');initial=read(source/'initialization.json')
            original=source/'pretrain/actor.pt';final=source/'actor-0100.pt'
            assert launch.sha256(final)==prior['models']['mlp_bc-s'+str(seed)]['actor_sha256']
            assert bc['completed_steps']==1000 and bc['selected_step']==100
            assert launch.sha256(original)==bc['actor_sha256']==initial['bc_actor_sha256']
            ppo=rt.read_checkpoint(final,allow_actor_only=True)
            assert ppo['config']['server_training_seed']==seed and ppo['config']['actor_kind']=='mlp'
            payload=torch.load(original,map_location='cpu')
            assert payload['schema']=='light-bc-v1' and payload['signature']==bc['signature']
            for condition in ('bc_only','bc_ppo'):
                label=condition+'-s'+str(seed);frozen=OUT/'frozen'/(label+'.pt')
                if condition=='bc_only':
                    converted=dict(schema_version='actor-only-v1',config=ppo['config'],config_sha256=ppo['config_sha256'],
                        actor=payload['actor'],provenance=dict(bc_source_sha256=bc['actor_sha256'],bc_selected_step=100,ppo_episodes=0))
                    torch.save(converted,frozen)
                    policy,loaded=rt.load_actor(frozen)
                    assert set(loaded['actor'])==set(payload['actor'])
                    assert all(torch.equal(loaded['actor'][key],tensor) for key,tensor in payload['actor'].items())
                    assert server.digest_actor(policy.actor_network)==initial['actor_fingerprint']
                else:shutil.copyfile(final,frozen)
                frozen.chmod(0o444);local=cache/'actors'/frozen.name;shutil.copyfile(frozen,local);local.chmod(0o444)
                assert launch.sha256(local)==launch.sha256(frozen)
                models[label]=dict(condition=condition,seed=seed,source=str(original if condition=='bc_only' else final),
                    source_sha256=launch.sha256(original if condition=='bc_only' else final),
                    frozen_actor=str(frozen),actor_sha256=launch.sha256(frozen),cached_actor=str(local),
                    selected_bc_step=100,ppo_episodes=0 if condition=='bc_only' else 100,
                    initialization_sha256=launch.sha256(source/'initialization.json'))
        selected=[x for x in prior['selected'] if x['instance'] in ('neos-1215259','neos-512201')]
        assert [x['instance'] for x in selected]==['neos-1215259','neos-512201']
        for item in selected:
            target=cache/'instances'/(item['instance']+'.mps.gz');shutil.copyfile(item['path'],target)
            assert launch.sha256(target)==launch.sha256(item['path'])==item['sha256']
        support.torch=torch;support.np=np;support.rt=rt
        paths=[m['cached_actor'] for m in models.values()];expected=support.signatures(paths)
        assert len(os.listdir('/proc/self/task'))==1 and not torch.cuda.is_initialized()
        original_context=rt.mp.get_context;context=original_context('fork');receive,send=context.Pipe(duplex=False)
        child=context.Process(target=support.probe_child,args=(send,paths));child.start();send.close()
        try:
            if not receive.poll(60):raise RuntimeError('Fork check timeout; no solver tasks started')
            observed=receive.recv();child.join(timeout=5)
            assert observed.get('values')==expected and observed['threads']==1 and not observed['cuda_initialized']
        finally:
            if child.is_alive():child.kill();child.join(timeout=1)
            receive.close()
        write(OUT/'cpu_fork_verification.json',dict(passed=True,parent_threads=1,child=observed))
        rt.mp.get_context=lambda method=None:original_context('fork' if method=='spawn' else method)
        cutoffs=read(ROOT/'tgppo/datasets/cutoffs.json');labels=['native']+list(models)
        for index,item in enumerate(selected):
            name=item['instance']
            for ss in (0,1):
                rotation=(2*index+ss)%5;order=labels[rotation:]+labels[:rotation]
                for label in order:
                    model=models.get(label);seed=model['seed'] if model else 11
                    jid='t{:03d}-{}-{}-s{}'.format(len(tasks),label,name,ss)
                    job=server.bound_job(ROOT,config,cutoffs,name,ss,seed,'eval' if model else 'native',
                        model['cached_actor'] if model else None,job_id=jid,instances=cache/'instances')
                    job['sampling_seed']=rt.action_seed(seed,'test-{}-s{}'.format(name,ss))
                    tasks.append(dict(condition=model['condition'] if model else 'native',training_seed=seed if model else None,job=job))
        assert len(tasks)==20
        manifest=dict(models=models,tasks=tasks,selected=selected,stack=info['stack'],host=info['host'],cpu_affinity=cores,
            workers=4,threads_per_worker=1,worker_start_method='fork',scope='selected previously inspected positive/negative cases; not a new test set',
            runner_sha256=launch.sha256(__file__),plan_sha256=launch.sha256(HERE/'PLAN.md'))
        write(OUT/'manifest.json',manifest)
        raw=Path(tempfile.mkdtemp(prefix='tgppo-bc-ablation-raw-',dir='/tmp'));(OUT/'raw').symlink_to(raw,target_is_directory=True)
        write(OUT/'LOCAL_RAW_OUTPUT.json',dict(local_directory=str(raw),shared_link=str(OUT/'raw')))
        state.update(status='running',stack=info['stack'],setup_seconds=time.time()-started);write(OUT/'status.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        rows=rt.run_jobs([t['job'] for t in tasks],OUT/'raw',max_workers=4,watchdog_seconds=180)
        write(OUT/'evaluation_rows.json',rows)
        for row in rows:
            assert row['valid'] and row['worker_cpu_complete'] and not row['cuda_initialized']
            if row['mode']=='eval':assert row['callback_seconds']['critic']==0 and row['fallback_action_count']==0
        for model in models.values():
            assert launch.sha256(model['source'])==model['source_sha256']
            assert launch.sha256(model['frozen_actor'])==launch.sha256(model['cached_actor'])==model['actor_sha256']
        assert len(rows)==20 and not torch.cuda.is_initialized()
        state.update(status='complete',valid_tasks=20,weights_unchanged=True,
            summed_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows))
    except BudgetReached as exc:state.update(status='budget_exhausted',error=str(exc))
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if raw is not None:
            with tarfile.open(OUT/'raw_results.tar','w') as archive:
                for path in sorted(raw.glob('t*.json')):archive.add(path,arcname=path.name,recursive=False)
        state.update(charged_wall_seconds=time.time()-started,raw_result_files=len(list((OUT/'raw').glob('t*.json'))))
        write(OUT/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
