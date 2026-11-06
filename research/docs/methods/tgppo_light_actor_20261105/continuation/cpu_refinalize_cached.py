"""Resume the SAME90min CPU budget with cached inputs and verified CPU fork.

Frozen Actor/SCIP/PDI code and canonical action RNG remain unchanged. A global
interleaved four-worker queue avoids per-model idle tails and repeated imports.
"""
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import tempfile
import time

HERE=Path('/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_20261105/continuation')
ROOT=HERE.parents[3]
ORIGINAL=ROOT/'tgppo/runs/light-actor-20261105'
OUT=ORIGINAL/'cpu-recovery-20261105'


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def signatures(paths):
    result=[]
    for path in paths:
        policy,_=rt.load_actor(path);digest=hashlib.sha256();actions=[]
        for length in (7,31):
            rng=np.random.RandomState(913+length)
            c=rng.normal(size=(length,25)).astype('float32');m=rng.normal(size=53).astype('float32');n=rng.normal(size=8).astype('float32')
            mask=np.zeros(length,dtype=bool);mask[-2:]=True
            torch.manual_seed(200+length)
            inputs=policy._inputs(c,m,n,mask)
            with torch.no_grad():probs=policy.actor_network(inputs[0],inputs[2],inputs[1],inputs[3])
            digest.update(probs.numpy().tobytes());action,logp=policy.act(c,m,n,mask)
            digest.update(struct.pack('id',action,logp));actions.append([action,logp])
        result.append(dict(probability_action_sha256=digest.hexdigest(),actions=actions))
    return result


def probe_child(connection,paths):
    try:
        from pyscipopt import Model
        model=Model();version=model.version();model.freeProb()
        connection.send(dict(values=signatures(paths),scip_version=version,
            threads=len(os.listdir('/proc/self/task')),cuda_initialized=torch.cuda.is_initialized()))
    except BaseException as exc:connection.send(dict(error=repr(exc)))
    finally:connection.close()


class BudgetReached(RuntimeError):pass


def main():
    global torch,np,rt
    # Heavy modules are preimported by the launcher from a local working directory.
    import torch
    import numpy as np
    import launch
    import server
    from project import runtime as rt
    from project.utils.protocol import fingerprint
    from results import report
    assert read(OUT/'authorization.json')['authorized'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(OUT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget_path=OUT/'evaluation/budget.json';budget=read(budget_path);attempt=budget['attempts'][-1]
    assert attempt['stage']=='cached_cpu_retry' and 'elapsed_seconds' not in attempt
    started=attempt['started_epoch'];remaining=attempt['allocated_seconds']-(time.time()-started)
    state=dict(status='preparing_cached_cpu',pid=os.getpid(),host=os.uname().nodename,
        started_epoch=started,additional_budget_seconds=5400,expected_model_tasks=160,expected_native_tasks=10,
        training_steps=0,worker_start_method='fork',batching='interleaved global four-worker queue',
        old_evaluation_preserved=True,prior_charged_seconds=budget['charged_before_attempt'])
    write(OUT/'status.json',state)
    def expire(signum,frame):raise BudgetReached('Remaining authorized CPU budget reached')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,max(.001,remaining-20))
    try:
        torch.set_num_threads(1);launch.verify()
        assert not torch.cuda.is_initialized() and len(os.listdir('/proc/self/task'))==1
        configs={v:launch.protocol(v)[0] for v in launch.VARIANTS}
        info=server.preflight(ROOT,configs['b0'],'cpu');cutoffs=read(ROOT/'tgppo/datasets/cutoffs.json')
        cohort=OUT/'evaluation/episode-0100'/fingerprint(info['stack'])[:24]
        cohort.mkdir(parents=True,exist_ok=True)
        cache=Path(tempfile.mkdtemp(prefix='tgppo-eval-',dir='/tmp'));(cache/'actors').mkdir();(cache/'instances').mkdir()
        assert cache.stat().st_dev!=ROOT.stat().st_dev,'Cache must reside outside Ceph'
        sources={};models=[];labels={}
        for v in launch.VARIANTS:
            c=configs[v]
            for seed in (11,22):
                out=server.group_dir(launch.protocol(v)[1],'B0',seed);cp=out/'episode-0100.pt';actor=out/'actor-0100.pt'
                assert actor.exists();payload=rt.read_checkpoint(cp);launch.ensure_export(cp,actor,payload)
                assert payload['config']==server.params(ROOT,c,'B0',seed,'cuda')
                assert payload['progress']['next_episode']==100 and payload['progress']['phase']=='ready'
                label='{}-s{}'.format(v,seed);local=cache/'actors'/(label+'.pt');shutil.copyfile(actor,local)
                digest=launch.sha256(actor);assert launch.sha256(local)==digest;local.chmod(0o444)
                item=dict(variant=v,seed=seed,checkpoint=str(cp),checkpoint_sha256=launch.sha256(cp),actor_sha256=digest,
                    actor=str(actor),cached_actor=str(local),actor_kind=c['actor_kind'],pretraining=c['expert_pretraining'],
                    bc=read(out/'pretrain/result.json') if c['expert_pretraining']=='bc' else None,
                    training_environments=read(ORIGINAL/'jobs'/(label+'.json'))['attempts'],
                    updates=payload['update_counter'],progress=payload['progress'])
                models.append(item);labels[label]=item
                del payload
        gc.collect()
        for name in configs['b0']['development']:
            source=ROOT/'tgppo/datasets/train_instances'/(name+'.mps.gz');dest=cache/'instances'/source.name
            shutil.copyfile(source,dest);assert launch.sha256(source)==launch.sha256(dest)
            sources[name]=dict(source=str(source),cache=str(dest),sha256=launch.sha256(dest))
        paths=[labels[k]['cached_actor'] for k in ('b0-s11','mlp_bc-s11','set_bc-s11','isab_bc-s11')]
        expected=signatures(paths);gc.collect()
        assert len(os.listdir('/proc/self/task'))==1 and not torch.cuda.is_initialized()
        original_context=rt.mp.get_context;context=original_context('fork');receive,send=context.Pipe(duplex=False)
        child=context.Process(target=probe_child,args=(send,paths));child.start();send.close()
        try:
            if not receive.poll(60):raise RuntimeError('CPU fork inference check did not finish; no solver jobs launched')
            observed=receive.recv();child.join(timeout=5)
            assert observed.get('values')==expected and observed['threads']==1 and not observed['cuda_initialized'],observed
        finally:
            if child.is_alive():child.kill();child.join(timeout=1)
            receive.close()
        write(OUT/'cpu_fork_verification.json',dict(passed=True,parent_threads=1,child=observed,
            paths=paths,scip_model_constructed_not_solved=True,probabilities_and_actions_bitwise_equal=True,
            source_references=['https://docs.python.org/3.8/library/multiprocessing.html',
                'https://raw.githubusercontent.com/pytorch/pytorch/v1.13.1/docs/source/notes/multiprocessing.rst']))
        rt.mp.get_context=lambda method=None: original_context('fork' if method=='spawn' else method)
        ordered_labels=['native']+list(labels);tasks=[]
        for index,name in enumerate(configs['b0']['development']):
            for ss in configs['b0']['evaluation_solver_seeds']:
                rotation=(2*index+ss)%len(ordered_labels);order=ordered_labels[rotation:]+ordered_labels[:rotation]
                for label in order:
                    m=labels.get(label);seed=m['seed'] if m else 11;v=m['variant'] if m else 'native'
                    canonical=('dev-' if m else 'native-')+name+'-s'+str(ss)
                    jid='t{:03d}-{}-{}-s{}'.format(len(tasks),label,name,ss)
                    job=server.bound_job(ROOT,configs[v if m else 'b0'],cutoffs,name,ss,seed,'eval' if m else 'native',
                        m['cached_actor'] if m else None,job_id=jid,instances=cache/'instances')
                    job['sampling_seed']=rt.action_seed(seed,canonical)
                    tasks.append(dict(variant=v,seed=seed if m else None,canonical_sampling_job_id=canonical,job=job))
        assert len(tasks)==170
        manifest=dict(tasks=tasks,models=models,stack=info['stack'],host=info['host'],sources=sources,
            cache=str(cache),worker_start_method='fork',workers=4,threads_per_worker=1,
            native_reference='fresh10, same host as all160 model evaluations',driver_sha256=launch.sha256(__file__))
        write(OUT/'cached_manifest.json',manifest)
        state.update(status='evaluating_cached_cpu',setup_seconds=time.time()-started,stack=info['stack'],cache=str(cache))
        write(OUT/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
        rows=rt.run_jobs([t['job'] for t in tasks],cohort/'cached-global-000',max_workers=4,watchdog_seconds=180)
        by_id={r['job_id']:r for r in rows};write(OUT/'cached_evaluation_rows.json',rows)
        if len(rows)!=170 or any(not r.get('valid') for r in rows):raise RuntimeError('Incomplete global CPU attempt; failures retained, no automatic retry')
        native=[]
        for m in models:m['rows']=[]
        model_map={(m['variant'],m['seed']):m for m in models}
        for task in tasks:
            r=dict(by_id[task['job']['job_id']],canonical_sampling_job_id=task['canonical_sampling_job_id'])
            assert not r['cuda_initialized'] and r['worker_cpu_complete']
            assert r['sampling_seed']==task['job']['sampling_seed']
            if task['variant']=='native':native.append(r)
            else:
                assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
                model_map[(task['variant'],task['seed'])]['rows'].append(r)
        for m in models:
            assert launch.sha256(m['actor'])==launch.sha256(m['cached_actor'])==m['actor_sha256']
            assert launch.sha256(m['checkpoint'])==m['checkpoint_sha256']
            folder=cohort/('{}-s{}'.format(m['variant'],m['seed']));folder.mkdir(exist_ok=True)
            write(folder/'selected_results.json',m['rows'])
        write(cohort/'native.json',native);write(cohort/'models-seeds-11-22.json',models)
        destination=OUT/'results';report(destination,models,native,100,(11,22),info)
        write(destination/'complete.json',dict(complete=True,seeds=[11,22],episode=100,model_tasks=160,cohort=str(cohort),
            stack=info['stack'],native_reference='fresh10, same host as all160 model evaluations',
            worker_start_method='fork',driver_sha256=launch.sha256(__file__),cpu_fork_verification=str(OUT/'cpu_fork_verification.json')))
        assert not torch.cuda.is_initialized()
        state.update(status='complete',model_tasks=160,native_tasks=10,weights_unchanged=True,cuda_initialized=False)
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        attempt['elapsed_seconds']=time.time()-started;budget['charged_seconds']=sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in budget['attempts'])
        state.update(charged_wall_seconds=attempt['elapsed_seconds'],total_charged_seconds=budget['charged_seconds'])
        write(budget_path,budget);write(OUT/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
