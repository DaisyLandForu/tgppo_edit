"""Custom-job entry: read-only feature preparation, four bounded head fits, audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from collections import defaultdict

START=time.monotonic()
import numpy as np
import torch
from torch.distributions import Categorical
from common import (PACKAGE,RESEARCH,sha,canonical,save,write_csv,read_csv,protocol,job_config,
                    fingerprint,set_threads,seed_all,make_head,evaluate,summarize_records,
                    backward_batch,objective,load_prepared)
from frozen_model.actor import Actor


def utc():return datetime.now(timezone.utc).isoformat()


def run_root(value):
    root=Path(value).resolve() if value else (RESEARCH/protocol()['run_directory']).resolve()
    forbidden=[RESEARCH,RESEARCH/'tgppo/runs/server6-20261022',RESEARCH/'docs/diagnostics']
    if any(root==p or p in root.parents for p in forbidden[1:]) or root in [RESEARCH,*RESEARCH.parents]:
        raise ValueError('Run directory overlaps protected source/history')
    return root


def identity(run):
    marker=run/'experiment.json';expected=dict(schema=protocol()['schema'],protocol_sha256=canonical(protocol()))
    if marker.exists():
        if json.loads(marker.read_text())!=expected:raise ValueError('Existing run belongs to another protocol')
    else:
        if run.exists() and any(run.iterdir()):raise ValueError('Nonempty directory has no experiment identity')
        run.mkdir(parents=True,exist_ok=True);save(marker,expected)


def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        f.close();raise ValueError('Another live process owns '+str(path))
    return f


def capture(actor,row):
    seen={}
    def take(name):
        def hook(mod,inp,out):seen[name]=out.detach().clone()
        return hook
    hooks=[actor.tree_embedding.register_forward_hook(take('tree')),
           actor.tree_refinement.register_forward_hook(take('features'))]
    try:
        with torch.no_grad():
            x,n,m=[torch.as_tensor(row[k],dtype=torch.float32)[None] for k in ['cands_state','node_state','mip_state']]
            p=actor(x,n,m)[0]
            replay=torch.softmax(actor.output_layer(seen['features'],seen['tree']),-1)[0]
    finally:
        for h in hooks:h.remove()
    if not torch.equal(p,replay):raise ValueError('Cached head path does not exactly replay the full Actor')
    return seen['features'][0].contiguous(),seen['tree'][0].contiguous(),float(Categorical(p).log_prob(torch.tensor(row['action'])))


def prepare_worker(run):
    p=protocol();before=fingerprint();replay=RESEARCH/'docs/diagnostics/tgppo_20261102_ppo_replay'
    location=RESEARCH/'docs/diagnostics/tgppo_20261102_signal_localization'
    if (run/'prepared/manifest.json').exists():
        load_prepared(run,True);print('Prepared inputs already verified; no recomputation.');return
    target=run/'prepared';target.mkdir(exist_ok=True)
    frozen=json.loads((replay/'frozen_inputs.json').read_text())
    sources=dict(frozen['original_sources']);sources.update(frozen['trajectory_and_config_hashes'])
    for rel,h in sources.items():
        if sha(RESEARCH/rel)!=h:raise ValueError('Accepted input drift: '+rel)
    # Verify the accepted delivery seals before reusing their results.
    for folder in [replay,location]:
        manifest=json.loads((folder/'delivery_manifest.json').read_text())['files']
        if not all(sha(folder/rel)==h for rel,h in manifest.items()):raise ValueError('Prior delivery drift')
        sources[str((folder/'delivery_manifest.json').relative_to(RESEARCH))]=sha(folder/'delivery_manifest.json')
    for rel in ['batch_manifest.csv','advantages.csv','frozen_inputs.json']:
        sources[str((replay/rel).relative_to(RESEARCH))]=sha(replay/rel)
    sources[str((location/'observations.pkl').relative_to(RESEARCH))]=sha(location/'observations.pkl')
    for name in ['actor.py','modules.py']:
        source=RESEARCH/'tgppo/experiments/server-20261022/code/project/policy'/name
        if sha(PACKAGE/'code/frozen_model'/name)!=sha(source):raise ValueError('Independent model source differs')
    monitor=pickle.loads((location/'observations.pkl').read_bytes())
    selected=read_csv(replay/'batch_manifest.csv')
    specs={r['model']:r for r in frozen['checkpoint_specs']}
    caches={};checks=[]
    for seed in p['seeds']:
        model=f'B0-s{seed}';spec=specs[model];cp_path=RESEARCH/spec['path']
        if sha(cp_path)!=spec['sha256']:raise ValueError('Checkpoint drift')
        sources[spec['path']]=spec['sha256'];cp=torch.load(cp_path,map_location='cpu');cfg=cp['config']
        arch=dict(cfg['state_dims'],**{k:cfg[k] for k in ['hidden_dim','num_heads','num_layers','dropout']})
        actor=Actor(**arch).eval();actor.load_state_dict(cp['actor'])
        episodes={}
        folder=RESEARCH/f'tgppo/runs/server6-20261022/training/{model}/collection/e00100/attempt-000'
        for path in sorted(folder.glob('e*.pkl')):
            if 100<=int(path.stem[1:6])<=103:
                payload=pickle.loads(path.read_bytes())
                for ep in payload['episodes']:
                    if ep['policy_version']!=cp['policy_version']:raise ValueError('Stale trajectory')
                    episodes[ep['episode_id']]=ep
        train=[];mapping=[];source_max_error=0.
        chosen=[r for r in selected if r['model']==model]
        if len(chosen)!=p['logical_batch_size']:raise ValueError('Incomplete accepted minibatch')
        for s in chosen:
            r=episodes[s['episode']]['transitions'][int(s['transition'])]
            features,tree,logp=capture(actor,r)
            source_max_error=max(source_max_error,abs(float(np.exp(logp-float(s['old_log_prob'])))-1))
            if r['action']!=int(s['action']) or r['log_prob']!=float(s['old_log_prob']):raise ValueError('Batch/action metadata mismatch')
            train.append(dict(features=features,tree=tree,action=r['action'],old_logprob=r['log_prob'],
                              advantage=float(s['normalized_advantage']),instance=s['instance'],key=f"{s['episode']}:t{s['transition']}"))
            mapping.append(dict(position=int(s['position']),global_index=int(s['global_index']),episode=s['episode'],transition=int(s['transition'])))
        monitors=[]
        for key,instance,r in monitor:
            features,tree,_=capture(actor,r)
            monitors.append(dict(features=features,tree=tree,action=r['action'],old_logprob=r['log_prob'],
                                 instance=instance,key=key))
        if source_max_error>p['initial_behavior_ratio_max_error']:raise ValueError('Full accepted minibatch old-policy mismatch')
        cache=dict(schema='cached-head-data-v1',seed=seed,hidden_dim=cfg['hidden_dim'],train=train,monitor=monitors,
                   head_state={k:v.detach().clone() for k,v in actor.output_layer.state_dict().items()},
                   source_checkpoint=spec,batch_mapping=mapping,real_full_episode_advantages=True,
                   source_policy_version=cp['policy_version'],source_train_states=3735 if seed==11 else 5902)
        paired=[]
        for variant in p['variants']:
            head=make_head(cache,variant);records=evaluate(head,train,p)
            if max(abs(r['old_action_ratio']-1) for r in records)>p['initial_behavior_ratio_max_error']:raise ValueError('Uniform initialization incompatible with behavior batch')
            if max(r['tv_uniform'] for r in records)>p['initial_pair_probability_atol']:raise ValueError('Initial head is not uniform')
            paired.append(records)
            checks.append(dict(seed=seed,variant=variant,source_ratio_max_error=source_max_error,
                               initial_ratio_max_error=max(abs(r['old_action_ratio']-1) for r in records),
                               trainable_parameters=sum(q.numel() for q in head.parameters()),
                               initial_uniform_max_tv=max(r['tv_uniform'] for r in records)))
        if any(np.max(np.abs(a['probability']-b['probability']))>p['initial_pair_probability_atol'] for a,b in zip(*paired)):
            raise ValueError('Initial policies differ across arms')
        dest=target/f'seed-{seed}.pt';temp=dest.with_suffix('.tmp');torch.save(cache,temp);temp.replace(dest)
        caches[str(seed)]=dict(path=str(dest.relative_to(run)),sha256=sha(dest),train_states=len(train),monitor_states=len(monitors),
                               source_checkpoint_sha256=spec['sha256'],paired_initialization_verified=True)
        print(f'Prepared seed {seed}: 256 train + 164 monitor, source ratio max error {source_max_error:.3g}',flush=True)
        del actor,cp,cache,train,monitors,head,paired
    if fingerprint()!=before:raise ValueError('Code changed during preparation')
    if any(sha(RESEARCH/rel)!=h for rel,h in sources.items()):raise ValueError('Source input changed during preparation')
    save(target/'manifest.json',dict(schema='prepared-head-probe-v1',created_at_utc=utc(),protocol_sha256=canonical(p),
                                   code_fingerprint=before,input_source_hashes=sources,caches=caches,checks=checks,
                                   torch_version=torch.__version__,numpy_version=np.__version__,
                                   optimizer_steps=0,new_solver_tasks=0,cuda_initialized=torch.cuda.is_initialized()))
    if torch.cuda.is_initialized() or 'pyscipopt' in sys.modules:raise ValueError('Preparation initialized forbidden GPU/solver')


def run_head_fit(cache,variant,p,directory,provenance):
    """Only this function performs real experimental optimizer steps."""
    seed_all(cache['seed']);head=make_head(cache,variant);opt=torch.optim.Adam(head.parameters(),lr=p['actor_lr'])
    train,monitor=cache['train'],cache['monitor']
    initial_train=evaluate(head,train,p);initial_monitor=evaluate(head,monitor,p)
    if max(abs(r['old_action_ratio']-1) for r in initial_train)>p['initial_behavior_ratio_max_error']:
        raise ValueError('Initial behavior ratio fails; do not train')
    history=[];states=[];steps=0;status='completed'
    started=time.monotonic()
    for step in range(p['max_actor_steps']+1):
        tr=evaluate(head,train,p,initial_train);mo=evaluate(head,monitor,p,initial_monitor)
        loss=objective(tr,train,p)
        row=dict(step=step,elapsed_seconds=time.monotonic()-started,**loss,
                 train=summarize_records(train,tr),monitor=summarize_records(monitor,mo))
        history.append(row)
        for dataset,data,records in [('train',train,tr),('monitor',monitor,mo)]:
            for sample,r in zip(data,records):
                states.append(dict(step=step,dataset=dataset,key=sample['key'],instance=sample['instance'],
                                   **{k:v for k,v in r.items() if k not in ['probability','logits']}))
        save(directory/'history.json',history)
        write_csv(directory/'state_metrics.csv',states)
        print(json.dumps(dict(step=step,surrogate=loss['surrogate_loss'],train_tv=row['train']['tv_initial_p50'],
                              monitor_tv=row['monitor']['tv_initial_p50'],train_kl=row['train']['mean_kl_initial'])),flush=True)
        if row['train']['mean_kl_initial']>p['mean_training_kl_stop']:
            status='stopped_kl';break
        if step==p['max_actor_steps']:break
        opt.zero_grad(set_to_none=True);actual=backward_batch(head,train,p)
        for q in head.parameters():
            if q.grad is not None and not torch.isfinite(q.grad).all():raise ValueError('Nonfinite head gradient')
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),p['gradient_clip'],error_if_nonfinite=True)
        # The journal is durable before every step; no implicit replay on retry.
        save(directory/'inflight.json',dict(next_step=step+1,previous_committed_step=steps,entered_at_utc=utc()))
        opt.step();steps+=1
        temporary=directory/'latest.tmp'
        torch.save(dict(schema='head-probe-checkpoint-v1',variant=variant,seed=cache['seed'],step=steps,
                        head=head.state_dict(),optimizer=opt.state_dict(),
                        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=__import__('random').getstate(),
                        provenance=provenance,full_actor_training=False),temporary)
        temporary.replace(directory/'latest.pt')
        save(directory/'steps.json',dict(committed_steps=steps,maximum_steps=p['max_actor_steps'],last_gradient_norm=float(norm),
                                        last_actual_microbatch_loss=actual))
        (directory/'inflight.json').unlink()
    result=dict(status=status,variant=variant,seed=cache['seed'],completed_steps=steps,maximum_steps=p['max_actor_steps'],
                first=history[0],last=history[-1],head_parameters=sum(q.numel() for q in head.parameters()),
                optimizer='fresh Adam; original moments not restored',matched_uniform_readout=True,
                new_solver_tasks=0,critic_updates=0,full_actor_training=False,provenance=provenance,
                torch_version=torch.__version__,device='cpu',cpu_model=next((r.split(':',1)[1].strip() for r in Path('/proc/cpuinfo').read_text().splitlines() if r.startswith('model name')),'unknown'),
                result_scope='conditional fitting to saved real advantages; no expert-ranking/PDI claim')
    save(directory/'result.json',result)
    return result


def worker(run,job_id):
    p,m=load_prepared(run);c=job_config(job_id);directory=run/'jobs'/job_id
    cache_entry=m['caches'][str(c['seed'])];cache=torch.load(run/cache_entry['path'],map_location='cpu')
    prov=dict(job_id=job_id,job_config_sha256=canonical(c),protocol_sha256=canonical(p),
              code_fingerprint=m['code_fingerprint'],cache_sha256=cache_entry['sha256'],prepared_manifest_sha256=sha(run/'prepared/manifest.json'))
    result=run_head_fit(cache,c['variant'],p,directory,prov)
    if 'pyscipopt' in sys.modules or torch.cuda.is_initialized():raise ValueError('Forbidden solver/GPU use')
    if fingerprint()!=m['code_fingerprint'] or sha(run/cache_entry['path'])!=cache_entry['sha256']:
        raise ValueError('Code or input cache changed during the job')
    if result['status']!='completed':raise SystemExit(3)


def supervised(run,action,job_id=None):
    """A foreground child, per-job flock, durable budget, no automatic retries."""
    identity(run);p=protocol()
    if action=='run':
        job_config(job_id);load_prepared(run,True)
        directory=run/'jobs'/job_id;cap=p['job_wall_seconds'];child=['_worker','--job',job_id]
    else:
        directory=run/'preparation';cap=p['prepare_wall_seconds'];child=['_prepare']
    directory.mkdir(parents=True,exist_ok=True)
    with lock(directory/'.lock'):
        return supervised_locked(run,action,job_id,directory,cap,child)


def supervised_locked(run,action,job_id,directory,cap,child):
    budget_path=directory/'budget.json'
    if budget_path.exists():
        old=json.loads(budget_path.read_text())
        if old['status']=='completed':
            if action=='prepare':load_prepared(run,True)
            else:verify_job(run,job_id)
            print('Already completed and verified; no new computation.');return
        raise ValueError('A previous attempt is incomplete/failed. Preserved for audit; refusing a silent fresh retry: '+str(directory))
    save(budget_path,dict(status='running',started_at_utc=utc(),limit_seconds=cap,job_id=job_id,
                          note='running is not proof of a live process; flock establishes live ownership'))
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,str(Path(__file__).resolve()),*child,'--run-root',str(run)]
    elapsed_before=time.monotonic()-START
    timeout=max(1.,cap-5.-elapsed_before)
    code=1;state='failed';child_process=None;previous_handlers={}
    class InterruptedJob(Exception):pass
    def interrupt(signum,frame):
        if child_process is not None and child_process.poll() is None:
            os.killpg(child_process.pid,signum)
        raise InterruptedJob(signum)
    try:
        # The shell tee and platform both see progress; the child writes all
        # scientific records inside its uniquely locked directory.
        child_process=subprocess.Popen(cmd,env=env,start_new_session=True)
        for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
            previous_handlers[signum]=signal.signal(signum,interrupt)
        try:code=child_process.wait(timeout=timeout)
        except InterruptedJob as exc:
            try:child_process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child_process.pid,signal.SIGKILL);child_process.wait()
            code=128+int(exc.args[0])
        except subprocess.TimeoutExpired:
            os.killpg(child_process.pid,signal.SIGTERM)
            try:child_process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child_process.pid,signal.SIGKILL);child_process.wait()
            code=124
        state='completed' if code==0 else ('stopped_kl' if code==3 else ('budget_timeout' if code==124 else 'failed'))
    finally:
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
        if child_process is not None and child_process.poll() is None:
            os.killpg(child_process.pid,signal.SIGKILL);child_process.wait()
        elapsed=time.monotonic()-START
        save(budget_path,dict(status=state,exit_code=code,elapsed_seconds=elapsed,limit_seconds=cap,
                              finished_at_utc=utc(),job_id=job_id))
    if code==0 and action=='run':
        files={n:sha(directory/n) for n in ['result.json','history.json','state_metrics.csv','latest.pt','steps.json','budget.json']}
        save(directory/'delivery.json',dict(job_id=job_id,files=files))
    if code:raise SystemExit(code)
    print(json.dumps(dict(status=state,wall_seconds=time.monotonic()-START,limit=cap)),flush=True)


def verify_job(run,job_id):
    p,m=load_prepared(run);c=job_config(job_id);d=run/'jobs'/job_id
    if not (d/'delivery.json').exists():raise ValueError('Job not sealed: '+job_id)
    seal=json.loads((d/'delivery.json').read_text())
    if any(sha(d/n)!=h for n,h in seal['files'].items()):raise ValueError('Job delivery corruption: '+job_id)
    r=json.loads((d/'result.json').read_text());history=json.loads((d/'history.json').read_text())
    expected=dict(job_id=job_id,job_config_sha256=canonical(c),protocol_sha256=canonical(p),code_fingerprint=m['code_fingerprint'],
                  cache_sha256=m['caches'][str(c['seed'])]['sha256'],prepared_manifest_sha256=sha(run/'prepared/manifest.json'))
    if r['provenance']!=expected or r['completed_steps']!=p['max_actor_steps'] or r['status']!='completed':raise ValueError('Incomplete or mismatched result')
    if [x['step'] for x in history]!=list(range(p['max_actor_steps']+1)):raise ValueError('Missing step measurements')
    if r['new_solver_tasks'] or r['critic_updates'] or r['full_actor_training']:raise ValueError('Wrong scope')
    return r,history


def aggregate(run):
    p,m=load_prepared(run,True);jobs=[f'{v}-s{s}' for s in p['seeds'] for v in p['variants']]
    results={};missing={}
    for j in jobs:
        try:results[j]=verify_job(run,j)
        except (ValueError,OSError,KeyError) as exc:missing[j]=str(exc)
    report=run/'summary';report.mkdir(exist_ok=True)
    if missing:
        save(report/'completeness.json',dict(complete=False,verified=list(results),missing_or_failed=missing))
        raise ValueError('Four complete, compatible jobs are required; see summary/completeness.json')
    pairs=[];curves=[]
    for seed in p['seeds']:
        a,ah=results[f'original-s{seed}'];b,bh=results[f'linear-s{seed}']
        if a['provenance']['cache_sha256']!=b['provenance']['cache_sha256']:raise ValueError('Unpaired input caches')
        ar=a['last'];br=b['last']
        pairs.append(dict(seed=seed,steps=p['max_actor_steps'],
                          original_surrogate_change=ar['surrogate_loss']-a['first']['surrogate_loss'],
                          linear_surrogate_change=br['surrogate_loss']-b['first']['surrogate_loss'],
                          original_train_tv=ar['train']['tv_initial_p50'],linear_train_tv=br['train']['tv_initial_p50'],
                          original_monitor_tv=ar['monitor']['tv_initial_p50'],linear_monitor_tv=br['monitor']['tv_initial_p50'],
                          original_parameters=a['head_parameters'],linear_parameters=b['head_parameters']))
        for variant,history in [('original',ah),('linear',bh)]:
            for r in history:
                curves.append(dict(seed=seed,variant=variant,step=r['step'],surrogate_loss=r['surrogate_loss'],
                                   entropy=r['entropy'],train_kl=r['train']['mean_kl_initial'],
                                   train_tv=r['train']['tv_initial_p50'],monitor_tv=r['monitor']['tv_initial_p50']))
    write_csv(report/'paired_results.csv',pairs);write_csv(report/'curves.csv',curves)
    save(report/'completeness.json',dict(complete=True,verified=jobs,missing_or_failed={}))
    save(report/'summary.json',dict(complete=True,paired_results=pairs,
                                    interpretation='Conditional scoring-head fit on two fixed real-advantage batches. No solver/quality/generalization claim.',
                                    caveats=['two source training seeds, not four independent repeats','shared uniform readout and fresh Adam differ from historical continuation',
                                             'original nonlinear intermediate weights retained; linear comparator has fewer parameters','same batch reused for 20 steps','monitor instances are training instances']))
    lines=['# 离线评分头对照结果','', '**四个作业完整且通过哈希核验。** 以下只描述给定表示和真实优势上的条件学习响应，不代表候选排序或 PDI 收益。','',
           '| 种子 | 原头 surrogate 变化 | 线性头 surrogate 变化 | 原头监测 TV 中位数 | 线性头监测 TV 中位数 |','|---|---:|---:|---:|---:|']
    for r in pairs:lines.append(f"| {r['seed']} | {r['original_surrogate_change']:.6g} | {r['linear_surrogate_change']:.6g} | {r['original_monitor_tv']:.6g} | {r['linear_monitor_tv']:.6g} |")
    lines+=['','surrogate loss 越低表示这批固定目标拟合较好；TV 只表示偏好变化，不能判定行动正确。两个种子分别报告，不按监测状态数作显著性推断。',
            '原头与线性头参数量不同，这是诊断对照的一部分，不能包装成参数匹配的新方法结果。',
            '本阶段结束；后续在线采样、正式训练和求解评估需要新的范围与预算。','']
    (report/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=True,summary=str(report/'REPORT.md')),ensure_ascii=False))


def status(run):
    p=protocol();output={}
    for seed in p['seeds']:
        for variant in p['variants']:
            j=f'{variant}-s{seed}';d=run/'jobs'/j
            budget=json.loads((d/'budget.json').read_text()) if (d/'budget.json').exists() else {}
            steps=json.loads((d/'steps.json').read_text()) if (d/'steps.json').exists() else {}
            live=False
            if (d/'.lock').exists():
                held=(d/'.lock').open('r+')
                try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:live=True
                finally:held.close()
            output[j]=dict(recorded_status=budget.get('status','not_started'),live_lock_owned=live,
                           committed_steps=steps.get('committed_steps',0),sealed=(d/'delivery.json').exists())
    print(json.dumps(output,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preflight','run','status','summarize','_prepare','_worker'])
    parser.add_argument('--run-root');parser.add_argument('--job')
    a=parser.parse_args();set_threads();root=run_root(a.run_root)
    if a.action in ['prepare','run']:supervised(root,a.action,a.job)
    elif a.action=='_prepare':prepare_worker(root)
    elif a.action=='_worker':worker(root,a.job)
    elif a.action=='preflight':
        p,m=load_prepared(root,True)
        print(json.dumps(dict(status='passed',jobs=4,device='cpu',torch=torch.__version__,prepared_caches=m['caches'],
                              solver_tasks=0,max_head_steps_per_job=p['max_actor_steps'],wall_limit_per_job=p['job_wall_seconds']),indent=2))
    elif a.action=='status':status(root)
    elif a.action=='summarize':aggregate(root)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise SystemExit(1)
