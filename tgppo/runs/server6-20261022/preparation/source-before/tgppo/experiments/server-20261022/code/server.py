"""Portable six-job server experiment; CPU SCIP collection, optional CUDA PPO.

No stage6/report or held-out test sweep is launched automatically.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import socket
import time
import uuid
import numpy as np
import torch
from project.runtime import (build_agent,config_with_protocol,seed_all,save_checkpoint,load_agent,
    load_actor,read_checkpoint,make_job,job_identity,run_jobs,action_seed,cpu_seconds,ARCH_KEYS)
from project.utils.protocol import fingerprint,sha256,write_json,validate_baseline

DEFAULT_ROOT=Path(__file__).resolve().parents[4]


def read(path):return json.loads(Path(path).read_text())


def plan(root,path):
    c=read(path)
    if c['schema']!='server-six-jobs-v1':raise ValueError('Explicit server protocol required')
    if set(c['groups'])-{'B0','R0','Rjoint'} or len(set(c['training_seeds']))!=len(c['training_seeds']):
        raise ValueError('Invalid groups or duplicate training seeds')
    if c['max_episodes']%c['batch_episodes'] or any(n%c['batch_episodes'] for n in c['milestones']):
        raise ValueError('Milestones must be complete collection-batch boundaries')
    if c['milestones'][-1]!=c['max_episodes'] or sorted(set(c['milestones']))!=c['milestones']:
        raise ValueError('Sorted milestones must end at final episode budget')
    original=read(root/'docs/stage1/protocol_draft.json')
    code=Path(__file__).resolve().parent
    source_files=sorted(code.rglob('*.py'))
    c.update(train=original['data']['train'],development=original['data']['development'],
             instance_hashes=original['data']['hashes'],
             public_hparams_sha256=sha256(root/'tgppo/configs/paper_hparams.json'),
             calibration_statistics_sha256=sha256(root/'tgppo/runs/stage4-calibration/final_train_only_statistics.json'),
             runtime_sources_sha256=fingerprint({str(p.relative_to(code)):sha256(p) for p in source_files}))
    if set(c['train'])&set(c['development']):raise ValueError('Train/development overlap')
    return c


def stack(root):
    from pyscipopt import Model
    methods=('getCandsState','getMIPState','getNodeState','getIntegralSnapshot','getOpenNodeSnapshot')
    if not all(hasattr(Model,m) for m in methods):raise RuntimeError('Custom SCIP interfaces missing')
    m=Model();version=m.version();m.freeProb()
    if int(version)!=6:raise RuntimeError('Expected SCIP6 custom binding')
    model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')),platform.machine())
    lib=root/'tgppo/solver/lib/libscip.so'
    so=Path(importlib.util.find_spec('pyscipopt.scip').origin)
    return dict(cpu_model=model,machine=platform.machine(),scip_version=version,
        libscip_sha256=sha256(lib),binding_sha256=sha256(so),torch_version=torch.__version__,
        python_abi='{}.{}'.format(*__import__('sys').version_info[:2]))


def params(root,c,group,seed,device):
    p=read(root/'tgppo/configs/paper_hparams.json')
    if c.get('engineering_mode'):p.update(c.get('model_overrides',{}))
    stats_path=root/'tgppo/runs/stage4-calibration/final_train_only_statistics.json'
    stats=read(stats_path)
    p.update(learner_device=device,server_group=group,server_training_seed=seed,
             server_protocol_sha256=fingerprint(c),task_time_limit=c['time_limit'],
             initialization='paired fresh public weights; no offline adapter warm start')
    if group!='B0':p.update(online_group=group,target_mean=stats['mean'],target_std=stats['std'],
        target_statistics_sha256=sha256(stats_path),extra_initialization_seed=1000+seed)
    return config_with_protocol(p)


def schedule(seed,c):
    # Every 20-task cycle covers the complete training split, including zero-decision cases.
    rng=np.random.RandomState(seed);result=[]
    while len(result)<c['max_episodes']:
        order=list(c['train']);rng.shuffle(order);result+=order
    return result[:c['max_episodes']]


def digest_agent(agent,configuration):
    import hashlib
    h=hashlib.sha256(fingerprint(configuration).encode())
    for prefix,net in [('actor',agent.actor_network),('critic',agent.critic_network)]:
        for k,v in sorted(net.state_dict().items()):
            h.update((prefix+k).encode());h.update(v.detach().cpu().numpy().tobytes())
    h.update(str(agent.policy_version).encode())
    return h.hexdigest()


def group_dir(run,group,seed):return run/'training'/('{}-s{}'.format(group,seed))


def bound_job(root,c,cutoffs,name,seed,training_seed,mode,checkpoint=None,job_id=None,instances=None):
    j=make_job(job_id or '{}-{}-s{}'.format(mode,name,seed),instances or root/'tgppo/datasets/train_instances',
        name,c['time_limit'],seed,training_seed,cutoffs,mode,checkpoint)
    j['address_space_limit_bytes']=c['worker_address_space_bytes']
    return j


def cached_jobs(jobs,base,identity,workers,watchdog):
    """Reuse only valid outputs from a matching immutable attempt manifest.

    Invalid attempts remain on disk; explicit restart retries missing/failed jobs
    in a new attempt, retaining every prior cost/error record.
    """
    base.mkdir(parents=True,exist_ok=True)
    lock=base/'identity.json'
    if lock.exists() and read(lock)!=identity:raise ValueError('Cached job identity changed')
    if not lock.exists():write_json(lock,identity)
    found={}
    for directory in sorted(base.glob('attempt-*')):
        manifest=directory/'manifest.json'
        if not manifest.exists() or read(manifest)['identity']!=identity:continue
        for job in jobs:
            f=directory/(job['job_id']+'.json');trajectory=directory/(job['job_id']+'.pkl')
            if not f.exists():continue
            row=read(f)
            if row.get('valid') and (job['mode']!='collect' or trajectory.exists()):
                found[job['job_id']]=(row,trajectory)
    missing=[j for j in jobs if j['job_id'] not in found]
    if missing:
        directory=base/('attempt-{:03d}'.format(len(list(base.glob('attempt-*')))))
        directory.mkdir()
        write_json(directory/'manifest.json',dict(identity=identity,jobs=missing))
        rows=run_jobs(missing,directory,workers,watchdog)
        for row in rows:
            if not row.get('valid'):raise RuntimeError('Invalid jobs retained at '+str(directory))
            found[row['job_id']]=(row,directory/(row['job_id']+'.pkl'))
    ordered=[found[j['job_id']] for j in sorted(jobs,key=lambda j:j['job_id'])]
    write_json(base/'selected_results.json',[r for r,_ in ordered])
    return ordered


def baseline_bundle(root,c,run):
    b=read(run/'normalization/baselines.json')
    if b['protocol_sha256']!=fingerprint(c) or b['stack']!=stack(root):
        raise ValueError('Baseline provenance differs from this CPU/software/protocol; remeasure on target')
    return b['baselines']


def baseline(root,c,run):
    signature=stack(root);cutoffs=read(root/'tgppo/datasets/cutoffs.json');jobs=[]
    for name in c['train']+c['development']:
        if sha256(root/'tgppo/datasets/train_instances'/(name+'.mps.gz'))!=c['instance_hashes'][name]:
            raise ValueError('Frozen dataset changed: '+name)
        for s in c['solver_seeds']:jobs.append(bound_job(root,c,cutoffs,name,s,11,'native'))
    pairs=cached_jobs(jobs,run/'normalization/jobs',dict(protocol=fingerprint(c),stack=signature),
                      c['workers'],c['watchdog'])
    byid={r['job_id']:r for r,_ in pairs};values={}
    for job in jobs:
        ident=job_identity(job);record=dict(identity=ident,result=byid[job['job_id']])
        validate_baseline(record,ident);values[fingerprint(ident)]=record
    write_json(run/'normalization/baselines.json',dict(protocol_sha256=fingerprint(c),
        stack=signature,host=socket.gethostname(),baselines=values))


class Interrupted(RuntimeError):pass


def interrupted(signum,frame):raise Interrupted('Job signal {}; last atomic recovery checkpoint retained'.format(signum))


def preflight(root,c,device):
    torch.set_num_threads(1);s=stack(root)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing silent CPU fallback')
        gpu=torch.cuda.get_device_name(0)
        if c.get('expected_gpu') and c['expected_gpu'] not in gpu and not c.get('engineering_mode'):
            raise RuntimeError('Expected '+c['expected_gpu']+', got '+gpu)
    else:gpu=None
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<c['workers'] and not c.get('engineering_mode'):raise RuntimeError('Allocated CPU quota too small')
    return dict(stack=s,gpu=gpu,device=device,allowed_cpus=allowed,host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))


def milestone_save(directory,agent,configuration,state,c):
    """Save only stable pre/post-update states, never interrupted half-updates."""
    save_checkpoint(directory/'latest.pt',agent,configuration,state)
    cursor=state['next_episode']
    if state['phase']=='ready' and cursor in c['milestones']:
        save_checkpoint(directory/('episode-{:04d}.pt'.format(cursor)),agent,configuration,state)


def train(root,c,run,group,seed,resume,device,stop_after_batches=None):
    if group not in c['groups'] or seed not in c['training_seeds']:raise ValueError('Job outside chosen matrix')
    if device=='cpu' and not c.get('engineering_mode'):raise ValueError('CPU training is engineering-only in this server protocol')
    info=preflight(root,c,device);bases=baseline_bundle(root,c,run)
    if not c.get('engineering_mode'):
        gate_path=run/'gpu-bridge/verification.json'
        if not gate_path.exists():raise RuntimeError('Run server-run/gpu_bridge.sh on the target GPU first')
        gate=read(gate_path)
        if not gate.get('passed') or gate['parent_protocol_sha256']!=fingerprint(c) or gate['stack']!=info['stack']:
            raise RuntimeError('Target GPU bridge not passed under this protocol/CPU/software')
    out=group_dir(run,group,seed);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.train.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    configuration=params(root,c,group,seed,device);seed_all(seed)
    if (out/'latest.pt').exists():
        if not resume:raise FileExistsError('Use explicit --resume for existing run')
        agent,payload=load_agent(out/'latest.pt',device=device)
        if payload['config']!=configuration:raise ValueError('Resume configuration changed')
        state=payload['progress']
    else:
        if (out/'config.json').exists():raise RuntimeError('Uncommitted existing run; inspect instead of resetting')
        agent=build_agent(configuration,device=device)
        agent.task_time_limit=c['time_limit']
        state=dict(next_episode=0,trained_episodes=0,phase='ready',optimizer_steps=0,history=[],
                   protocol_sha256=fingerprint(c),group=group,seed=seed)
        write_json(out/'config.json',configuration)
        write_json(out/'task_schedule.json',schedule(seed,c))
        milestone_save(out,agent,configuration,state,c)
    agent.task_time_limit=c['time_limit']
    initial_fp=out/'initialization.json'
    if not initial_fp.exists():
        write_json(initial_fp,dict(actor_fingerprint=digest_actor(agent.actor_network),
            common_critic_fingerprint=digest_actor(agent.critic_network,exclude_extra=True),
            public_torch_rng_sha256=__import__('hashlib').sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
    write_json(out/'preflight.json',info)
    attempt=out/'attempts'/(uuid.uuid4().hex+'.json');start=time.monotonic();batch_count=0
    data=read(root/'tgppo/datasets/cutoffs.json');order=read(out/'task_schedule.json')
    def heartbeat(**extra):
        write_json(attempt,dict(pid=os.getpid(),job=info['slurm_job_id'],host=info['host'],
            learner_cpu_seconds=cpu_seconds(),attempt_wall_seconds=time.monotonic()-start,
            phase=state['phase'],next_episode=state['next_episode'],complete=False,**extra))
    oldhandlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
    for s in oldhandlers:signal.signal(s,interrupted)
    try:
        heartbeat()
        while state['next_episode']<c['max_episodes'] or state['phase']=='collected':
            if state['phase']=='ready':
                begin=state['next_episode'];end=min(begin+c['batch_episodes'],c['max_episodes'])
                collection_fp=digest_agent(agent,configuration);jobs=[]
                for idx in range(begin,end):
                    name=order[idx];ss=c['solver_seeds'][(idx//len(c['train']))%len(c['solver_seeds'])]
                    job=bound_job(root,c,data,name,ss,seed,'collect',out/'latest.pt',
                                  job_id='e{:05d}-{}-s{}'.format(idx,name,ss))
                    ident=job_identity(job);job['baseline']=bases[fingerprint(ident)]
                    validate_baseline(job['baseline'],ident);jobs.append(job)
                pairs=cached_jobs(jobs,out/'collection'/('e{:05d}'.format(begin)),
                    dict(policy_fingerprint=collection_fp,protocol=fingerprint(c)),c['workers'],c['watchdog'])
                for row,path in pairs:
                    if row['policy_version']!=agent.policy_version:raise ValueError('Stale cached trajectory')
                    with path.open('rb') as f:agent.memory.import_dict(pickle.load(f))
                state.update(next_episode=end,phase='collected')
                # Recovery contains complete trajectories, raw values, both optimizers and RNG.
                save_checkpoint(out/'latest.pt',agent,configuration,state)
                heartbeat()
            begin_cpu=cpu_seconds();begin_wall=time.monotonic()
            event_start=event_end=None
            if device.startswith('cuda'):
                event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
                event_start.record()
            metrics=agent.learn(microbatch_size=c['microbatch_size'],
                progress_callback=lambda p:heartbeat(learning_progress=p))
            if event_end is not None:event_end.record();torch.cuda.synchronize()
            metrics.update(episodes_seen=state['next_episode'],learning_cpu_seconds=cpu_seconds()-begin_cpu,
                learning_wall_seconds=time.monotonic()-begin_wall,
                gpu_event_seconds=event_start.elapsed_time(event_end)/1000. if event_start is not None else None,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if event_start is not None else 0)
            if metrics['updated']:
                state['trained_episodes']=state['next_episode'];state['optimizer_steps']+=metrics['num_updates']
            state['history'].append(metrics);state['phase']='ready'
            milestone_save(out,agent,configuration,state,c)
            write_json(out/'updates.json',state['history']);heartbeat();batch_count+=1
            print(json.dumps(metrics),flush=True)
            if stop_after_batches and batch_count>=stop_after_batches:break
        finished=state['next_episode']==c['max_episodes'] and state['phase']=='ready'
        if finished:
            write_json(out/'summary.json',dict(status='training_complete',group=group,seed=seed,
                episodes_seen=state['next_episode'],trained_episodes=state['trained_episodes'],
                complete_ppo_updates=agent.update_counter,optimizer_steps=state['optimizer_steps'],
                pending_transitions=len(agent.memory),final_checkpoint_sha256=sha256(out/'latest.pt'),
                protocol_sha256=fingerprint(c),gpu=info['gpu']))
        heartbeat(attempt_finished=True,training_finished=finished)
        return state
    except BaseException as exc:
        heartbeat(error=repr(exc),attempt_interrupted=True)
        raise
    finally:
        for s,h in oldhandlers.items():signal.signal(s,h)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def digest_actor(network,exclude_extra=False):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(network.state_dict().items()):
        if exclude_extra and k.startswith(('extra.','extra_projection.','time_projection.','target_mean','target_std','return_')):continue
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def export_actor(checkpoint,path):
    p=read_checkpoint(checkpoint)
    payload=dict(schema_version='actor-only-v1',config=p['config'],config_sha256=p['config_sha256'],
        actor={k:v.detach().cpu() for k,v in p['actor'].items()},
        provenance=dict(checkpoint_sha256=sha256(checkpoint),progress=p['progress']))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def selection(rows_by_episode,names,seeds):
    """Same development coverage for each checkpoint; no test-based selection."""
    scores=[];expected={(n,s) for n in names for s in seeds}
    for episode,rows in sorted(rows_by_episode.items()):
        if len(rows)!=len(expected) or {(r['instance_name'],r['solver_seed']) for r in rows}!=expected:
            raise ValueError('Development coverage incomplete or duplicated')
        if any(not r.get('valid') for r in rows):raise ValueError('Invalid dev runs cannot be filtered')
        per_instance=[float(np.mean([r['pdi_horizon_midpoint'] for r in rows if r['instance_name']==n])) for n in names]
        scores.append(dict(episode=episode,mean_pdi=float(np.mean(per_instance)),
            mean_pdi_lower=float(np.mean([r['pdi_horizon_lower'] for r in rows])),
            mean_pdi_upper=float(np.mean([r['pdi_horizon_upper'] for r in rows]))))
    return min(scores,key=lambda r:(r['mean_pdi'],r['episode'])),scores


def select(root,c,run,group,seed):
    torch.set_num_threads(1);out=group_dir(run,group,seed)
    summary=read(out/'summary.json')
    if summary['protocol_sha256']!=fingerprint(c) or summary['episodes_seen']!=c['max_episodes'] or summary['complete_ppo_updates']<1:
        raise ValueError('Training not complete under selected protocol')
    stack_signature=stack(root);cutoffs=read(root/'tgppo/datasets/cutoffs.json');allrows={}
    for episode in c['milestones']:
        cp=out/('episode-{:04d}.pt'.format(episode));cp_hash=sha256(cp);jobs=[]
        for name in c['development']:
            for ss in c['solver_seeds']:
                j=bound_job(root,c,cutoffs,name,ss,seed,'eval',cp,job_id='dev-{}-s{}'.format(name,ss))
                j['sampling_seed']=action_seed(seed,'dev-{}-s{}'.format(name,ss));jobs.append(j)
        pairs=cached_jobs(jobs,out/'development'/('episode-{:04d}'.format(episode)),
             dict(checkpoint_sha256=cp_hash,stack=stack_signature,protocol=fingerprint(c)),c['workers'],c['watchdog'])
        if sha256(cp)!=cp_hash:raise RuntimeError('Evaluation mutated checkpoint')
        allrows[episode]=[r for r,_ in pairs]
    best,scores=selection(allrows,c['development'],c['solver_seeds'])
    chosen=out/('episode-{:04d}.pt'.format(best['episode']))
    write_json(out/'selection.json',dict(rule='minimum instance-equal dev mean 120s PDI; exact ties favor earlier episode',
        best=best,all_checkpoints=scores,selection_set='frozen development only',
        selection_instances=c['development'],solver_seeds=c['solver_seeds'],training_seed=seed,
        checkpoint_sha256=sha256(chosen),protocol_sha256=fingerprint(c),stack=stack_signature))
    export_actor(chosen,out/'actor_best.pt')
    export_actor(out/('episode-{:04d}.pt'.format(c['max_episodes'])),out/'actor_final.pt')


def evaluate(root,c,run,actor,instances,names,label,training_seed,native=False):
    torch.set_num_threads(1)
    if set(names)&set(c['train']+c['development']):
        raise ValueError('Held-out evaluation manifest overlaps training or selection instances')
    if not names or len(set(names))!=len(names):raise ValueError('Unique nonempty held-out manifest required')
    if not label.replace('-','').replace('_','').isalnum():raise ValueError('Unsafe dataset label')
    payload=None
    if not native:
        policy,payload=load_actor(actor)
        if hasattr(policy,'critic_network'):raise RuntimeError('Actor-only export required')
        if payload['config'].get('server_training_seed',training_seed)!=training_seed:
            raise ValueError('Evaluation seed must match the exported model training seed')
    contaminated={c['instance_hashes'][n] for n in c['train']+c['development']}
    hashes={n:sha256(instances/(n+'.mps.gz')) for n in names}
    if set(hashes.values())&contaminated:raise ValueError('Held-out dataset duplicates training/development content')
    cutoffs=read(root/'tgppo/datasets/cutoffs.json');jobs=[]
    for name in names:
        for ss in c['solver_seeds']:
            j=bound_job(root,c,cutoffs,name,ss,training_seed,'native' if native else 'eval',actor,
                job_id='test-{}-s{}'.format(name,ss),instances=instances)
            j['sampling_seed']=action_seed(training_seed,'test-{}-s{}'.format(name,ss));jobs.append(j)
    identity=dict(actor_sha256='relpscost-native' if native else sha256(actor),stack=stack(root),protocol=fingerprint(c),
                  names=names,instance_hashes=hashes)
    pairs=cached_jobs(jobs,run/'test'/label/identity['actor_sha256'][:16],identity,c['workers'],c['watchdog'])
    write_json(run/'test'/label/(identity['actor_sha256'][:16]+'-summary.json'),dict(runs=len(pairs),
        mean_pdi=float(np.mean([r['pdi_horizon_midpoint'] for r,_ in pairs])),
        valid=sum(r['valid'] for r,_ in pairs),actor_sha256=identity['actor_sha256'],stack=identity['stack'],
        method='relpscost' if native else payload['config']['server_group'],training_seed=None if native else training_seed,
        proof_count=sum(r['proof_completed'] and r['scip_solve_time']<=c['time_limit'] for r,_ in pairs),
        mean_nodes=float(np.mean([r['nnodes'] for r,_ in pairs])),
        critic_seconds=sum(r.get('callback_seconds',{}).get('critic',0.) for r,_ in pairs)))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['preflight','baselines','train','select','evaluate','export'])
    a.add_argument('--root',type=Path,default=Path(os.environ.get('RESEARCH_ROOT',DEFAULT_ROOT)))
    a.add_argument('--config',type=Path,required=True);a.add_argument('--run-root',type=Path)
    a.add_argument('--group',choices=['B0','R0','Rjoint']);a.add_argument('--seed',type=int,default=11)
    a.add_argument('--resume',action='store_true');a.add_argument('--device',default='cuda')
    a.add_argument('--checkpoint',type=Path);a.add_argument('--output',type=Path)
    a.add_argument('--native',action='store_true');a.add_argument('--instances-dir',type=Path);a.add_argument('--names-json',type=Path);a.add_argument('--label')
    args=a.parse_args();root=args.root.resolve();c=plan(root,args.config)
    run=(args.run_root or root/c['run_directory']).resolve()
    if args.command=='preflight':print(json.dumps(preflight(root,c,args.device),indent=2))
    elif args.command=='baselines':baseline(root,c,run)
    elif args.command=='train':train(root,c,run,args.group,args.seed,args.resume,args.device)
    elif args.command=='select':select(root,c,run,args.group,args.seed)
    elif args.command=='export':export_actor(args.checkpoint,args.output)
    elif args.command=='evaluate':
        if bool(args.checkpoint)==args.native:raise ValueError('Choose either --checkpoint or --native')
        evaluate(root,c,run,args.checkpoint,args.instances_dir,read(args.names_json),args.label,args.seed,args.native)

if __name__=='__main__':main()
