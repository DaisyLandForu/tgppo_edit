"""Frozen stage-5 CPU online PPO experiment; no stage-6 work is queued."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import time
import traceback
import torch
import numpy as np
from project.runtime import (config_with_protocol,seed_all,build_agent,save_checkpoint,
    load_agent,make_job,job_identity,run_jobs,import_trajectories,cpu_seconds,load_actor)
from project.utils.protocol import sha256,fingerprint,validate_baseline,write_json

ROOT=Path('/root/research')
DOC=ROOT/'docs/stage5'
RUN=ROOT/'tgppo/runs/stage5'
PROTOCOL=DOC/'execution_config.json'
GROUPS=('R0','Rprod','Rjoint')


def freeze():
    p=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())
    stats=ROOT/'tgppo/runs/stage4-calibration/final_train_only_statistics.json'
    c=dict(stage=5,date='2026-10-21',authorization='验收通过，进入下一个阶段',
        groups=list(GROUPS),training_seeds=[11,22,33],solver_seeds=[0,1,2],loss='mse',
        target_statistics=str(stats),target_statistics_sha256=sha256(stats),
        initialization='fresh Actor and original Critic per train seed; shared train-only calibrated zero readout; no offline adapter warm start',
        extra_initialization='forked torch RNG seed 1000+training_seed; does not advance public RNG',
        task_order='first lseu,stein27,misc03,pp08aCUTS; remaining 16 seeded permutation; later full 20 seeded permutations',
        task_solver_seed='cycle number modulo 3, common across groups',
        action_rng='group-neutral episode job ID plus training seed SHA256; reset after model construction',
        time_limit=120.,watchdog=180.,batch_episodes=4,min_transitions=256,microbatch_size=16,
        training_cpu_cap_seconds=7200.,training_max_episodes=50,bridge_cpu_cap_seconds=1200.,
        bridge_max_episodes=4,budget_points_seconds=[1800.,3600.,7200.],
        checkpoint_rule='latest complete valid PPO batch at or below each nominal total CPU budget; initial if no update; no development selection',
        interruption_rule='discard whole incomplete PPO iteration, retain earlier valid checkpoint and all consumed CPU/data',
        collection_guard='reserve 180 CPU seconds per next episode plus 30 CPU seconds before collection',
        online_device='cpu',affinity=[0,1,2,3],concurrent_training_runs=2,workers_per_training_run=2,
        max_total_collection_workers=4,eval_workers=4,worker_address_space_bytes=8*1024**3,
        learner_address_space_bytes=32*1024**3,critic_encoder='fully trainable',
        critic_units='raw rollout/GAE; normalized MSE residual only',
        r0_collection='remaining time only; zero unused frontier; no open-node scan',
        rprod_rjoint_collection='same pre-action full frontier capture; marginal product vs actual joint in Critic',
        evaluation='Actor only; Critic and extended search extraction forbidden; stochastic paired actions',
        final_eval_jobs=135,early_eval_jobs=90,early_eval_solver_seeds=[0],
        train=p['data']['train'],development=p['data']['development'],
        instance_hashes=p['data']['hashes'],
        native_baselines=str(ROOT/'tgppo/runs/stage3/native/baselines.json'),
        native_baselines_sha256=sha256(ROOT/'tgppo/runs/stage3/native/baselines.json'),
        early_and_bridge_shared_reserve_cpu_seconds=10*3600.,stop_after_stage5=True)
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text())!=c:
        raise ValueError('Frozen stage5 execution configuration changed')
    write_json(PROTOCOL,c)
    write_json(RUN/'execution_config.json',c)
    return c


def online_config(group,seed):
    c=json.loads(PROTOCOL.read_text());stats=json.loads(Path(c['target_statistics']).read_text())
    if sha256(c['target_statistics'])!=c['target_statistics_sha256']:raise ValueError('Statistics changed')
    params=json.loads((ROOT/'tgppo/configs/paper_hparams.json').read_text())
    params.update(online_group=group,target_mean=stats['mean'],target_std=stats['std'],
        target_statistics_sha256=c['target_statistics_sha256'],extra_initialization_seed=1000+seed,
        online_protocol='stage5-calibrated-full-critic-v1',online_protocol_sha256=sha256(PROTOCOL))
    return config_with_protocol(params)


def task_schedule(seed,c):
    first=['lseu','stein27','misc03','pp08aCUTS']
    rng=np.random.RandomState(seed)
    remaining=[n for n in c['train'] if n not in first];rng.shuffle(remaining)
    result=first+remaining
    while len(result)<c['training_max_episodes']:
        order=list(c['train']);rng.shuffle(order);result+=order
    return result[:c['training_max_episodes']]


class CPULimit(RuntimeError):pass


def limit_handler(signum,frame):
    signal.signal(signal.SIGXCPU,signal.SIG_IGN)
    raise CPULimit('Combined worker/learner CPU budget reached')


def parameter_digest(state):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(group,seed,bridge=False):
    c=json.loads(PROTOCOL.read_text());os.sched_setaffinity(0,c['affinity'])
    out=RUN/('bridge' if bridge else 'training')/('{}-s{}'.format(group,seed))
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS,(c['learner_address_space_bytes'],resource.getrlimit(resource.RLIMIT_AS)[1]))
    cap=c['bridge_cpu_cap_seconds'] if bridge else c['training_cpu_cap_seconds']
    max_ep=c['bridge_max_episodes'] if bridge else c['training_max_episodes']
    cutoffs=json.loads((ROOT/'tgppo/datasets/cutoffs.json').read_text())
    baselines=json.loads(Path(c['native_baselines']).read_text())
    if sha256(c['native_baselines'])!=c['native_baselines_sha256']:raise ValueError('Native baselines changed')
    for name in c['train']:
        if sha256(ROOT/'tgppo/datasets/train_instances'/(name+'.mps.gz'))!=c['instance_hashes'][name]:
            raise ValueError('Dataset changed: '+name)
    params=online_config(group,seed);seed_all(seed);agent=build_agent(params)
    actor_initial={k:v.detach().clone() for k,v in agent.actor_network.state_dict().items()}
    critic_initial={k:v.detach().clone() for k,v in agent.critic_network.state_dict().items()}
    write_json(out/'config.json',params)
    write_json(out/'initialization.json',dict(actor_sha256=parameter_digest(actor_initial),
        common_critic_sha256=parameter_digest({k:v for k,v in critic_initial.items() if not k.startswith(('extra.','extra_projection.'))}),
        critic_parameters=sum(p.numel() for p in agent.critic_network.parameters()),
        all_critic_parameters_trainable=all(p.requires_grad for p in agent.critic_network.parameters()),
        rng_torch_sha256=parameter_digest({'rng':torch.get_rng_state()})))
    worker_cpu=0.;completed=0;history=[];collected=[];checkpoints=[];batch=0
    def progress():return dict(stage=5,group=group,training_seed=seed,next_episode=completed,
        worker_cpu_seconds=worker_cpu,learner_cpu_seconds=cpu_seconds(),total_cpu_seconds=worker_cpu+cpu_seconds())
    save_checkpoint(out/'initial.pt',agent,params,progress())
    checkpoints.append(dict(path=str(out/'initial.pt'),updates=0,episodes=0,total_cpu_seconds=worker_cpu+cpu_seconds()))
    shutil.copyfile(out/'initial.pt',out/'latest.pt')
    schedule=task_schedule(seed,c);write_json(out/'task_schedule.json',schedule)
    stop='episode_limit';limits=resource.getrlimit(resource.RLIMIT_CPU);handler=signal.getsignal(signal.SIGXCPU)
    first_row=None
    try:
        while completed<max_ep:
            count=min(c['batch_episodes'],max_ep-completed)
            if cap-worker_cpu-cpu_seconds()<count*c['watchdog']+30.:
                stop='insufficient_budget_for_next_complete_collection_batch';break
            before=out/'rollout-policy.pt'
            save_checkpoint(before,agent,params,progress())
            jobs=[]
            for idx in range(completed,completed+count):
                name=schedule[idx];solver_seed=(idx//len(c['train']))%3
                j=make_job('e{:04d}-{}-s{}'.format(idx,name,solver_seed),ROOT/'tgppo/datasets/train_instances',
                    name,c['time_limit'],solver_seed,seed,cutoffs,'collect',before)
                j['address_space_limit_bytes']=c['worker_address_space_bytes']
                j['baseline']=baselines[fingerprint(job_identity(j))];validate_baseline(j['baseline'],job_identity(j))
                jobs.append(j)
            directory=out/('batch-{:03d}'.format(batch))
            rows=run_jobs(jobs,directory,c['workers_per_training_run'],c['watchdog'])
            # Incomplete worker CPU gets its watchdog upper bound, never zero cost.
            worker_cpu+=sum(r['worker_cpu_seconds'] if r.get('worker_cpu_complete') else c['watchdog'] for r in rows)
            completed+=len(rows);collected+=rows;write_json(out/'collection_results.json',collected)
            if any(not r.get('valid') for r in rows):stop='invalid_collection_batch_retained_no_update';break
            import_trajectories(agent,rows,directory)
            if first_row is None and agent.memory.rows():first_row=agent.memory.rows()[0]
            save_checkpoint(out/'collected.pt',agent,params,progress())
            deadline=int(cap-worker_cpu-5.)
            if deadline<=cpu_seconds():stop='cpu_budget_before_update';break
            begin=cpu_seconds();signal.signal(signal.SIGXCPU,limit_handler)
            resource.setrlimit(resource.RLIMIT_CPU,(deadline,limits[1]))
            def log(p):
                write_json(out/'learning-progress.json',dict(p,batch=batch,state='updating',
                    worker_cpu_seconds=worker_cpu,learner_cpu_seconds=cpu_seconds(),
                    learning_cpu_seconds=cpu_seconds()-begin,total_cpu_seconds=worker_cpu+cpu_seconds()))
            try:
                metrics=agent.learn(progress_callback=log,microbatch_size=c['microbatch_size'])
            except CPULimit:
                stop='budget_partial_iteration_discarded';write_json(out/'interrupted_update.json',progress());break
            except Exception:
                stop='learning_error_last_complete_checkpoint_retained'
                write_json(out/'learning_error.json',dict(error=traceback.format_exc(),**progress()));break
            finally:
                resource.setrlimit(resource.RLIMIT_CPU,limits);signal.signal(signal.SIGXCPU,handler)
            metrics.update(batch=batch,episodes_completed=completed,learning_cpu_seconds=cpu_seconds()-begin,**progress())
            metrics['actor_l2_change_from_initial']=sum((v.detach()-actor_initial[k]).double().square().sum().item()
                for k,v in agent.actor_network.state_dict().items())**.5
            metrics['critic_encoder_l2_change_from_initial']=sum((v.detach()-critic_initial[k]).double().square().sum().item()
                for k,v in agent.critic_network.state_dict().items() if k.startswith(('var_embedding','transformer','tree_embedding','tree_refinement','aggregation')))**.5
            history.append(metrics);write_json(out/'updates.json',history)
            if metrics['updated']:
                checkpoint=out/('update-{:03d}.pt'.format(agent.update_counter))
                save_checkpoint(checkpoint,agent,params,progress())
                checkpoints.append(dict(path=str(checkpoint),updates=agent.update_counter,episodes=completed,
                    total_cpu_seconds=worker_cpu+cpu_seconds(),num_samples=metrics['num_samples']))
                shutil.copyfile(checkpoint,out/'latest.pt')
            print(json.dumps(metrics,sort_keys=True),flush=True)
            batch+=1
    finally:
        resource.setrlimit(resource.RLIMIT_CPU,limits);signal.signal(signal.SIGXCPU,handler)
    write_json(out/'checkpoints.json',checkpoints)
    # Freeze budget selection using CPU metadata only, before any development solve.
    selected={}
    for budget in c['budget_points_seconds']:
        valid=[p for p in checkpoints if p['total_cpu_seconds']<=budget]
        selected[str(int(budget))]=dict(valid[-1],nominal_cpu_budget_seconds=budget)
    write_json(out/'selected_checkpoints.json',selected)
    restored,payload=load_agent(out/'latest.pt',restore_random_state=False)
    restore_audit=dict(optimizer_states_restored=bool(restored.actor_optimizer.state) and bool(restored.critic_optimizer.state),
        full_critic_trainable=all(p.requires_grad for p in restored.critic_network.parameters()),
        raw_target_mean=float(restored.critic_network.target_mean),raw_target_std=float(restored.critic_network.target_std),
        checkpoint_schema_version=payload['schema_version'],trajectory_schema_version=payload['memory']['schema_version'])
    if first_row is not None:
        state={k:first_row[k] for k in ('cands_state','mip_state','node_state','remaining_time','frontier')}
        _,raw_value,_=restored.choose_action(**state,deterministic=True)
        restore_audit['restored_raw_value_first_real_state']=raw_value
        restore_audit['initial_raw_value_first_real_state']=first_row['value']
        restore_audit['actor_only_has_no_critic']=not hasattr(load_actor(out/'latest.pt')[0],'critic_network')
    write_json(out/'restoration.json',restore_audit)
    total=worker_cpu+cpu_seconds()
    summary=dict(stage=5,group=group,training_seed=seed,bridge=bridge,stop_reason=stop,
        episodes_attempted=completed,valid_episodes=sum(bool(r.get('valid')) for r in collected),
        checkpoint_episodes=payload['progress']['next_episode'],complete_ppo_iterations=restored.update_counter,
        complete_optimizer_steps=sum(m.get('num_updates',0) for m in history),
        trained_transitions=sum(m['num_samples'] for m in history if m['updated']),
        worker_cpu_seconds=worker_cpu,learner_cpu_seconds=cpu_seconds(),total_cpu_seconds=total,
        cpu_cap_seconds=cap,within_budget=total<=cap,
        peak_learner_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        latest_checkpoint=str(out/'latest.pt'),latest_checkpoint_sha256=sha256(out/'latest.pt'))
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,sort_keys=True),flush=True)
    if restored.update_counter<1 or not summary['within_budget'] or stop.startswith(('invalid','learning_error')):
        raise RuntimeError('Run did not pass stage5 training checks; all evidence retained')


def evaluate():
    c=json.loads(PROTOCOL.read_text());os.sched_setaffinity(0,c['affinity'])
    cutoffs=json.loads((ROOT/'tgppo/datasets/cutoffs.json').read_text());jobs=[];manifest=[]
    for group in c['groups']:
        for seed in c['training_seeds']:
            directory=RUN/'training'/('{}-s{}'.format(group,seed))
            s=json.loads((directory/'summary.json').read_text())
            if not s['within_budget'] or not s['complete_ppo_iterations']:raise ValueError('Training run invalid')
            checkpoints=json.loads((directory/'selected_checkpoints.json').read_text())
            for budget in c['budget_points_seconds']:
                checkpoint=checkpoints[str(int(budget))];path=Path(checkpoint['path'])
                manifest.append(dict(group=group,training_seed=seed,**checkpoint,sha256=sha256(path)))
                for name in c['development']:
                    for solver_seed in (c['solver_seeds'] if budget==7200. else c['early_eval_solver_seeds']):
                        jobid='{}-t{}-b{}-{}-s{}'.format(group,seed,int(budget),name,solver_seed)
                        j=make_job(jobid,ROOT/'tgppo/datasets/train_instances',name,120.,solver_seed,seed,cutoffs,'eval',path)
                        # Group- and budget-neutral random stream in every deployment comparison.
                        from project.runtime import action_seed
                        j['sampling_seed']=action_seed(seed,'dev-{}-s{}'.format(name,solver_seed))
                        j['address_space_limit_bytes']=c['worker_address_space_bytes']
                        j.update(group=group,nominal_budget_seconds=budget)
                        jobs.append(j)
    write_json(RUN/'evaluation/manifest.json',manifest)
    write_json(RUN/'evaluation/jobs.json',jobs)
    rows=run_jobs(jobs,RUN/'evaluation/jobs',c['eval_workers'],c['watchdog'])
    write_json(RUN/'evaluation/summary.json',dict(jobs=len(rows),valid=sum(bool(r.get('valid')) for r in rows),
        failures=sum(not r.get('valid') for r in rows),
        critic_seconds=sum(r.get('callback_seconds',{}).get('critic',0.) for r in rows),
        worker_cpu_seconds=sum(r.get('worker_cpu_seconds') or 0. for r in rows),
        parent_cpu_seconds=cpu_seconds(),
        checkpoint_hashes_preserved=all(sha256(r['path'])==r['sha256'] for r in manifest)))
    if any(not r.get('valid') for r in rows):raise RuntimeError('Evaluation failures retained')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['freeze','train','bridge','evaluate'])
    a.add_argument('--group',choices=GROUPS);a.add_argument('--seed',type=int,default=11)
    args=a.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='evaluate':evaluate()
    elif args.command in ('train','bridge'):
        if not args.group:raise ValueError('--group is required')
        train(args.group,args.seed,args.command=='bridge')

if __name__=='__main__':main()
