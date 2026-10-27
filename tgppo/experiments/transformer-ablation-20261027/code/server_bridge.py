"""Target-machine GPU bridge. Does not start the six full training jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import uuid
import numpy as np
import torch
from unittest.mock import patch
import server
from project.runtime import load_agent,load_actor,read_checkpoint
from project.agent.agent import Agent
from project.utils.protocol import write_json,fingerprint


def bridge(root,c,run):
    info=server.preflight(root,c,'cuda')  # Enforces expected V100 before engineering changes.
    probe=copy.deepcopy(c)
    probe.update(engineering_mode=True,max_episodes=4,milestones=[4],training_seeds=[11],
        train=['lseu','stein27','misc03','pp08aCUTS'],development=['vpm2'])
    probe.pop('model_overrides',None)  # Full production 256/5/8 model, FP32.
    directory=run/'gpu-bridge'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    write_json(directory/'probe_protocol.json',probe)
    server.baseline(root,probe,directory)
    rows=[];checks={};initializations=[]
    for group in c['groups']:
        normal_learn=Agent.learn
        def fail_after_first_step(agent,progress_callback=None,**kwargs):
            def crash(progress):
                if progress_callback:progress_callback(progress)
                raise server.Interrupted('GPU bridge injected interruption after first optimizer step')
            return normal_learn(agent,progress_callback=crash,**kwargs)
        did_interrupt=False
        with patch.object(Agent,'learn',fail_after_first_step):
            try:server.train(root,probe,directory,group,11,False,'cuda')
            except server.Interrupted:did_interrupt=True
        if not did_interrupt:raise RuntimeError('GPU bridge needs enough real decisions for an interrupted update')
        out=server.group_dir(directory,group,11)
        reference,pending=load_agent(out/'latest.pt')
        checks[group+'_interrupted_state_uncommitted']=(pending['progress']['phase']=='collected' and reference.policy_version==0)
        reference.learn(microbatch_size=probe['microbatch_size'])
        expected_actor=server.digest_actor(reference.actor_network)
        expected_critic=server.digest_actor(reference.critic_network)
        expected_optimizers=copy.deepcopy((reference.actor_optimizer.state_dict(),reference.critic_optimizer.state_dict()))
        expected_rng=torch.cuda.get_rng_state().clone()
        del reference
        torch.cuda.empty_cache()
        trajectories={str(p):p.stat().st_mtime_ns for p in out.glob('collection/**/*.pkl')}
        state=server.train(root,probe,directory,group,11,True,'cuda')
        out=server.group_dir(directory,group,11);agent,p=load_agent(out/'latest.pt')
        checks[group+'_resume_actor_matches']=expected_actor==server.digest_actor(agent.actor_network)
        checks[group+'_resume_critic_matches']=expected_critic==server.digest_actor(agent.critic_network)
        checks[group+'_resume_cuda_rng_matches']=torch.equal(expected_rng,torch.cuda.get_rng_state())
        checks[group+'_resume_optimizers_match']=same_state(expected_optimizers,
            (agent.actor_optimizer.state_dict(),agent.critic_optimizer.state_dict()))
        checks[group+'_resume_reuses_collection']=trajectories=={str(p):p.stat().st_mtime_ns for p in out.glob('collection/**/*.pkl')}
        first_torch=torch.rand(4,device='cuda').cpu()
        restored,_=load_agent(out/'latest.pt');next_torch=torch.rand(4,device='cuda').cpu()
        checks[group+'_cuda_rng_recovery']=torch.equal(first_torch,next_torch)
        checks[group+'_full_cuda_critic']=all(t.requires_grad and t.device.type=='cuda' for t in restored.critic_network.parameters())
        checks[group+'_optimizers_recovered']=bool(restored.actor_optimizer.state) and bool(restored.critic_optimizer.state)
        checks[group+'_real_update']=restored.update_counter==1 and state['trained_episodes']==4
        checks[group+'_ratio_cpu_rollout_vs_cuda_update']=state['history'][0]['initial_ratio_max_error']<1e-4
        init=server.read(out/'initialization.json')
        initializations.append(init)
        checks[group+'_actor_changed']=init['actor_fingerprint']!=server.digest_actor(restored.actor_network)
        checks[group+'_critic_changed']=init['common_critic_fingerprint']!=server.digest_actor(restored.critic_network,exclude_extra=True)
        policy,_=load_actor(out/'latest.pt')
        checks[group+'_actor_only_cpu']=not hasattr(policy,'critic_network') and all(p.device.type=='cpu' for p in policy.actor_network.parameters())
        worker_rows=[server.read(p) for p in out.glob('collection/**/e*.json')]
        checks[group+'_workers_cpu_only']=len(worker_rows)==4 and all(not r['cuda_initialized'] for r in worker_rows)
        if group!='B0':
            checks[group+'_raw_unit_decode']=float(restored.critic_network.target_mean)==torch.tensor(p['config']['target_mean'],dtype=torch.float32).item()
        rows.append(dict(group=group,gpu=info['gpu'],metrics=state['history'],source=str(out)))
        del agent,restored,policy,expected_optimizers
        torch.cuda.empty_cache()
    checks['paired_initial_actor']=len({r['actor_fingerprint'] for r in initializations})==1
    checks['paired_initial_cpu_rng']=len({r['public_torch_rng_sha256'] for r in initializations})==1
    checks['paired_initial_cuda_rng']=len({r['public_cuda_rng_sha256'] for r in initializations})==1
    evidence=dict(passed=all(checks.values()),parent_protocol_sha256=fingerprint(c),stack=info['stack'],
        gpu=info['gpu'],checks=checks,rows=rows,scope='GPU bridge only, not full training or performance validation')
    write_json(run/'gpu-bridge/verification.json',evidence)
    print(json.dumps(evidence,indent=2))
    if not evidence['passed']:raise RuntimeError('Target GPU bridge failed; full training forbidden')


def same_state(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(same_state(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(same_state(a,b) for a,b in zip(left,right))
    return left==right


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path(os.environ.get('RESEARCH_ROOT',server.DEFAULT_ROOT)))
    a.add_argument('--config',type=Path,required=True);a.add_argument('--run-root',type=Path,required=True)
    args=a.parse_args();bridge(args.root,server.plan(args.root,args.config),args.run_root)
