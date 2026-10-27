"""Target-machine GPU bridge. Does not start the six full training jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import uuid
import numpy as np
import torch
import server
from project.runtime import load_agent,load_actor,read_checkpoint
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
    rows=[];checks={}
    for group in c['groups']:
        state=server.train(root,probe,directory,group,11,False,'cuda')
        out=server.group_dir(directory,group,11);agent,p=load_agent(out/'latest.pt')
        first_torch=torch.rand(4,device='cuda').cpu()
        restored,_=load_agent(out/'latest.pt');next_torch=torch.rand(4,device='cuda').cpu()
        checks[group+'_cuda_rng_recovery']=torch.equal(first_torch,next_torch)
        checks[group+'_full_cuda_critic']=all(t.requires_grad and t.device.type=='cuda' for t in restored.critic_network.parameters())
        checks[group+'_optimizers_recovered']=bool(restored.actor_optimizer.state) and bool(restored.critic_optimizer.state)
        checks[group+'_real_update']=restored.update_counter==1 and state['trained_episodes']==4
        checks[group+'_ratio_cpu_rollout_vs_cuda_update']=state['history'][0]['initial_ratio_max_error']<1e-4
        init=server.read(out/'initialization.json')
        checks[group+'_actor_changed']=init['actor_fingerprint']!=server.digest_actor(restored.actor_network)
        checks[group+'_critic_changed']=init['common_critic_fingerprint']!=server.digest_actor(restored.critic_network,exclude_extra=True)
        policy,_=load_actor(out/'latest.pt')
        checks[group+'_actor_only_cpu']=not hasattr(policy,'critic_network') and all(p.device.type=='cpu' for p in policy.actor_network.parameters())
        worker_rows=[server.read(p) for p in out.glob('collection/**/e*.json')]
        checks[group+'_workers_cpu_only']=len(worker_rows)==4 and all(not r['cuda_initialized'] for r in worker_rows)
        if group!='B0':
            checks[group+'_raw_unit_decode']=float(restored.critic_network.target_mean)==torch.tensor(p['config']['target_mean'],dtype=torch.float32).item()
        rows.append(dict(group=group,gpu=info['gpu'],metrics=state['history'],source=str(out)))
        del agent,restored,policy
        torch.cuda.empty_cache()
    evidence=dict(passed=all(checks.values()),parent_protocol_sha256=fingerprint(c),stack=info['stack'],
        gpu=info['gpu'],checks=checks,rows=rows,scope='GPU bridge only, not full training or performance validation')
    write_json(run/'gpu-bridge/verification.json',evidence)
    print(json.dumps(evidence,indent=2))
    if not evidence['passed']:raise RuntimeError('Target GPU bridge failed; full training forbidden')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path(os.environ.get('RESEARCH_ROOT',server.DEFAULT_ROOT)))
    a.add_argument('--config',type=Path,required=True);a.add_argument('--run-root',type=Path,required=True)
    args=a.parse_args();bridge(args.root,server.plan(args.root,args.config),args.run_root)
