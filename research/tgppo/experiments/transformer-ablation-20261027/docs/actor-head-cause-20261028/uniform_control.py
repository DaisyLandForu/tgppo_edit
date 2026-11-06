"""Small, CPU-only node-budget diagnostic, never a formal quality evaluation.

Use the unchanged SCIP callback, features and Categorical sampler. Compare each
trained final Actor with a uniform Actor under common random numbers. Recording
is outside ActorPolicy's forward timer and is identical for every policy.
"""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from torch import nn
import server
from project.runtime import seed_all, action_seed, load_actor
from project.agent.agent import ActorPolicy
from project.environment import Environment
from project.utils.settings import state_dims
from project.utils.protocol import write_json, sha256
from diagnose_cause import HERE, RUN


class UniformActor(nn.Module):
    def forward(self,cands,node,mip,mask=None):
        logits=cands.new_zeros(cands.shape[:2])
        if mask is not None:logits=logits.masked_fill(mask,float('-inf'))
        return logits.softmax(-1)


class TracedPolicy(ActorPolicy):
    def __init__(self,actor):
        super().__init__(actor)
        self.trace=[]
    def act(self,cands_state,mip_state,node_state,**kwargs):
        action,logprob=super().act(cands_state,mip_state,node_state,**kwargs)
        self.trace.append(dict(candidates=len(cands_state),action=action,
            candidate_features_sha256=hashlib.sha256(np.asarray(cands_state,dtype=np.float32).tobytes()).hexdigest()))
        return action,logprob


def run(args):
    torch.set_num_threads(1)
    directory=HERE/'uniform-control'/('attempt-'+uuid.uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    c=server.plan(server.DEFAULT_ROOT,PathConfig)
    cutoffs=server.read(server.DEFAULT_ROOT/'tgppo/datasets/cutoffs.json')
    # Previously used development instances only; no held-out test selection.
    names=('vpm2','dcmulti');policies=('uniform','transformer-11','bimatching-11','transformer-22','bimatching-22')
    checkpoints={v:RUN/v.split('-')[0]/'training'/('B0-s'+v.split('-')[1])/'latest.pt' for v in policies if v!='uniform'}
    protected={str(p):sha256(p) for p in checkpoints.values()};rows=[]
    for ii,name in enumerate(names):
        for ss in (0,1):
            # Rotate order across cases to reduce systematic warm-up/drift bias.
            offset=(ii*2+ss)%len(policies);order=policies[offset:]+policies[:offset]
            for label in order:
                actor=UniformActor() if label=='uniform' else load_actor(checkpoints[label])[0].actor_network
                policy=TracedPolicy(actor)
                job=server.bound_job(server.DEFAULT_ROOT,c,cutoffs,name,ss,11,'eval',job_id='control-'+name+'-s'+str(ss))
                job['limits'].update(time_limit=args.seconds,node_limit=args.nodes)
                seed_all(action_seed(11,job['job_id']))
                env=Environment('cpu',policy,state_dims,job['limits'],job['params'],ss,mode='eval')
                env.reset(job['instance'],job['cutoff'],None,job['cutoff_provenance'])
                valid,row,_=env.run_episode()
                # This old binding maps NODELIMIT to "unknown", while the frozen
                # production protocol accepts proof/time limits only. Preserve
                # raw valid/status, explicitly classify the bounded diagnostic;
                # do not alter solver_stats or claim a solved finite task.
                budget_reached=(row.get('status')=='unknown' and row.get('nnodes')==args.nodes
                    and not row.get('error') and row.get('scip_solve_time',args.seconds)<args.seconds)
                row.update(label=label,instance_name=name,solver_seed=ss,action_trace=policy.trace,
                    trained=label!='uniform',quality_comparison=False,diagnostic_node_budget_reached=budget_reached,
                    diagnostic_termination_accepted=valid or budget_reached)
                write_json(directory/(name+'-s'+str(ss)+'-'+label+'.json'),row)
                print(json.dumps({k:row.get(k) for k in ('label','instance_name','solver_seed','valid','status','nnodes','branch_count','end_to_end_wall_seconds','inference_mean_ms')}),flush=True)
                rows.append(row)
                if not row['diagnostic_termination_accepted'] or row['fallback_action_count'] or row['callback_seconds']['critic']:
                    raise RuntimeError('Online diagnostic failed; raw attempt retained: '+str(directory))
    comparisons=[]
    for name in names:
        for ss in (0,1):
            case=[r for r in rows if r['instance_name']==name and r['solver_seed']==ss]
            base=next(r for r in case if r['label']=='uniform')
            for row in case:
                if row is base:continue
                a=row['action_trace'];b=base['action_trace'];prefix=0
                for x,y in zip(a,b):
                    if (x['candidates'],x['action'])!=(y['candidates'],y['action']):break
                    prefix+=1
                comparisons.append(dict(instance=name,solver_seed=ss,label=row['label'],
                    matched_action_prefix=prefix,model_decisions=len(a),uniform_decisions=len(b),
                    actions_identical=a==b,
                    action_indices_and_counts_identical=prefix==len(a)==len(b),
                    same_final_nodes=row['nnodes']==base['nnodes'],same_final_gap=row['gap']==base['gap'],
                    model_status=row['status'],uniform_status=base['status'],
                    model_e2e_seconds=row['end_to_end_wall_seconds'],uniform_e2e_seconds=base['end_to_end_wall_seconds']))
    checks=dict(checkpoints_unchanged=all(sha256(p)==protected[str(p)] for p in checkpoints.values()),
        actor_only=all(r['callback_seconds']['critic']==0 for r in rows),
        accepted_diagnostic_terminations=all(r['diagnostic_termination_accepted'] and not r['fallback_action_count'] for r in rows),
        all_reached_node_budget_or_proof=all(r['status'] not in ('timelimit','error') for r in rows))
    result=dict(checks=checks,comparisons=comparisons,directory=str(directory),stack=server.stack(server.DEFAULT_ROOT),
        resources=server.resource_limits(),nodes=args.nodes,seconds=args.seconds,rows=rows,protected_checkpoint_sha256=protected,
        scope='20 local CPU engineering runs; two previously used dev instances and two solver seeds; fixed node budget, no training, no formal test-quality claim.')
    write_json(HERE/'uniform-control.json',result)
    if not all(checks.values()):raise RuntimeError('Diagnostic incomplete; inspect limits/raw rows')


PathConfig=Path(__file__).resolve().parents[2]/'configs/transformer.json'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nodes',type=int,default=200);p.add_argument('--seconds',type=float,default=30.)
    run(p.parse_args())
