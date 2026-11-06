"""Read-only late-state policy and output-head diagnostic, including legacy B0."""
from pathlib import Path
import json,pickle,math
import numpy as np
import torch
import ablation,server
from project.runtime import load_actor
from project.utils.protocol import write_json

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
RUN=ablation.ROOT/'tgppo/runs/transformer-ablation-20261027'

def sample(out):
    states=[]
    for f in sorted(out.glob('collection/*/selected_results.json'))[-5:]:
        for row in server.read(f):
            if len(states)>=48:break
            if not any(2<=n<=512 for n in row.get('candidate_counts',[])):continue
            p=next(f.parent.glob('attempt-*/'+row['job_id']+'.pkl'))
            with p.open('rb') as stream:payload=pickle.load(stream)
            valid=[r for ep in payload['episodes'] for r in ep['transitions'] if 2<=len(r['cands_state'])<=512]
            for i in sorted(set([0,len(valid)//2,len(valid)-1])) if valid else []:
                r=valid[i];states.append(dict(instance=row['instance_name'],episode=row['job_id'],**{k:np.array(r[k],copy=True) for k in ('cands_state','node_state','mip_state')}))
    return states

def probe(actor,states):
    traces={};handles=[]
    def track(name):
        def hook(module,args,output):
            x=output.detach()
            if x.ndim==3:span=float((x.max(1).values-x.min(1).values).abs().max())
            elif x.ndim==2:span=float(x.max()-x.min())
            else:span=0.
            traces.setdefault(name,[]).append(span)
        return hook
    handles.append(actor.tree_refinement.register_forward_hook(track('BiMatching')))
    for i,layer in enumerate(actor.output_layer.BranchingNet):
        handles.append(layer.register_forward_hook(track('head_layer_'+str(i))))
        for j,sub in enumerate(layer):
            if isinstance(sub,torch.nn.LayerNorm):handles.append(sub.register_forward_hook(track('head_norm_'+str(i)) ))
    rows=[]
    with torch.no_grad():
        for r in states:
            c,n,m=[torch.as_tensor(r[k]).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
            p=actor(c,n,m)[0].double();num=p.numel();u=torch.full_like(p,1/num)
            rows.append(dict(instance=r['instance'],episode=r['episode'],candidates=num,total_variation_from_uniform=float((p-u).abs().sum()/2),probability_span_relative_to_uniform=float((p.max()-p.min())*num),normalized_entropy=float(-(p*p.clamp_min(1e-38).log()).sum()/math.log(num))))
    for h in handles:h.remove()
    return dict(samples=len(rows),instances=sorted({r['instance'] for r in rows}),candidate_range=[min(r['candidates'] for r in rows),max(r['candidates'] for r in rows)],
        max_total_variation_from_uniform=max(r['total_variation_from_uniform'] for r in rows),mean_total_variation_from_uniform=float(np.mean([r['total_variation_from_uniform'] for r in rows])),
        min_normalized_entropy=min(r['normalized_entropy'] for r in rows),states_tv_above_1e_3=sum(r['total_variation_from_uniform']>1e-3 for r in rows),
        layer_candidate_span={name:dict(max=max(v),median=float(np.median(v))) for name,v in traces.items()},per_state=rows)

rows=[];shared=None
for variant in ('transformer','bimatching'):
    for seed in (11,22):
        out=RUN/variant/'training'/('B0-s'+str(seed));states=sample(out)
        if shared is None:shared=states
        policy,_=load_actor(out/'latest.pt');result=dict(variant=variant,seed=seed,source='own last 20 training episodes',**probe(policy.actor_network.eval(),states))
        rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!='per_state'}),flush=True)
        del policy
for seed in (11,22):
    path=ablation.ROOT/'tgppo/runs/server6-20261022/training'/('B0-s'+str(seed))/'actor_best.pt'
    policy,_=load_actor(path);result=dict(variant='legacy-B0',seed=seed,source='new Transformer/s11 last 20 episode states; legacy dev-selected weights',**probe(policy.actor_network.eval(),shared))
    rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!='per_state'}),flush=True)
    del policy
write_json(HERE/'late-policy-probe.json',dict(scope='Offline diagnostics on real late training states with 2..512 candidates; not held-out performance evaluation. No model changes.',rows=rows))
