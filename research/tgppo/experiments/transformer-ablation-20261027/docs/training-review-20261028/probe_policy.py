"""Read-only fixed-state policy check; no training and no solver run."""
from pathlib import Path
import json,pickle,math
import numpy as np
import torch
import ablation,server
from project.runtime import read_checkpoint,build_agent,seed_all
from project.utils.protocol import write_json

torch.set_num_threads(1)
run=ablation.ROOT/'tgppo/runs/transformer-ablation-20261027'
reference=run/'transformer/training/B0-s11'
states=[];names=[]
for rowfile in sorted(reference.glob('collection/*/selected_results.json'))[:5]:
 rows=server.read(rowfile)
 for row in rows:
  if len(states)>=40:break
  if not any(2<=n<=256 for n in row.get('candidate_counts',[])):continue
  files=list(rowfile.parent.glob('attempt-*/'+row['job_id']+'.pkl'))
  if not files:continue
  with files[0].open('rb') as f:payload=pickle.load(f)
  candidates=[r for e in payload['episodes'] for r in e['transitions'] if 2<=len(r['cands_state'])<=256]
  for idx in sorted(set([0,len(candidates)//2,len(candidates)-1])) if candidates else []:
   r=candidates[idx];states.append({k:np.array(r[k],copy=True) for k in ('cands_state','node_state','mip_state')});names.append(row['instance_name'])
 del payload
rows=[]
for variant in ('transformer','bimatching'):
 for seed in (11,22):
  out=run/variant/'training'/('B0-s'+str(seed));payload=read_checkpoint(out/'latest.pt')
  seed_all(seed);agent=build_agent(payload['config'],training=False);actor=agent.actor_network.eval()
  initial_matches=server.digest_actor(actor)==server.read(out/'initialization.json')['actor_fingerprint']
  initial={k:v.clone() for k,v in actor.state_dict().items()}
  before=[];after=[];logit_spans=[]
  with torch.no_grad():
   for state in states:
    c,n,m=[torch.as_tensor(state[k],dtype=torch.float32).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
    before.append(actor(c,n,m)[0].numpy().copy())
   actor.load_state_dict(payload['actor'])
   def logits(module,args,output):logit_spans.append(float(output.max()-output.min()))
   h=actor.output_layer.register_forward_hook(logits)
   for state in states:
    c,n,m=[torch.as_tensor(state[k],dtype=torch.float32).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
    after.append(actor(c,n,m)[0].numpy().copy())
   h.remove()
  delta=sum(float((payload['actor'][k].double()-v.double()).square().sum()) for k,v in initial.items())**.5
  differences=[float(np.abs(a-b).sum()/2) for a,b in zip(before,after)]
  rel_spans=[float((p.max()-p.min())*len(p)) for p in after]
  entropy_ratios=[float(-(p*np.log(np.maximum(p,1e-38))).sum()/math.log(len(p))) for p in after]
  row=dict(variant=variant,seed=seed,samples=len(states),instances=sorted(set(names)),candidate_count_range=[min(len(s['cands_state']) for s in states),max(len(s['cands_state']) for s in states)],initialization_matches=initial_matches,actor_parameter_delta_l2=delta,changed_actor_tensors=sum(not torch.equal(v,payload['actor'][k]) for k,v in initial.items()),actor_tensor_count=len(initial),final_policy_max_logit_span=max(logit_spans),final_policy_mean_normalized_entropy=float(np.mean(entropy_ratios)),final_policy_min_normalized_entropy=min(entropy_ratios),final_policy_max_probability_span_relative_to_uniform=max(rel_spans),initial_to_final_mean_total_variation=float(np.mean(differences)),initial_to_final_max_total_variation=max(differences),per_state=[dict(instance=name,n=len(s['cands_state']),logit_span=span,normalized_entropy=er,relative_probability_span=rs,total_variation=tv) for name,s,span,er,rs,tv in zip(names,states,logit_spans,entropy_ratios,rel_spans,differences)])
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='per_state'}),flush=True)
  del payload,agent,actor,initial
write_json(Path(__file__).with_name('policy-probe.json'),dict(scope='fixed real training states from the first 20 Transformer/s11 episodes; excludes candidates outside [2,256]; no quality evaluation',rows=rows))
