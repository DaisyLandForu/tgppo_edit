"""Frozen Actor diagnostics on saved TRAIN observations, CPU only, no optimizer.

No SCIP imports. Observations are selected before model evaluation using the
first 20 B0-s11 training episodes, first/middle/last decisions (deduplicated).
All checkpoints see exactly the same untruncated candidate sets, no padding
except in the explicit padding invariance check. No init weights are recreated.
"""
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
RUN=ROOT/'tgppo/runs/server6-20261022'
sys.path.insert(0,str(ROOT/'tgppo/experiments/server-20261022/code'))
from project.policy.actor import Actor

torch.set_num_threads(1)
torch.manual_seed(20261030)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stats(xs):
    return dict(zip(['min','p50','p95','max'],map(float,np.quantile(xs,[0,.5,.95,1]))))


def inputs(r):
    return [torch.as_tensor(r[k],dtype=torch.float32)[None] for k in ['cands_state','node_state','mip_state']]


def spread(t):
    t=t.detach().double()
    return float((t.max(dim=1).values-t.min(dim=1).values).abs().mean())


def normprob(p):
    p=np.asarray(p,dtype=np.float64);return p/p.sum()


def kl(p,q):
    return float(np.sum(p*np.log(p/q)))


def rankcorr(a,b):
    # Exact tied ranks. Near ties must be interpreted against logit spread.
    def rank(x):
        u,inv,c=np.unique(x,return_inverse=True,return_counts=True)
        ranks=np.cumsum(c)-.5*(c+1)
        return ranks[inv]
    a,b=rank(a),rank(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None


def main():
    start=time.perf_counter()
    observations=[]; manifest=[]
    ps=sorted((RUN/'training/B0-s11/collection').rglob('*.pkl'))[:20]
    train=set(json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']['train'])
    for p in ps:
        d=pickle.loads(p.read_bytes());meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['instance_name'] in train
        for e in d['episodes']:
            rows=e['transitions']
            for i in sorted(set([0,len(rows)//2,len(rows)-1])) if rows else []:
                r=rows[i]; key=f'{p.stem}:t{i}'
                assert all(np.isfinite(r[k]).all() for k in ['cands_state','mip_state','node_state'])
                observations.append((key,meta['instance_name'],r))
                manifest.append(dict(observation=key,instance=meta['instance_name'],trajectory=str(p.relative_to(ROOT)),
                                     trajectory_sha256=sha(p),transition_index=i,candidates=len(r['cands_state']),
                                     normalized_depth_feature=float(r['node_state'][0]),split='train'))
    write('fixed_observation_manifest.csv',manifest)
    allrows=[]; comparisons=[]; grads=[]; activations=[]; invariance=[]; sensitivities=[]; weights=[]
    for model in ['B0-s11','B0-s22','R0-s11','R0-s22','Rjoint-s11','Rjoint-s22']:
        outputs={}
        for episode in [100,500]:
            p=RUN/'training'/model/f'episode-{episode:04d}.pt'
            cp=torch.load(p,map_location='cpu');c=cp['config']
            a=Actor(**c['state_dims'],**{k:c[k] for k in ['hidden_dim','num_heads','num_layers','dropout']})
            a.load_state_dict(cp['actor']);a.eval()
            before={k:v.clone() for k,v in a.state_dict().items()}
            weights.append(dict(model=model,episode=episode,path=str(p.relative_to(ROOT)),sha256=sha(p),
                                actor_parameters=sum(x.numel() for x in a.parameters()),policy_version=cp['policy_version']))
            out=[]
            for idx,(key,instance,r) in enumerate(observations):
                x,n,m=inputs(r); L=x.shape[1]; seen={}; handles=[]
                def hook(name):
                    def fn(mod,inp,value):
                        t=value.transpose(0,1) if name=='transformer' else value
                        seen[name]=t.detach().clone()
                        if name=='head6_ln':seen['head6_ln_input']=inp[0].detach().clone()
                    return fn
                for name,mod in [('var_embedding',a.var_embedding),('transformer',a.transformer),('bimatching',a.tree_refinement),
                                 ('logits',a.output_layer)]+[(f'head{i}',layer) for i,layer in enumerate(a.output_layer.BranchingNet)]+[
                                     ('head6_ln',a.output_layer.BranchingNet[6][1])]:
                    handles.append(mod.register_forward_hook(hook(name)))
                with torch.no_grad(): probs=a(x,n,m)[0]
                for h in handles:h.remove()
                p64=normprob(probs.numpy()); logits=seen['logits'][0].numpy(); entropy=-float(np.sum(p64*np.log(p64)))
                ls=np.sort(logits)
                allrows.append(dict(model=model,episode=episode,observation=key,instance=instance,L=L,
                                    entropy=entropy,normalized_entropy=entropy/math.log(L) if L>1 else None,
                                    kl_uniform=math.log(L)-entropy,logit_spread=float(np.ptp(logits)),
                                    top2_margin=float(ls[-1]-ls[-2]) if L>1 else None,
                                    max_abs_probability_minus_uniform=float(np.max(np.abs(p64-1/L))),top1=int(np.argmax(p64))))
                out.append((p64,logits))
                if model.startswith('B0'):
                    for name,t in seen.items():
                        if name in ['logits']:continue
                        activations.append(dict(model=model,episode=episode,observation=key,L=L,module=name,
                                                mean_feature_candidate_range=spread(t)))
                    if L>1:
                        ln=seen['head6_ln_input'][0]
                        diff=ln[:,0]-ln[:,1]
                        activations.append(dict(model=model,episode=episode,observation=key,L=L,module='head6_pre_ln_signed_channel_difference',
                                                mean_feature_candidate_range=float(diff.abs().min())))
                if L>1 and idx%3==1 and model.startswith('B0'):
                    rng=np.random.RandomState(20261030+idx)
                    perm=torch.tensor(rng.permutation(L))
                    with torch.no_grad():
                        pp=a(x[:,perm],n,m)[0].numpy()[np.argsort(perm.numpy())]
                        xp=torch.cat([x,torch.zeros((1,7,x.shape[2]))],dim=1)
                        mask=torch.tensor([[False]*L+[True]*7])
                        pad=a(xp,n,m,mask)[0].numpy()
                        # Permute one candidate feature column; a sensitivity probe,
                        # not a claim that the perturbed state is solver-reachable.
                        altered=x.clone();altered[:,:,0]=x[:,perm,0]
                        altered_prob=normprob(a(altered,n,m)[0].numpy())
                    invariance.append(dict(model=model,episode=episode,observation=key,L=L,
                                           permute_inverse_max_abs_error=float(np.max(np.abs(pp-probs.numpy()))),
                                           padding_valid_max_abs_error=float(np.max(np.abs(pad[:L]-probs.numpy()))),
                                           padded_probability_sum=float(pad[L:].sum())))
                    sensitivities.append(dict(model=model,episode=episode,observation=key,L=L,intervention='permute_candidate_feature_column_0',
                                              tv_distance=float(.5*np.abs(altered_prob-p64).sum())))
                if L>1 and idx%6==1 and model.startswith('B0'):
                    a.zero_grad(set_to_none=True)
                    xx=x.clone().requires_grad_(True)
                    dist=torch.distributions.Categorical(a(xx,n,m))
                    # Chosen-action score gradient, no optimizer and no weighting
                    # by guessed advantages; this is not a historical PPO gradient.
                    (-dist.log_prob(torch.tensor([r['action']]))).backward()
                    for name,module in [('var_embedding',a.var_embedding),('transformer',a.transformer),('bimatching',a.tree_refinement),('treegate',a.output_layer)]:
                        gs=[q.grad for q in module.parameters() if q.grad is not None]
                        grads.append(dict(model=model,episode=episode,observation=key,L=L,module=name,
                                          l2_norm=math.sqrt(sum(float(g.double().square().sum()) for g in gs)),
                                          max_abs=max(float(g.abs().max()) for g in gs),
                                          parameter_count=sum(q.numel() for q in module.parameters())))
                    sensitivities.append(dict(model=model,episode=episode,observation=key,L=L,intervention='chosen_action_score_input_gradient_l2',
                                              tv_distance=float(xx.grad.norm())))
            assert all(torch.equal(v,before[k]) for k,v in a.state_dict().items())
            outputs[episode]=out
            print('done',model,episode,len(out),flush=True)
        for (key,instance,r),(p,lp),(q,lq) in zip(observations,outputs[100],outputs[500]):
            L=len(p);mid=(p+q)/2;k=min(5,L)
            comparisons.append(dict(model=model,observation=key,instance=instance,L=L,
                                    kl_100_to_500=kl(p,q),js=.5*(kl(p,mid)+kl(q,mid)),
                                    tv_distance=float(.5*np.abs(p-q).sum()),
                                    top1_changed=int(p.argmax()!=q.argmax()),
                                    top5_overlap=len(set(np.argsort(p)[-k:])&set(np.argsort(q)[-k:]))/k,
                                    logit_rank_spearman=rankcorr(lp,lq),
                                    max_logit_spread=max(float(np.ptp(lp)),float(np.ptp(lq))),
                                    all_candidate_ratio_min=float((q/p).min()),all_candidate_ratio_max=float((q/p).max())))
    write('actor_checkpoint_metrics.csv',allrows);write('actor_checkpoint_comparison.csv',comparisons)
    write('actor_gradient_metrics.csv',grads);write('actor_activation_spread.csv',activations)
    write('actor_invariance_checks.csv',invariance);write('actor_sensitivity_checks.csv',sensitivities)
    write('diagnostic_checkpoint_hashes.csv',weights)
    # Exact historical old-policy check: ep100 weights are the policy for ep100-103.
    ratio=[]
    for model in ['B0-s11','B0-s22']:
        cp=torch.load(RUN/'training'/model/'episode-0100.pt',map_location='cpu');c=cp['config']
        a=Actor(**c['state_dims'],**{k:c[k] for k in ['hidden_dim','num_heads','num_layers','dropout']});a.load_state_dict(cp['actor']);a.eval()
        for p in sorted((RUN/'training'/model/'collection/e00100').rglob('*.pkl')):
            e=pickle.loads(p.read_bytes())['episodes'][0]
            assert e['policy_version']==cp['policy_version']
            for i in sorted(set([0,len(e['transitions'])//2,len(e['transitions'])-1])) if e['transitions'] else []:
                r=e['transitions'][i]
                with torch.no_grad():dist=torch.distributions.Categorical(a(*inputs(r)))
                delta=float(dist.log_prob(torch.tensor([r['action']])))-r['log_prob']
                ratio.append(dict(model=model,episode_id=e['episode_id'],transition=i,L=len(r['cands_state']),
                                  ratio=math.exp(delta),logprob_difference=delta))
    write('old_policy_recompute.csv',ratio)
    summary=dict(observations=len(observations),instances=len({n for _,n,_ in observations}),
                 observations_L_gt_1=sum(len(r['cands_state'])>1 for _,_,r in observations),
                 candidate_counts=stats([len(r['cands_state']) for _,_,r in observations]),
                 init_checkpoint='N/A: no saved init weights found; initialization.json contains hashes only',
                 checkpoints=weights,per_model=[])
    for model in ['B0-s11','B0-s22','R0-s11','R0-s22','Rjoint-s11','Rjoint-s22']:
        rs=[r for r in allrows if r['model']==model and r['L']>1]
        cs=[r for r in comparisons if r['model']==model and r['L']>1]
        summary['per_model'].append(dict(model=model,logit_spread=stats([r['logit_spread'] for r in rs]),
                                         normalized_entropy=stats([r['normalized_entropy'] for r in rs]),
                                         tv_100_500=stats([r['tv_distance'] for r in cs]),
                                         kl_100_500=stats([max(0,r['kl_100_to_500']) for r in cs]),
                                         top1_changed_fraction=float(np.mean([r['top1_changed'] for r in cs]))))
    summary.update(permutation_max_abs_error=max(r['permute_inverse_max_abs_error'] for r in invariance),
                   padding_max_abs_error=max(r['padding_valid_max_abs_error'] for r in invariance),
                   max_padded_probability=max(r['padded_probability_sum'] for r in invariance),
                   old_policy_recompute_ratio=stats([r['ratio'] for r in ratio]),
                   cuda_initialized=torch.cuda.is_initialized(),torch_version=torch.__version__,
                   elapsed_seconds=time.perf_counter()-start)
    save('actor_offline_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
