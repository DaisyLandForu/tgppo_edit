"""Frozen Actor local intervention; no training, solver, or CUDA initialization."""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from collections import Counter,defaultdict

START=time.perf_counter()
import numpy as np
import torch
from torch import nn

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
RUN=ROOT/'tgppo/runs/server6-20261022'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261102)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantiles(values):
    values=[float(x) for x in values if x is not None]
    return dict(zip(['min','p50','p95','max'],map(float,np.quantile(values,[0,.5,.95,1])))) if values else {}


def probability(p):
    p=p.detach().cpu().numpy().astype(np.float64).reshape(-1)
    assert np.isfinite(p).all() and (p>0).all()
    return p/p.sum()


def kl(p,q):return max(0.,float(np.sum(p*np.log(p/q))))


def inputs(r,requires_grad=False):
    x=torch.as_tensor(r['cands_state'],dtype=torch.float32)[None].clone()
    if requires_grad:x.requires_grad_(True)
    return x,torch.as_tensor(r['node_state'],dtype=torch.float32)[None],torch.as_tensor(r['mip_state'],dtype=torch.float32)[None]


def candidate_range(t):
    a=t.detach().double()
    return float((a.max(dim=1).values-a.min(dim=1).values).abs().mean())


class AffineOnly(nn.Module):
    """Drop normalization only; preserve learned affine values and state keys."""
    def __init__(self,layer):
        super().__init__();self.weight=layer.weight;self.bias=layer.bias
    def forward(self,x):return x*self.weight+self.bias


def prepare_copy():
    dest=OUT/'model_copy';dest.mkdir(exist_ok=True)
    (dest/'__init__.py').write_text('')
    source=ROOT/'tgppo/experiments/server-20261022/code/project/policy'
    for name in ['actor.py','modules.py']:shutil.copy2(source/name,dest/name)
    sys.path.insert(0,str(OUT))
    from model_copy.actor import Actor
    return Actor


def select_observations():
    split=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']
    old=list(csv.DictReader((ROOT/'docs/diagnostics/tgppo_20261030/fixed_observation_manifest.csv').open()))
    previous={(r['trajectory'],int(r['transition_index'])) for r in old}
    observations=[];manifest=[];coverage=[];source_hashes={}
    available=defaultdict(list)
    for model in ['B0-s11','B0-s22']:
        for p in sorted((RUN/'training'/model/'collection').glob('e*/attempt-*/e*.json')):
            if int(p.stem[1:6])<200:continue
            d=json.loads(p.read_text())
            if d.get('valid') and d.get('num_transitions',0)>0:
                available[model,d['instance_name']].append((p,d))
    for instance in sorted(split['train']):
        for model in ['B0-s11','B0-s22']:
            matches=available[model,instance]
            chosen=matches[0] if matches else None
            if chosen is None:
                coverage.append(dict(instance=instance,source_model=model,states=0,reason='no nonempty trajectory at episode>=200'))
                continue
            p,d=chosen;q=p.with_suffix('.pkl');raw=q.read_bytes();source_hashes[str(q.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
            payload=pickle.loads(raw);assert len(payload['episodes'])==1
            rows=payload['episodes'][0]['transitions'];assert len(rows)==d['num_transitions']
            indices=sorted(set(np.linspace(0,len(rows)-1,min(5,len(rows)),dtype=int).tolist()))
            coverage.append(dict(instance=instance,source_model=model,states=len(indices),reason='selected by metadata and row position only'))
            for i in indices:
                r=rows[i];key=f'{model}/{q.stem}:t{i}'
                assert (str(q.relative_to(ROOT)),i) not in previous
                assert all(np.isfinite(r[k]).all() for k in ['cands_state','node_state','mip_state'])
                assert 0<=r['action']<len(r['cands_state'])
                observations.append((key,instance,r))
                manifest.append(dict(observation=key,instance=instance,source_model=model,
                                     trajectory=str(q.relative_to(ROOT)),trajectory_sha256=source_hashes[str(q.relative_to(ROOT))],
                                     transition_index=i,trajectory_transitions=len(rows),candidate_count=len(r['cands_state']),
                                     normalized_depth_feature=float(r['node_state'][0]),split='train',overlap_with_previous_pool=False))
    assert len(observations)<=200
    assert set(x[1] for x in observations).isdisjoint(split['development'])
    write('observation_manifest.csv',manifest);write('observation_coverage.csv',coverage)
    with (OUT/'observations.pkl').open('wb') as f:pickle.dump(observations,f,pickle.HIGHEST_PROTOCOL)
    return observations,manifest,source_hashes


def snapshot_sources():
    code=ROOT/'tgppo/experiments/server-20261022/code'
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(code.rglob('*.py'))}
    sources['docs/stage1/protocol_draft.json']=sha(ROOT/'docs/stage1/protocol_draft.json')
    return sources


def main():
    before_sources=snapshot_sources();save('runtime_sources_before.json',before_sources)
    Actor=prepare_copy()
    observations,manifest,trajectory_hashes=select_observations()
    check_ids={}
    for i,(key,name,row) in enumerate(observations):
        if len(row['cands_state'])>1 and name not in check_ids:check_ids[name]=i
    plan=dict(max_observations=200,observations=len(observations),instances=len({x[1] for x in observations}),
              checkpoint_models=['B0-s11','B0-s22'],episodes=[100,500],variants=['original','ln2_affine_only'],
              material_nonuniform_tv_threshold=.01,invariance_atol=1e-6,check_indices=sorted(check_ids.values()),
              manifest_sha256=sha(OUT/'observation_manifest.csv'),no_previous_pool_overlap=True)
    save('frozen_plan.json',plan)
    rows=[];paired=[];grads=[];invariants=[];sensitivity=[];checkpoint_info=[];summary_by=[]
    for model in plan['checkpoint_models']:
        for episode in plan['episodes']:
            assert time.perf_counter()-START<360,'Internal budget guard: reserve reporting time'
            cp_path=RUN/'training'/model/f'episode-{episode:04d}.pt'
            cp_hash=sha(cp_path);cp=torch.load(cp_path,map_location='cpu');c=cp['config']
            a=Actor(**c['state_dims'],**{k:c[k] for k in ['hidden_dim','num_heads','num_layers','dropout']})
            a.load_state_dict(cp['actor']);a.eval();del cp
            b=copy.deepcopy(a);layer=b.output_layer.BranchingNet[6][1]
            assert isinstance(layer,nn.LayerNorm) and tuple(layer.normalized_shape)==(2,)
            eps=layer.eps;b.output_layer.BranchingNet[6][1]=AffineOnly(layer);b.eval()
            before={k:v.detach().clone() for k,v in a.state_dict().items()}
            assert all(torch.equal(v,b.state_dict()[k]) for k,v in before.items())
            assert not ({id(p) for p in a.parameters()}&{id(p) for p in b.parameters()})
            result={};local_rows=[]
            for variant,net in [('original',a),('ln2_affine_only',b)]:
                dist_out=[]
                for idx,(key,instance,r) in enumerate(observations):
                    assert time.perf_counter()-START<360,'Internal budget guard'
                    x,n,m=inputs(r);L=x.shape[1];seen={};handles=[]
                    def capture(name):
                        def hook(mod,inp,value):
                            seen[name]=value.detach().clone()
                            if name=='ln2':seen['ln2_input']=inp[0].detach().clone()
                        return hook
                    for name,mod in [('bimatching',net.tree_refinement),('head4',net.output_layer.BranchingNet[5]),
                                     ('ln2',net.output_layer.BranchingNet[6][1]),('head2',net.output_layer.BranchingNet[6]),
                                     ('logits',net.output_layer)]:
                        handles.append(mod.register_forward_hook(capture(name)))
                    with torch.no_grad():prob=net(x,n,m)[0]
                    for h in handles:h.remove()
                    p=probability(prob);logits=seen['logits'][0].double().numpy();uniform=np.full(L,1/L)
                    k=kl(p,uniform);diff=seen['ln2_input'][0,:,0]-seen['ln2_input'][0,:,1]
                    amp=diff.abs()/torch.sqrt(diff.square()+4*eps)
                    d=dict(model=model,episode=episode,variant=variant,observation=key,instance=instance,L=L,
                           normalized_entropy=(math.log(L)-k)/math.log(L) if L>1 else None,
                           kl_to_uniform=k,tv_to_uniform=float(.5*np.abs(p-uniform).sum()),
                           logit_spread=float(np.ptp(logits)),mean_logit=float(logits.mean()),
                           top2_margin=float(np.sort(logits)[-1]-np.sort(logits)[-2]) if L>1 else None,
                           bimatching_candidate_range=candidate_range(seen['bimatching']),
                           head4_candidate_range=candidate_range(seen['head4']),
                           ln2_input_candidate_range=candidate_range(seen['ln2_input']),
                           ln2_output_candidate_range=candidate_range(seen['ln2']),
                           head2_candidate_range=candidate_range(seen['head2']),
                           ln2_input_min_abs_channel_difference=float(diff.abs().min()),
                           ln2_input_sign_crosses_zero=bool((diff<0).any() and (diff>0).any()),
                           normalization_amplitude_ge_099_fraction=float((amp>=.99).float().mean()),
                           head2_relu_zero_fraction=float((seen['head2']==0).float().mean()),
                           head2_all_candidates_zero=bool((seen['head2']==0).all()))
                    rows.append(d);local_rows.append(d);dist_out.append((p,logits))
                    if idx in check_ids.values():
                        rng=np.random.RandomState(20261102+idx);perm=torch.tensor(rng.permutation(L))
                        with torch.no_grad():
                            repeat=net(x,n,m)[0].numpy()
                            permed=net(x[:,perm],n,m)[0].numpy()[np.argsort(perm.numpy())]
                            mask=torch.tensor([[False]*L+[True]*7])
                            pad=net(torch.cat([x,torch.zeros((1,7,x.shape[2]))],dim=1),n,m,mask)[0].numpy()
                            altered=x.clone();altered[:,:,0]=x[:,perm,0]
                            alt=probability(net(altered,n,m)[0])
                        invariants.append(dict(model=model,episode=episode,variant=variant,observation=key,L=L,
                                               repeat_max_error=float(np.max(np.abs(repeat-prob.numpy()))),
                                               permutation_max_error=float(np.max(np.abs(permed-prob.numpy()))),
                                               padding_max_error=float(np.max(np.abs(pad[:L]-prob.numpy()))),
                                               pad_probability=float(pad[L:].sum())))
                        sensitivity.append(dict(model=model,episode=episode,variant=variant,observation=key,L=L,
                                                feature_0_permutation_tv=float(.5*np.abs(alt-p).sum())))
                        net.zero_grad(set_to_none=True);xx,nn_,mm=inputs(r,True)
                        loss=-torch.distributions.Categorical(net(xx,nn_,mm)).log_prob(torch.tensor([r['action']])).mean()
                        loss.backward()
                        for module_name,module in [('embedding',net.var_embedding),('transformer',net.transformer),
                                                   ('bimatching',net.tree_refinement),('treegate',net.output_layer)]:
                            gs=[q.grad for q in module.parameters() if q.grad is not None]
                            norm=math.sqrt(sum(float(g.double().square().sum()) for g in gs))
                            assert math.isfinite(norm)
                            grads.append(dict(model=model,episode=episode,variant=variant,observation=key,L=L,
                                              module=module_name,l2_norm=norm,
                                              input_score_gradient_l2=float(xx.grad.norm())))
                assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
                result[variant]=dist_out
            for (key,name,r),(p,l1),(q,l2) in zip(observations,result['original'],result['ln2_affine_only']):
                mid=(p+q)/2
                paired.append(dict(model=model,episode=episode,observation=key,instance=name,L=len(p),
                                   tv_original_to_intervention=float(.5*np.abs(p-q).sum()),kl_original_to_intervention=kl(p,q),
                                   js=.5*(kl(p,mid)+kl(q,mid)),mean_logit_shift=float(l2.mean()-l1.mean()),
                                   top1_changed=bool(np.argmax(p)!=np.argmax(q))))
            assert sha(cp_path)==cp_hash
            checkpoint_info.append(dict(model=model,episode=episode,path=str(cp_path.relative_to(ROOT)),sha256=cp_hash,
                                        parameters_unchanged=True,variant_state_dict_equal_to_original=True))
            for variant in plan['variants']:
                rr=[r for r in local_rows if r['variant']==variant and r['L']>1]
                im=[]
                for name in sorted({r['instance'] for r in rr}):
                    rs=[r for r in rr if r['instance']==name]
                    im.append(float(np.median([r['tv_to_uniform'] for r in rs])))
                summary_by.append(dict(model=model,episode=episode,variant=variant,states=len(rr),instances=len(im),
                                       logit_spread=quantiles([r['logit_spread'] for r in rr]),
                                       tv_to_uniform=quantiles([r['tv_to_uniform'] for r in rr]),
                                       normalized_entropy=quantiles([r['normalized_entropy'] for r in rr]),
                                       instance_median_tv_to_uniform=quantiles(im),
                                       instances_median_tv_ge_001=sum(x>=.01 for x in im),
                                       states_tv_ge_001=sum(r['tv_to_uniform']>=.01 for r in rr),
                                       head2_all_candidates_zero=sum(r['head2_all_candidates_zero'] for r in rr)))
            for filename,data in [('policy_metrics.csv',rows),('paired_intervention.csv',paired),('score_gradients.csv',grads),
                                  ('invariance_checks.csv',invariants),('sensitivity_checks.csv',sensitivity)]:write(filename,data)
            print('completed',model,episode,'states',len(observations),'elapsed',time.perf_counter()-START,flush=True)
    assert all(sha(ROOT/p)==h for p,h in before_sources.items())
    assert all(sha(ROOT/p)==h for p,h in trajectory_hashes.items())
    assert 'pyscipopt' not in sys.modules and not torch.cuda.is_initialized()
    save('input_trajectory_hashes.json',trajectory_hashes);save('checkpoint_audit.json',checkpoint_info)
    save('runtime_sources_after.json',snapshot_sources())
    summary=dict(plan=plan,source_model_unchanged=True,source_trajectories_unchanged=True,checkpoints_unchanged=True,
                 all_variant_parameter_values_unchanged=True,cuda_initialized=False,pyscipopt_imported=False,
                 optimizer_steps=0,new_solver_tasks=0,torch_version=torch.__version__,torch_threads=torch.get_num_threads(),
                 torch_interop_threads=torch.get_num_interop_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                 cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
                 candidate_counts=quantiles([m['candidate_count'] for m in manifest]),
                 L1_states=sum(m['candidate_count']==1 for m in manifest),rows=len(rows),paired_rows=len(paired),
                 gradient_rows=len(grads),invariance_cases=len(invariants),per_checkpoint_variant=summary_by,
                 max_permutation_error=max(r['permutation_max_error'] for r in invariants),
                 max_padding_error=max(r['padding_max_error'] for r in invariants),
                 max_repeat_error=max(r['repeat_max_error'] for r in invariants),
                 max_pad_probability=max(r['pad_probability'] for r in invariants),
                 elapsed_seconds=time.perf_counter()-START)
    save('local_validation_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
