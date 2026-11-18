"""CPU-only frozen-output audit; no optimizer steps, SCIP solves or checkpoint writes."""
import csv
import datetime as dt
import gc
import hashlib
import io
import json
import pickle
import statistics as st
import time
from pathlib import Path

START = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT/'tgppo/runs'

import numpy as np
import torch
from project.policy.light_actor import make_actor

torch.set_num_threads(1)
torch.manual_seed(20261108)


def write(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def load(path):
    # Hash and decode the same bytes, even if a training process replaces latest.pt.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return torch.load(io.BytesIO(raw),map_location='cpu'),digest


raw = (ROOT/'docs/diagnostics/tgppo_20261102_signal_localization/observations.pkl').read_bytes()
pool = pickle.loads(raw)
anchors, seen_instances = [], set()
for key, instance, row in pool:
    if instance not in seen_instances and len(row['cands_state'])>1:
        anchors.append((key,instance,row,'historical_fixed_anchor'))
        seen_instances.add(instance)
pending, pending_sha = load(RUN/'paper-aligned-20261106/training/mlp-s22/pending.pt')
current = pending['memory']['episodes'][0]['transitions']
for i in sorted(set((0,len(current)//4,len(current)//2,3*len(current)//4,len(current)-1))):
    anchors.append(('current-rmatr-'+str(i),'rmatr200-p5',current[i],'current3600_saved_state'))
del pending
write('anchors.csv',[dict(observation=k,instance=n,pool=p,candidates=len(r['cands_state']))
                     for k,n,r,p in anchors])

models = []
for seed in (11,22):
    base = RUN/'paper-aligned-20261106/training'/('b0-s'+str(seed))
    models.extend([('current_b0_s{}_initial'.format(seed),base/'episode-0000.pt','b0',seed),
                   ('current_b0_s{}_latest'.format(seed),base/'latest.pt','b0',seed)])
for seed in (11,22):
    models.extend([('old_light_b0_s{}_ep100'.format(seed),RUN/'light-actor-20261105/b0/training'/('B0-s'+str(seed))/'episode-0100.pt','b0',seed),
                   ('old_server_b0_s{}_ep500'.format(seed),RUN/'server6-20261022/training'/('B0-s'+str(seed))/'episode-0500.pt','b0',seed)])
for variant in ('mlp','mlp_bc'):
    models.append(('current_'+variant+'_s11_latest',RUN/'paper-aligned-20261106/training'/(variant+'-s11')/'latest.pt','mlp',11))

summaries, policies, layers, gradients, parameter_changes = [], [], [], [], []
initial_outputs, initial_weights = {}, {}
for name,path,kind,seed in models:
    payload, digest = load(path)
    cfg = payload['config']
    arch = dict(cfg['state_dims'],**{k:cfg[k] for k in ('hidden_dim','num_heads','num_layers','dropout')})
    actor = make_actor(arch,kind,cfg.get('actor_width',64),cfg.get('inducing_points',8))
    actor.load_state_dict(payload['actor']);actor.eval()
    if name.startswith('current_b0') and name.endswith('initial'):
        initial_weights[seed] = {k:v.clone() for k,v in payload['actor'].items()}
    if name.startswith('current_b0') and name.endswith('latest'):
        groups = payload['actor_optimizer']['param_groups']
        states = payload['actor_optimizer']['state']
        names = [k for k,_ in actor.named_parameters()]
        ids = [i for g in groups for i in g['params']]
        assert len(groups)==1 and len(ids)==len(names)
        group = groups[0]
        delta2,decay2,residual2,init2 = 0.,0.,0.,0.
        steps = []
        for key,i in zip(names,ids):
            before = initial_weights[seed][key].double()
            after = payload['actor'][key].double()
            step = int(states[i]['step'].item()) if i in states else 0
            steps.append(step)
            predicted = before*((1-group['lr']*group['weight_decay'])**step)
            init2 += before.square().sum().item()
            delta2 += (after-before).square().sum().item()
            decay2 += (predicted-before).square().sum().item()
            residual2 += (after-predicted).square().sum().item()
        parameter_changes.append(dict(model=name,actor_change_l2=delta2**.5,
            initial_norm=init2**.5,zero_gradient_decay_prediction_l2=decay2**.5,
            residual_from_decay_prediction_l2=residual2**.5,
            adam_step_min=min(steps),adam_step_max=max(steps),optimizer=cfg.get('optimizer'),
            note='Decay counterfactual assumes explicit zero gradients on every step; no causal fraction inferred.'))
    output_rows = []
    for index,(key,instance,row,pool_name) in enumerate(anchors):
        x = torch.as_tensor(row['cands_state'],dtype=torch.float32)[None]
        n = torch.as_tensor(row['node_state'],dtype=torch.float32)[None]
        m = torch.as_tensor(row['mip_state'],dtype=torch.float32)[None]
        captured, handles = {}, []
        def hook(label):
            def capture(mod,inputs,value):
                captured[label] = value.detach().clone()
                if label=='ln2':
                    captured['ln2_input'] = inputs[0].detach().clone()
            return capture
        if kind=='b0':
            modules = [('embedding',actor.var_embedding),('fusion',actor.global_embedding),
                       ('transformer',actor.transformer),('bimatching',actor.tree_refinement),
                       ('ln2',actor.output_layer.BranchingNet[-2][1]),('logits',actor.output_layer)]
            modules += [('head_'+str(seq[0].out_features),seq) for seq in actor.output_layer.BranchingNet]
            for label,mod in modules:
                handles.append(mod.register_forward_hook(hook(label)))
        else:
            handles.append(actor.head.register_forward_hook(hook('logits')))
        with torch.no_grad():
            p = actor(x,n,m)[0].double().numpy()
        for h in handles:h.remove()
        p = p/p.sum()
        logits = captured['logits'].reshape(-1).double().numpy()
        assert len(p)==len(logits)==len(row['cands_state'])
        assert np.isfinite(p).all() and (p>0).all()
        result = dict(model=name,observation=key,instance=instance,pool=pool_name,candidates=len(p),
            tv_uniform=float(.5*np.abs(p-1/len(p)).sum()),entropy_fraction=float(-(p*np.log(p)).sum()/np.log(len(p))),
            probability_max_min=float(p.max()-p.min()),logit_range=float(np.ptp(logits)),
            max_probability_ties=int((p==p.max()).sum()),argmax=int(np.argmax(p)))
        if name.startswith('current_b0') and name.endswith('initial'):
            initial_outputs[(seed,key)] = (p.copy(),logits.copy())
        if name.startswith('current_b0') and name.endswith('latest'):
            p0,l0 = initial_outputs[(seed,key)]
            change = logits-l0
            result.update(tv_initial=float(.5*np.abs(p-p0).sum()),
                max_ratio_change_from_initial=float(np.max(np.abs(p/p0-1))),
                common_logit_shift=float(np.mean(change)),
                centered_logit_change_rms=float(np.sqrt(np.mean((change-change.mean())**2))))
        if kind=='b0':
            d = captured['ln2_input'][0,:,0]-captured['ln2_input'][0,:,1]
            result.update(ln2_difference_min=float(d.min()),ln2_difference_max=float(d.max()),
                ln2_positive_fraction=float((d>0).float().mean()),
                ln2_variance_over_eps_median=float(captured['ln2_input'].var(-1,unbiased=False).median()/actor.output_layer.BranchingNet[-2][1].eps))
            for label,tensor in captured.items():
                if tensor.ndim==3:
                    z = tensor[0] if tensor.shape[0]==1 else tensor[:,0,:]
                else:
                    z = tensor.reshape(-1,1)
                z = z.double()
                centered = z-z.mean(0,keepdim=True)
                rms = float(centered.square().mean().sqrt())
                total = float(z.square().mean().sqrt())
                layers.append(dict(model=name,observation=key,instance=instance,stage=label,
                    width=z.shape[-1],centered_rms=rms,total_rms=total,
                    relative_candidate_contrast=rms/max(total,1e-30)))
        policies.append(result);output_rows.append(result)
    if name.startswith('current_') and name.endswith('latest'):
        # Fixed saved actions: score gradients, NOT a PPO update or fresh rollout.
        chosen = sorted(range(len(anchors)),key=lambda i:len(anchors[i][2]['cands_state']))
        for index in sorted(set((chosen[0],chosen[len(chosen)//2],chosen[-1]))):
            key,instance,row,pool_name = anchors[index]
            x = torch.as_tensor(row['cands_state'],dtype=torch.float32)[None].clone().requires_grad_(True)
            n = torch.as_tensor(row['node_state'],dtype=torch.float32)[None]
            m = torch.as_tensor(row['mip_state'],dtype=torch.float32)[None]
            actor.zero_grad(set_to_none=True)
            probs = actor(x,n,m)
            loss = -torch.distributions.Categorical(probs).log_prob(torch.tensor([row['action']])).mean()
            loss.backward()
            norms = {}
            for key2,param in actor.named_parameters():
                prefix = key2.split('.')[0]
                norms[prefix] = norms.get(prefix,0.)+(param.grad.double().square().sum().item() if param.grad is not None else 0.)
            gradients.append(dict(model=name,observation=key,instance=instance,candidates=x.shape[1],
                score_gradient_l2=sum(norms.values())**.5,input_gradient_l2=float(x.grad.double().norm()),
                **{k+'_grad_l2':v**.5 for k,v in norms.items()}))
    summary = dict(model=name,path=str(path),sha256=digest,kind=kind,training_seed=seed,
        episode=payload.get('progress',{}).get('next_episode'),policy_version=payload.get('policy_version'),
        states=len(output_rows),tv_uniform_median=st.median(r['tv_uniform'] for r in output_rows),
        tv_uniform_max=max(r['tv_uniform'] for r in output_rows),
        entropy_fraction_median=st.median(r['entropy_fraction'] for r in output_rows))
    if 'tv_initial' in output_rows[0]:
        summary.update(tv_initial_median=st.median(r['tv_initial'] for r in output_rows),
            tv_initial_max=max(r['tv_initial'] for r in output_rows),
            common_shift_abs_median=st.median(abs(r['common_logit_shift']) for r in output_rows),
            centered_logit_change_median=st.median(r['centered_logit_change_rms'] for r in output_rows))
    summaries.append(summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    del actor,payload
    gc.collect()
write('policies.csv',policies);write('layers.csv',layers);write('score_gradients.csv',gradients)
write('parameter_changes.csv',parameter_changes)
report = dict(observed_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    models=summaries,parameter_changes=parameter_changes,gradients=gradients,
    observations_sha256=hashlib.sha256(raw).hexdigest(),current_pending_sha256=pending_sha,
    states=len(anchors),historical_instances=len(seen_instances),
    cuda_initialized=torch.cuda.is_initialized(),optimizer_steps=0,new_solves=0,
    elapsed_seconds=time.monotonic()-START)
(OUT/'OUTPUT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('COMPLETE',report['elapsed_seconds'],flush=True)
