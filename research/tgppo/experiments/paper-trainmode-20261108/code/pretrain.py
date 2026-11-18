"""Bounded instance-balanced BC; separate optimizer; no PPO replay or solver."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from expert_data import OUTPUT, load_arrays, batch, sha256, write_json


def signature(config):
    result={k:config[k] for k in ('actor_kind','actor_width','inducing_points','server_training_seed',
        'expert_dataset_sha256','bc_steps','bc_batch_size','bc_lr')}
    if 'bc_selection_milestones' in config:result['bc_selection_milestones']=config['bc_selection_milestones']
    return result


def loss_for(model, inputs):
    x,node,mip,mask,label=inputs
    if hasattr(model,'logits'):
        return F.cross_entropy(model.logits(x,node,mip,mask),label)
    probs=model(x,node,mip,mask)
    return F.nll_loss(probs.clamp_min(torch.finfo(probs.dtype).tiny).log(),label)


@torch.no_grad()
def evaluate(model,data,split,device='cpu'):
    model.eval();ids=np.flatnonzero(data['split']==split);per_instance={}
    for start in range(0,len(ids),8):
        selected=ids[start:start+8];x,n,m,mask,label=batch(data,selected,device)
        probs=model(x,n,m,mask)
        chosen=probs.gather(1,label[:,None]).squeeze(1).clamp_min(torch.finfo(probs.dtype).tiny)
        nll=-chosen.log();lengths=(~mask).sum(1);norm=nll/lengths.float().log()
        correct=probs.argmax(1)==label
        for i,idx in enumerate(selected):
            per_instance.setdefault(str(data['instances'][idx]),[]).append(
                (float(nll[i]),float(norm[i]),float(correct[i]),1./int(lengths[i])))
    details={name:dict(zip(('nll','normalized_nll','accuracy','uniform_accuracy'),np.mean(rows,axis=0).tolist()))
             for name,rows in sorted(per_instance.items())}
    return dict(instances=details,macro={k:float(np.mean([r[k] for r in details.values()]))
        for k in ('nll','normalized_nll','accuracy','uniform_accuracy')},states=len(ids))


def cpu_state(network):
    return {k:v.detach().cpu().clone() for k,v in network.state_dict().items()}


def fit(model,config,directory,device='cpu',resume=False,heartbeat=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    sig=signature(config);meta=directory/'result.json';actor_file=directory/'actor.pt'
    if meta.exists():
        result=json.loads(meta.read_text())
        if result['signature']!=sig or sha256(actor_file)!=result['actor_sha256']:
            raise ValueError('BC initialization provenance changed')
        model.load_state_dict(torch.load(actor_file,map_location=device)['actor'])
        return result
    if sha256(OUTPUT/'expert.npz')!=config['expert_dataset_sha256']:
        raise ValueError('Expert dataset hash mismatch')
    data=load_arrays();train=np.flatnonzero(data['split']=='train')
    by_instance={n:train[data['instances'][train]==n] for n in sorted(set(data['instances'][train]))}
    names=sorted(by_instance);model.to(device).eval();optimizer=torch.optim.Adam(model.parameters(),lr=config['bc_lr'])
    checkpoint=directory/'latest.pt';next_step=0;start=time.monotonic();previous_seconds=0.
    if checkpoint.exists():
        if not resume:raise FileExistsError('BC state exists; use --resume')
        saved=torch.load(checkpoint,map_location=device)
        if saved['signature']!=sig:raise ValueError('BC resume configuration changed')
        model.load_state_dict(saved['actor']);optimizer.load_state_dict(saved['optimizer'])
        next_step=saved['next_step'];previous_seconds=saved['seconds']
        initial=saved['initial'];history=saved['history']
        best=saved.get('best')
    else:
        initial={s:evaluate(model,data,s,device) for s in ('train','validation')};history=[]
        best=None
    milestones=config.get('bc_selection_milestones',[])
    if milestones and best is None:
        best=dict(step=0,score=initial['validation']['macro']['normalized_nll'],actor=cpu_state(model))
    order_hash=hashlib.sha256()
    def save(step):
        value=dict(signature=sig,actor=cpu_state(model),optimizer=optimizer.state_dict(),next_step=step,
            seconds=previous_seconds+time.monotonic()-start,initial=initial,history=history)
        value['best']=best
        tmp=checkpoint.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(checkpoint)
    if next_step==0:save(0)
    for step in range(next_step,config['bc_steps']):
        # Independent per-step stream: exact cursor recovery without global RNG changes.
        rng=np.random.RandomState(12000+config['server_training_seed']+step*7919)
        ids=[int(rng.choice(by_instance[names[int(rng.randint(len(names)))]])) for _ in range(config['bc_batch_size'])]
        order_hash.update(np.asarray(ids,np.int64).tobytes())
        optimizer.zero_grad(set_to_none=True);loss=loss_for(model,batch(data,ids,device))
        if not torch.isfinite(loss):raise ValueError('Nonfinite BC loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%100==0 or step+1 in milestones or step+1==config['bc_steps']:
            event=dict(step=step+1,loss=float(loss.detach()),seconds=previous_seconds+time.monotonic()-start)
            if step+1 in milestones:
                validation=evaluate(model,data,'validation',device)
                score=validation['macro']['normalized_nll'];event['validation']=validation
                if score < best['score']:
                    best=dict(step=step+1,score=score,actor=cpu_state(model))
            history.append(event);save(step+1)
            if heartbeat:heartbeat(event)
    if best is not None:model.load_state_dict(best['actor'])
    final={s:evaluate(model,data,s,device) for s in ('train','validation')}
    payload=dict(schema='light-bc-v1',signature=sig,actor=cpu_state(model))
    tmp=actor_file.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(actor_file)
    result=dict(signature=sig,initial=initial,final=final,history=history,actor_sha256=sha256(actor_file),
        seconds=previous_seconds+time.monotonic()-start,device=device,
        batch_order_hash_this_attempt=order_hash.hexdigest(),completed_steps=config['bc_steps'],
        selected_step=best['step'] if best is not None else config['bc_steps'],
        selection='minimum heldout-instance macro normalized NLL; exact ties favor earlier; fixed training budget completed' if best else 'fixed final step',
        label='offline native expert transfer; instance heldout within TGPPO train; not solver gains')
    write_json(meta,result);return result


def load_initialization(agent,config,directory):
    directory=Path(directory);record=json.loads((directory/'result.json').read_text())
    path=directory/'actor.pt'
    if record['signature']!=signature(config) or record['actor_sha256']!=sha256(path):
        raise ValueError('Uncertified BC warm start')
    agent.actor_network.load_state_dict(torch.load(path,map_location=agent.device)['actor'])
    if agent.actor_optimizer.state:raise ValueError('PPO must start with fresh optimizer, separate from BC')
