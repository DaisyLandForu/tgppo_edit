"""Small shared helpers for a bounded end-to-end diagnostic, not a new baseline."""
from pathlib import Path
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import numpy as np
import torch

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[2]
OUT=ROOT/'docs/diagnostics/tgppo_20261102_end2end'
SOURCE=ROOT/'tgppo/runs/server6-20261022/training'
HEAD=OUT.parent/'tgppo_20261102_head_reset'
REPLAY=OUT.parent/'tgppo_20261102_ppo_replay'
BINDING=ROOT/'tgppo/experiments/stage4/metrics_binding/src'
sys.path[:0]=[str(PACKAGE/'code'),str(BINDING)]
torch.set_num_threads(1);torch.set_num_interop_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from project import runtime as rt
from project.policy import Actor
from project.policy.tail8 import replace_tail


def plan():return json.loads((OUT/'protocol.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify_inputs():
    frozen=json.loads((OUT/'frozen_inputs.json').read_text())
    for rel,digest in frozen.items():assert sha(ROOT/rel)==digest,rel
    return len(frozen)


def make_initial(seed,variant):
    cp=torch.load(SOURCE/f'B0-s{seed}/episode-0100.pt',map_location='cpu')
    cfg=dict(cp['config']);cfg.pop('server_protocol_sha256',None);cfg.pop('server_group',None)
    cfg.update(actor_variant=variant,learner_device='cpu',task_time_limit=plan()['solve_seconds'],n_epochs=1,
        diagnostic='tail8-bounded-end2end-v1',source_training_seed=seed,head_initialization_seed=101,
        initialization='ep100 upstream and Critic; paired reset-head init101; zero scalar readout; fresh Adam')
    actor=Actor(**dict(cfg['state_dims'],**{k:cfg[k] for k in rt.ARCH_KEYS}))
    actor.load_state_dict(cp['actor'])
    actor.output_layer.load_state_dict(torch.load(HEAD/f'trials/s{seed}-init101/initial.pt',map_location='cpu'))
    if variant=='tail8':replace_tail(actor)
    agent=rt.build_agent(cfg,training=True,device='cpu')
    agent.actor_network.load_state_dict(actor.state_dict());agent.critic_network.load_state_dict(cp['critic'])
    agent.policy_version=cp['policy_version']
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
    assert cp['memory']['num_transitions']==0
    return agent,cfg


def collate(rows,ids):
    rows=[rows[int(i)] for i in ids]
    states=[torch.as_tensor(r['cands_state'],dtype=torch.float32) for r in rows]
    cands=torch.nn.utils.rnn.pad_sequence(states,batch_first=True)
    lens=torch.tensor([len(x) for x in states]);mask=torch.arange(cands.shape[1])[None]>=lens[:,None]
    return dict(cands=cands,mask=mask,node_state=torch.tensor(np.stack([r['node_state'] for r in rows])),
        mip_state=torch.tensor(np.stack([r['mip_state'] for r in rows])),action=torch.tensor([r['action'] for r in rows]),
        log_prob=torch.tensor([r['log_prob'] for r in rows],dtype=torch.float32),indices=np.array(ids))


def responses(actor,rows,advantages=None):
    actor.eval();values=[];losses=[]
    order=np.argsort([len(r['cands_state']) for r in rows],kind='stable')
    with torch.no_grad():
        for start in range(0,len(order),16):
            ids=order[start:start+16];b=collate(rows,ids)
            prob=actor(b['cands'],b['node_state'],b['mip_state'],b['mask'])
            ratios=torch.exp(torch.distributions.Categorical(prob).log_prob(b['action'])-b['log_prob'])
            for j,i in enumerate(ids):
                n=len(rows[i]['cands_state']);p=prob[j,:n].double();p/=p.sum()
                values.append(float(.5*(p-1/n).abs().sum()))
                if advantages is not None:
                    a=float(advantages[i]);ratio=float(ratios[j]);losses.append(-min(ratio*a,min(max(ratio,.84),1.16)*a))
    return dict(tv_p50=float(np.median(values)),tv_max=max(values),response_states=sum(v>1e-6 for v in values),
                surrogate_loss=float(np.mean(losses)) if losses else None,states=len(rows))


def bounded_learn(agent,batches,directory):
    """Original learn/GAE/loss; one explicitly budgeted sequence of logical batches."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rows=agent.memory.rows();counts={'actor':0,'critic':0};gradients=[]
    original_generator=agent.memory.get_batch_generator
    agent.memory.get_batch_generator=lambda:iter(collate(rows,ids) for ids in batches)
    for kind,opt,net in [('actor',agent.actor_optimizer,agent.actor_network),('critic',agent.critic_optimizer,agent.critic_network)]:
        step=opt.step
        def wrapped(*args,_kind=kind,_step=step,_net=net,**kwargs):
            k=counts[_kind]+1;assert k<=len(batches)
            save(directory/'inflight.json',dict(network=_kind,next_step=k))
            modules={}
            for name,p in _net.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
                    key=name.split('.')[0];modules[key]=modules.get(key,0.)+float(p.grad.detach().double().square().sum())
            gradients.append(dict(network=_kind,step=k,modules={n:v**.5 for n,v in modules.items()}))
            value=_step(*args,**kwargs);counts[_kind]=k
            save(directory/'steps.json',dict(counts=counts,cap_each=len(batches)))
            (directory/'inflight.json').unlink()
            return value
        opt.step=wrapped
    result=agent.learn(microbatch_size=16)
    agent.memory.get_batch_generator=original_generator
    assert result['num_updates']==len(batches) and counts=={'actor':len(batches),'critic':len(batches)}
    for opt in [agent.actor_optimizer,agent.critic_optimizer]:
        assert {int(s['step']) for s in opt.state.values()}=={len(batches)}
    save(directory/'gradients.json',gradients);save(directory/'update.json',result)
    return result,gradients


def export_actor(path,agent,cfg):
    torch.save(dict(schema_version='actor-only-v1',config=cfg,config_sha256=rt.fingerprint(cfg),
                    actor=agent.actor_network.state_dict(),diagnostic=True),path)


def roundtrip(agent,cfg,directory,probe_rows):
    directory=Path(directory)
    full=directory/'latest.pt';rt.save_checkpoint(full,agent,cfg,progress={'diagnostic_block_complete':True})
    loaded,_=rt.load_agent(full,training=True,restore_random_state=False,device='cpu')
    assert all(torch.equal(v,loaded.actor_network.state_dict()[k]) for k,v in agent.actor_network.state_dict().items())
    exported=directory/'actor.pt';export_actor(exported,agent,cfg)
    # A constructor trap verifies deployment never builds a Critic.
    saved=rt.Critic
    def forbidden(*args,**kwargs):raise AssertionError('Critic constructed during Actor-only load')
    rt.Critic=forbidden
    try:policy,_=rt.load_actor(exported)
    finally:rt.Critic=saved
    b=collate(probe_rows,list(range(min(8,len(probe_rows)))))
    agent.actor_network.eval();loaded.actor_network.eval();policy.actor_network.eval()
    with torch.no_grad():
        a=agent.actor_network(b['cands'],b['node_state'],b['mip_state'],b['mask'])
        assert torch.equal(a,loaded.actor_network(b['cands'],b['node_state'],b['mip_state'],b['mask']))
        assert torch.equal(a,policy.actor_network(b['cands'],b['node_state'],b['mip_state'],b['mask']))
        assert (a[b['mask']]==0).all()
    return dict(checkpoint_reload_exact=True,actor_only_reload_exact=True,no_critic_constructed=True,padding_probability_zero=True)
