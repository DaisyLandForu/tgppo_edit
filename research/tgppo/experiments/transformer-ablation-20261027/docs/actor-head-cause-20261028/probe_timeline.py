"""Fixed real-state probes at initialization and saved milestones; no learning."""
import json
import numpy as np
import torch
import ablation
import server
from project.policy import Actor
from project.runtime import architecture, load_actor, seed_all
from project.utils.protocol import write_json, sha256
from diagnose_cause import HERE, RUN, sample, distribution


def run():
    torch.set_num_threads(1)
    # One shared pool for every model/time point. Full candidate sets, no truncation.
    states=sample(RUN/'transformer/training/B0-s11')[::4]
    records=[];checks={};protected={}
    for variant in ('transformer','bimatching'):
        for seed in (11,22):
            out=RUN/variant/'training'/('B0-s'+str(seed))
            config=server.read(out/'config.json')
            for episode in (0,100,200,300,400,500):
                if episode:
                    path=out/('episode-{:04d}.pt'.format(episode))
                    protected[str(path)]=sha256(path)
                    policy,payload=load_actor(path);actor=policy.actor_network
                else:
                    seed_all(seed);actor=Actor(**architecture(config))
                    checks[variant+str(seed)+'_initial_digest']=server.digest_actor(actor)==server.read(out/'initialization.json')['actor_fingerprint']
                actor.eval().requires_grad_(False);cache={};rows=[]
                def hook(module,args,output):cache['ln2']=args[0].detach()
                handle=actor.output_layer.BranchingNet[6][1].register_forward_hook(hook)
                def logit_hook(module,args,output):cache['logits']=output.detach()
                handle2=actor.output_layer.register_forward_hook(logit_hook)
                with torch.no_grad():
                    for state in states:
                        args=[torch.as_tensor(state[k]).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
                        p=actor(*args);x=cache['ln2'].double();d=x[...,0]-x[...,1]
                        rows.append(dict(instance=state['instance'],n=p.numel(),
                            same_channel_order=bool((d>0).all() or (d<0).all()),
                            min_abs_channel_difference=float(d.abs().min()),**distribution(cache['logits'])))
                handle.remove();handle2.remove()
                records.append(dict(variant=variant,seed=seed,episode=episode,states=len(rows),
                    logit_span_median=float(np.median([r['logit_span'] for r in rows])),
                    tv_uniform_max=max(r['tv_uniform'] for r in rows),
                    same_channel_order_count=sum(r['same_channel_order'] for r in rows),per_state=rows))
                print(json.dumps({k:v for k,v in records[-1].items() if k!='per_state'}),flush=True)
                if episode:checks[str(path)+'_unchanged']=sha256(path)==protected[str(path)]
                del actor
    write_json(HERE/'timeline.json',dict(checks=checks,records=records,protected_checkpoint_sha256=protected,
        scope='Shared 12 real training states; initialization reconstruction and checkpoints; no optimization or test-instance use.'))
    if not all(checks.values()):raise RuntimeError('Timeline parity/protection check failed')


if __name__=='__main__':run()
