"""Actual target CPU/CUDA stream pairing check, without any SCIP optimization."""
import gc
import hashlib
import torch
import server
from project import runtime as rt
from project.utils.protocol import write_json


def digest_rng():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else [])


def check_rng_pairing(c,seed,destination):
    cfg=server.params(server.DEFAULT_ROOT,c,'B0',seed,'cuda')
    states=[]
    # All three builders consume the original Actor/Critic stream, isolating
    # lightweight construction from already-initialized CUDA as well as CPU.
    for kind,weight in [('b0',0.),('mlp',0.),('mlp',.01)]:
        rt.seed_all(seed)
        agent=rt.build_agent(dict(cfg,actor_kind=kind,expert_supervision_weight=weight),device='cuda')
        states.append(dict(kind=kind,weight=weight,critic=server.digest_actor(agent.critic_network),
            actor=server.digest_actor(agent.actor_network),rng=digest_rng()))
        del agent
        gc.collect()
        torch.cuda.empty_cache()
    assert all(s['critic']==states[0]['critic'] and s['rng']==states[0]['rng'] for s in states)
    assert states[1]['actor']==states[2]['actor']
    result=dict(passed=True,actual_cuda_checked=True,states=states,scip_solves=0)
    write_json(destination,result)
    return result
