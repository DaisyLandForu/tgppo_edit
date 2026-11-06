"""Paired, forward-only timing of real final Actors on identical real inputs."""
import argparse
import time
import numpy as np
import torch
import server
from project.runtime import load_actor
from project.utils.protocol import write_json,sha256
from diagnose_cause import HERE,RUN,sample


def main(args):
    torch.set_num_threads(1)
    cause=server.read(HERE/'causal-results.json')
    if len(cause['summaries'])!=4 or not all(cause['checks'].values()):raise RuntimeError('Verify all four heads first')
    device=args.device
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    models={};protected={};counts={}
    for variant in ('transformer','bimatching'):
        for seed in (11,22):
            key=variant+'/'+str(seed);path=RUN/variant/'training'/('B0-s'+str(seed))/'latest.pt'
            protected[str(path)]=sha256(path);policy,payload=load_actor(path)
            models[key]=policy.actor_network.to(device).eval().requires_grad_(False)
            counts[key]=dict(actor=sum(p.numel() for p in models[key].parameters()),
                actor_transformer=sum(p.numel() for n,p in models[key].named_parameters() if n.startswith('transformer.')),
                critic=sum(v.numel() for v in payload['critic'].values()))
    states=sample(RUN/'transformer/training/B0-s11')[::4];rows=[]
    with torch.no_grad():
        for idx,state in enumerate(states):
            inputs=[torch.as_tensor(state[k],device=device).unsqueeze(0) for k in ('cands_state','node_state','mip_state')]
            timings={key:[] for key in models}
            for model in models.values():
                for _ in range(args.warmup):model(*inputs)
            for rep in range(args.repeats):
                keys=list(models);offset=(idx+rep)%len(keys);keys=keys[offset:]+keys[:offset]
                for key in keys:
                    if device.startswith('cuda'):torch.cuda.synchronize()
                    start=time.perf_counter();models[key](*inputs)
                    if device.startswith('cuda'):torch.cuda.synchronize()
                    timings[key].append((time.perf_counter()-start)*1000)
            for key,values in timings.items():
                rows.append(dict(model=key,instance=state['instance'],job_id=state['job_id'],transition=state['transition'],
                    candidates=len(state['cands_state']),mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),
                    p95_ms=float(np.percentile(values,95)),samples_ms=values))
    summary={key:dict(mean_ms=float(np.mean([t for r in rows if r['model']==key for t in r['samples_ms']])),
        p50_ms=float(np.percentile([t for r in rows if r['model']==key for t in r['samples_ms']],50)),
        p95_ms=float(np.percentile([t for r in rows if r['model']==key for t in r['samples_ms']],95))) for key in models}
    checks=dict(checkpoints_unchanged=all(sha256(p)==v for p,v in protected.items()),
        complete_same_inputs=len(rows)==4*len(states))
    write_json(HERE/('trained-benchmark-'+device.replace(':','-')+'.json'),dict(checks=checks,summary=summary,rows=rows,
        parameters=counts,device=device,repeats=args.repeats,warmup=args.warmup,states=len(states),batch_size=1,
        stack=server.stack(server.DEFAULT_ROOT),resources=server.resource_limits(),protected_checkpoint_sha256=protected,
        scope='Same 12 real training states, candidate-count range restricted by the probe to 2..512; forward-only latency, no feature/sampling/solver work; local CPU or explicitly selected CUDA.'))
    print(summary,flush=True)
    if not all(checks.values()):raise RuntimeError('Benchmark protection failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmup',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1 or args.warmup<1:raise ValueError('Positive warmup and repeats required')
    main(args)
