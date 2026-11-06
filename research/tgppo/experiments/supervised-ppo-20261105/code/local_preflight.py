"""One-thread local BC/latency/interface checks, with cumulative 600s ceiling."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import server
from project import runtime as rt
from project.policy.light_actor import make_actor
from project.agent.agent import ActorPolicy
from expert_data import OUTPUT, ROOT, load_arrays, batch, sha256, write_json
from pretrain import fit
from engineering import bridge


def configuration(kind,seed=11):
    config=json.loads((ROOT/'tgppo/configs/paper_hparams.json').read_text())
    audit=json.loads((OUTPUT/'data_audit.json').read_text())
    config.update(actor_kind=kind,actor_width=64,inducing_points=8,server_training_seed=seed,
        learner_device='cpu',expert_dataset_sha256=audit['dataset_sha256'],
        bc_steps=400,bc_batch_size=16,bc_lr=.001)
    return rt.config_with_protocol(config)


def model_checks():
    checks={};critics=[];streams=[]
    rng=np.random.RandomState(23)
    x=torch.tensor(rng.randn(2,13,25),dtype=torch.float32)
    n=torch.tensor(rng.randn(2,8),dtype=torch.float32);m=torch.tensor(rng.randn(2,53),dtype=torch.float32)
    mask=torch.arange(13)[None,:]>=torch.tensor([7,13])[:,None]
    perm=torch.tensor(rng.permutation(13));inverse=perm.argsort()
    for kind in ('b0','mlp','set','isab'):
        rt.seed_all(11);agent=rt.build_agent(configuration(kind),training=False)
        critics.append(server.digest_actor(agent.critic_network));streams.append(torch.get_rng_state().clone())
        model=agent.actor_network.eval()
        with torch.no_grad():
            p=model(x,n,m,mask);q=model(x[:,perm],n,m,mask[:,perm])[:,inverse]
            single=model(x[:1,:7],n[:1],m[:1]);one=model(x[:1,:1],n[:1],m[:1])
        checks[kind+'_finite_normalized']=bool(torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(2),atol=1e-6))
        checks[kind+'_padding_zero']=bool((p[mask]==0).all())
        checks[kind+'_permutation']=bool(torch.allclose(p,q,atol=2e-6,rtol=2e-5))
        checks[kind+'_padding_invariance']=bool(torch.allclose(p[0,:7],single[0],atol=2e-6,rtol=2e-5))
        checks[kind+'_one_candidate']=bool(torch.equal(one,torch.ones_like(one)))
        if kind!='b0':
            model.zero_grad();(-model(x,n,m,mask)[:,0].log().mean()).backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            checks[kind+'_finite_nonzero_gradients']=bool(all(torch.isfinite(g).all() for g in grads)
                and sum(float(g.abs().sum()) for g in grads)>0.)
    checks['paired_original_critic_initialization']=len(set(critics))==1
    checks['paired_public_rng_stream']=all(torch.equal(streams[0],s) for s in streams)
    return checks


def benchmark(models,data):
    lengths=np.diff(data['offsets']);ids=np.argsort(lengths,kind='stable')[np.linspace(0,len(lengths)-1,12,dtype=int)]
    rows=[]
    for kind,model in models.items():
        model.eval();forward=[];path=[];policy=ActorPolicy(model)
        by_state=[]
        for idx in ids:
            x,n,m,mask,_=batch(data,[idx]);c=x[0].numpy();node=n[0].numpy();mip=m[0].numpy()
            with torch.no_grad():
                for _ in range(2):model(x,n,m,mask)
                for _ in range(5):
                    start=time.perf_counter();model(x,n,m,mask);forward.append(time.perf_counter()-start)
                    start=time.perf_counter();policy.act(c,mip,node);elapsed=time.perf_counter()-start
                    path.append(elapsed);by_state.append(dict(L=len(c),path_seconds=elapsed))
        rows.append(dict(actor=kind,parameters=sum(p.numel() for p in model.parameters()),
            forward_ms={p:float(np.percentile(forward,q)*1000) for p,q in [('p50',50),('p95',95)]},
            actor_path_ms={p:float(np.percentile(path,q)*1000) for p,q in [('p50',50),('p95',95)]},
            calls=len(path),L_by_state=[int(lengths[i]) for i in ids],measurements=by_state))
    baseline=next(r for r in rows if r['actor']=='b0')
    for row in rows:
        row['actor_path_ratio_vs_b0']={p:row['actor_path_ms'][p]/baseline['actor_path_ms'][p] for p in ('p50','p95')}
    return dict(cpu_model=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
        torch=torch.__version__,threads=torch.get_num_threads(),dtype='float32',batch_states=1,rows=rows,
        interpretation='12 real expert states stratified by L, 5 hot calls/state. Percentiles are individual calls in this probe, not deployment-wide estimates.')


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('Local checks must be CPU-only')
    ledger_path=OUTPUT/'local_budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(limit_seconds=600.,
        data_preparation_seconds=json.loads((OUTPUT/'data_audit.json').read_text())['seconds'],attempts=[])
    if any('elapsed_seconds' not in a for a in ledger['attempts']):
        raise RuntimeError('Previous unaccounted local attempt; inspect before proceeding')
    used=ledger['data_preparation_seconds']+sum(a['elapsed_seconds'] for a in ledger['attempts'])
    remaining=ledger['limit_seconds']-used
    if remaining<=0:raise RuntimeError('Local 600-second budget exhausted')
    record=dict(started_epoch=time.time(),allocated_seconds=remaining);ledger['attempts'].append(record);write_json(ledger_path,ledger)
    start=time.monotonic();result=dict(passed=False,checks={},bc={},bridges={})
    def expired(*_):raise TimeoutError('Cumulative local 600-second wall cap')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        result['checks']=model_checks()
        if not all(result['checks'].values()):raise RuntimeError('Model structural check failed')
        print('Structural, paired-Critic and RNG checks passed',flush=True)
        models={}
        for kind in ('mlp','set','isab'):
            cfg=configuration(kind);rt.seed_all(11);agent=rt.build_agent(cfg,training=False)
            model=agent.actor_network
            result['bc'][kind]=fit(model,cfg,OUTPUT/'probe'/kind,'cpu',resume=True,
                heartbeat=lambda event:print(json.dumps(dict(actor=kind,**event)),flush=True))
            models[kind]=model
        cfg=configuration('b0');rt.seed_all(11);models['b0']=rt.build_agent(cfg,training=False).actor_network
        arch=dict(cfg['state_dims'],**{k:cfg[k] for k in rt.ARCH_KEYS})
        models['tiny_transformer']=make_actor(arch,'tiny_transformer')
        result['benchmark']=benchmark(models,load_arrays())
        write_json(OUTPUT/'latency.json',result['benchmark'])
        # Real optimizers and checkpoint recovery, on synthetic transitions only.
        for kind in ('b0','mlp','set','isab'):
            cfg=configuration(kind)
            state=models[kind].state_dict()
            result['bridges'][kind]=bridge(cfg,'cpu',OUTPUT/'bridges'/kind,actor_state=state)
            print(kind+' synthetic PPO/recovery passed',flush=True)
        result['bc_validation_gate']={k:r['final']['validation']['macro']['normalized_nll']<=.90 for k,r in result['bc'].items()}
        result['latency_gate']={r['actor']:all(v<=.5 for v in r['actor_path_ratio_vs_b0'].values())
            for r in result['benchmark']['rows'] if r['actor'] in ('mlp','set','isab')}
        result['passed']=all(result['checks'].values()) and all(b['passed'] for b in result['bridges'].values())
        result['all_three_eligible']=all(result['bc_validation_gate'].values()) and all(result['latency_gate'].values())
        result['scope']='CPU offline BC probes, latency, and synthetic update/recovery. No real PPO episode or SCIP solve.'
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        record['elapsed_seconds']=time.monotonic()-start;record['passed']=result['passed']
        ledger['used_seconds']=ledger['data_preparation_seconds']+sum(a['elapsed_seconds'] for a in ledger['attempts'])
        result['local_cumulative_seconds']=ledger['used_seconds'];result['cuda_initialized']=torch.cuda.is_initialized()
        write_json(ledger_path,ledger);write_json(OUTPUT/'local_checks.json',result)
    print(json.dumps({k:result[k] for k in ('passed','all_three_eligible','bc_validation_gate','latency_gate','local_cumulative_seconds')},indent=2))


if __name__=='__main__':main()
