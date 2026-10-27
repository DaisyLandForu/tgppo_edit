"""Thin experiment commands around the verified B0 server trainer/evaluator."""
import argparse
import copy
import csv
import fcntl
import json
from pathlib import Path
import statistics
import time
import uuid
import numpy as np
import torch
import server
from project.runtime import (seed_all, build_agent, load_actor, action_seed)
from project.agent.agent import ActorPolicy
from project.environment import Environment
from project.utils.settings import state_dims
from project.utils.protocol import write_json, fingerprint, sha256

HERE = Path(__file__).resolve().parents[1]
ROOT = server.DEFAULT_ROOT
VARIANTS = ('transformer', 'bimatching')


def protocol(variant, smoke=False):
    c = server.plan(ROOT,HERE/'configs'/(variant+('-smoke' if smoke else '')+'.json'))
    if c['groups'] != ['B0'] or c['use_transformer'] != (variant == 'transformer'):
        raise ValueError('Only the paired original B0 architecture ablation is allowed')
    if smoke:
        c.update(train=['lseu','stein27','misc03','pp08aCUTS'],development=['vpm2','dcmulti'])
    return c, ROOT/c['run_directory']


def check_pair():
    a, _ = protocol('transformer'); b, _ = protocol('bimatching')
    allowed = {'use_transformer','variant','run_directory'}
    changed = {k for k in set(a)|set(b) if a.get(k) != b.get(k)}
    if changed != allowed:
        raise ValueError('Unexpected experimental variables: '+str(changed))
    if a['training_seeds'] != [11,22,33]:
        raise ValueError('Formal comparison requires the three frozen training seeds')


def baseline_pair(smoke=False):
    """Native results are architecture-independent; reuse only exact job identities."""
    check_pair()
    first, run = protocol('transformer',smoke)
    server.baseline(ROOT,first,run)
    original = server.read(run/'normalization/baselines.json')
    other, target = protocol('bimatching',smoke)
    signature = server.stack(ROOT)
    if original['stack'] != signature:
        raise ValueError('Native normalization machine/software changed')
    cutoffs = server.read(ROOT/'tgppo/datasets/cutoffs.json')
    for name in other['train']+other['development']:
        for ss in other['solver_seeds']:
            job = server.bound_job(ROOT,other,cutoffs,name,ss,11,'native')
            identity = server.job_identity(job)
            server.validate_baseline(original['baselines'][fingerprint(identity)],identity)
    write_json(target/'normalization/baselines.json',dict(original,
        protocol_sha256=fingerprint(other),source=str(run/'normalization/baselines.json'),
        source_sha256=sha256(run/'normalization/baselines.json')))


def smoke():
    """Run this on a GPU Custom job: two real batches and dev evaluation per arm."""
    if not torch.cuda.is_available():
        raise RuntimeError('Run smoke on a GPU Custom job; local MILP training is disabled')
    baseline_pair(smoke=True)
    checks = {}
    for variant in VARIANTS:
        c,run = protocol(variant,True)
        state = server.train(ROOT,c,run,'B0',11,True,'cuda')
        checks[variant+'_two_updates'] = sum(bool(m['updated']) for m in state['history']) >= 2
        checks[variant+'_finite_losses'] = all(np.isfinite(m.get('actor_loss',np.nan)) and
            np.isfinite(m.get('critic_loss',np.nan)) for m in state['history'])
        if not all(checks.values()):
            raise RuntimeError('Smoke needs two complete finite updates: '+str(checks))
        server.select(ROOT,c,run,'B0',11)
        out = server.group_dir(run,'B0',11)
        rows = [r for f in out.glob('development/*/selected_results.json') for r in server.read(f)]
        checks[variant+'_actor_only_online'] = (bool(rows) and all(
            r['valid'] and r['callback_seconds']['critic']==0 and r['fallback_action_count']==0 for r in rows))
    evidence = dict(passed=all(checks.values()),checks=checks,scope='GPU short training; no performance conclusion')
    evidence.update(protocols={v:fingerprint(protocol(v,True)[0]) for v in VARIANTS},
                    stack=server.stack(ROOT),gpu=torch.cuda.get_device_name(0))
    write_json(ROOT/'tgppo/runs/transformer-ablation-20261027-smoke/verification.json',evidence)
    if not evidence['passed']:raise RuntimeError('Smoke failed')


def prepare():
    """One GPU Custom job before the six parallel formal training jobs."""
    from server_bridge import bridge
    from server_pilot import pilot
    check_pair()
    for variant in VARIANTS:
        c,_ = protocol(variant);server.preflight(ROOT,c,'cuda')
    smoke_path=ROOT/'tgppo/runs/transformer-ablation-20261027-smoke/verification.json'
    if not smoke_path.exists():smoke()
    evidence=server.read(smoke_path)
    if (not evidence['passed'] or evidence['protocols']!={v:fingerprint(protocol(v,True)[0]) for v in VARIANTS}
        or evidence['stack']!=server.stack(ROOT) or evidence['gpu']!=torch.cuda.get_device_name(0)):
        raise RuntimeError('Short training proof does not match this code/protocol/hardware')
    baseline_pair()
    for variant in VARIANTS:
        c,run = protocol(variant)
        bridge(ROOT,c,run)
        pilot(ROOT,c,run)


def online_check():
    """Untrained policies test SCIP integration only. Never a quality comparison."""
    check_pair(); base = HERE/'docs'/'online-check'
    directory = base/('attempt-'+uuid.uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    cutoffs = server.read(ROOT/'tgppo/datasets/cutoffs.json');rows=[]
    for variant in VARIANTS:
        c,_ = protocol(variant);c=copy.deepcopy(c);c['time_limit']=2.
        cfg=server.params(ROOT,c,'B0',11,'cpu');seed_all(11)
        agent=build_agent(cfg,training=False)
        policy=ActorPolicy(agent.actor_network)
        j=server.bound_job(ROOT,c,cutoffs,'lseu',0,11,'eval')
        seed_all(action_seed(11,'online-lseu-s0'))
        env=Environment('cpu',policy,state_dims,j['limits'],j['params'],0,mode='eval')
        env.reset(j['instance'],j['cutoff'],None,j['cutoff_provenance'])
        _,row,_=env.run_episode()
        row.update(variant=variant,trained=False,scope='untrained callback integration only')
        write_json(directory/(variant+'.json'),row);rows.append(row)
        if not row['valid'] or row.get('branch_count',0)==0 or row['fallback_action_count']:
            raise RuntimeError('Online integration failed: '+str(directory))
    # Additionally verify the actual existing trained B0 export in the new loader.
    policy,_=load_actor(ROOT/'tgppo/runs/server6-20261022/training/B0-s11/actor_best.pt')
    seed_all(action_seed(11,'legacy-lseu-s0'))
    env=Environment('cpu',policy,state_dims,j['limits'],j['params'],0,mode='eval')
    env.reset(j['instance'],j['cutoff'],None,j['cutoff_provenance'])
    _,row,_=env.run_episode();write_json(directory/'legacy-b0.json',row)
    if not row['valid'] or row.get('branch_count',0)==0:raise RuntimeError('Legacy B0 integration failed')
    write_json(base/'verification.json',dict(passed=True,training_executed=False,directory=str(directory),
        stack=server.stack(ROOT),scope='online callback compatibility, not ablation performance'))


def benchmark(device,repeats):
    """Paired fixed-input forward benchmark; initialization only, no trained quality claims."""
    check_pair();torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    seed_all(13)
    data={length:(torch.randn(1,length,25,device=device),torch.randn(1,8,device=device),
                  torch.randn(1,53,device=device)) for length in (1,8,32,128,512)}
    models={};counts={}
    for variant in VARIANTS:
        c,_=protocol(variant);cfg=server.params(ROOT,c,'B0',11,'cpu');seed_all(11)
        a=build_agent(cfg,training=False)
        counts[variant]=dict(actor=sum(p.numel() for p in a.actor_network.parameters()),
            critic=sum(p.numel() for p in a.critic_network.parameters()))
        models[variant]=a.actor_network.to(device).eval()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    rows=[]
    with torch.no_grad():
        for length,args in data.items():
            samples={v:[] for v in VARIANTS}
            for v in VARIANTS:
                for _ in range(10):models[v](*args)
            for i in range(repeats):
                # Alternate order to reduce drift bias.
                for v in VARIANTS[::1 if i%2==0 else -1]:
                    if device.startswith('cuda'):torch.cuda.synchronize()
                    start=time.perf_counter();models[v](*args)
                    if device.startswith('cuda'):torch.cuda.synchronize()
                    samples[v].append(1000*(time.perf_counter()-start))
            for v,times in samples.items():
                rows.append(dict(variant=v,candidates=length,mean_ms=float(np.mean(times)),
                    p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),samples_ms=times))
    result=dict(scope='untrained fixed synthetic inputs; no solve-quality or end-to-end conclusion',
        device=device,repeats=repeats,batch_size=1,warmup=10,parameters=counts,rows=rows,
        stack=server.stack(ROOT),resources=server.resource_limits(),
        cuda_peak_allocated_combined_models_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    write_json(HERE/'docs'/('benchmark-'+device.replace(':','-')+'.json'),result)
    print(json.dumps(dict(parameters=counts,rows=[{k:v for k,v in r.items() if k!='samples_ms'} for r in rows]),indent=2))


def model_rows(variant,seed):
    c,run=protocol(variant);out=server.group_dir(run,'B0',seed)
    actor=out/'actor_best.pt'
    if not actor.exists():raise FileNotFoundError(actor)
    path=run/'test'/'official66'/sha256(actor)[:16]/'selected_results.json'
    rows=server.read(path)
    names=server.read(ROOT/'migration/2026-10-22/server-run/test66.json')
    expected={(n,s) for n in names for s in c['solver_seeds']}
    if len(rows)!=len(expected) or {(r['instance_name'],r['solver_seed']) for r in rows}!=expected:
        raise ValueError('Incomplete/duplicate final test coverage: '+str(path))
    if any(not r.get('valid') for r in rows):raise ValueError('Failed runs must be resolved/reported, not dropped')
    summary=server.read(out/'summary.json')
    if summary['protocol_sha256']!=fingerprint(c) or summary['episodes_seen']!=c['max_episodes']:
        raise ValueError('Training budget/protocol mismatch')
    return rows,out


def selection_manifest():
    records=[];stacks=[]
    for variant in VARIANTS:
        c,run=protocol(variant)
        for seed in c['training_seeds']:
            out=server.group_dir(run,'B0',seed)
            chosen=server.read(out/'selection.json');summary=server.read(out/'summary.json')
            actor=out/'actor_best.pt';policy,payload=load_actor(actor)
            if (payload['config']['server_protocol_sha256']!=fingerprint(c) or
                payload['config']['use_transformer']!=c['use_transformer'] or
                payload['config']['server_training_seed']!=seed or
                summary['episodes_seen']!=c['max_episodes'] or
                summary['protocol_sha256']!=fingerprint(c) or
                chosen['protocol_sha256']!=fingerprint(c) or
                payload['provenance']['checkpoint_sha256']!=chosen['checkpoint_sha256']):
                raise ValueError('Selection/model/protocol mismatch: '+str(out))
            records.append(dict(variant=variant,seed=seed,actor_sha256=sha256(actor),
                selection_sha256=sha256(out/'selection.json'),protocol_sha256=fingerprint(c),
                selected_episode=chosen['best']['episode']))
            stacks.append(chosen['stack'])
    if any(s!=stacks[0] for s in stacks):raise ValueError('Selections used different CPU/software')
    return dict(models=records,stack=stacks[0])


def freeze(verify=False):
    path=ROOT/'tgppo/runs/transformer-ablation-20261027/frozen_selections.json'
    current=selection_manifest()
    if path.exists():
        if server.read(path)!=current:raise ValueError('Frozen selections were modified')
    elif verify:raise FileNotFoundError('Run freeze after all six development selections: '+str(path))
    else:write_json(path,current)
    return current


def aggregate(rows,horizon):
    times=[t*1000 for r in rows for t in r.get('inference_seconds',[])]
    names={r['instance_name'] for r in rows}
    return dict(solve_time=float(np.mean([r['end_to_end_wall_seconds'] for r in rows])),
        worker_wall_seconds=float(np.mean([r.get('worker_wall_seconds',r['end_to_end_wall_seconds']) for r in rows])),
        nodes=float(np.mean([r['nnodes'] for r in rows])),
        fair_nodes=float(np.mean([r['fair_nodes'] for r in rows])),
        pdi=float(np.mean([r['pdi_horizon_midpoint'] for r in rows])),
        pdi_lower=float(np.mean([r['pdi_horizon_lower'] for r in rows])),
        pdi_upper=float(np.mean([r['pdi_horizon_upper'] for r in rows])),
        final_gap=float(np.mean([r['gap'] for r in rows])),
        gap_unknown_count=sum(not r['gap_finite'] for r in rows),
        solved=sum(bool(r['proof_completed']) and r['scip_solve_time']<=horizon for r in rows),
        solved_instances=sum(all(r['proof_completed'] and r['scip_solve_time']<=horizon
            for r in rows if r['instance_name']==n) for n in names),
        timeouts=sum(r['status']=='timelimit' for r in rows),
        inference_mean_ms=float(np.mean(times)) if times else None,
        inference_p50_ms=float(np.percentile(times,50)) if times else None,
        inference_p95_ms=float(np.percentile(times,95)) if times else None,
        feature_seconds=float(np.mean([r['callback_seconds']['features'] for r in rows])),
        callback_seconds=float(np.mean([r['callback_seconds']['callback'] for r in rows])))


def report():
    check_pair(); destination=HERE/'docs'; benchmark_path=destination/'benchmark-cpu.json'
    counts=server.read(benchmark_path)['parameters'] if benchmark_path.exists() else None
    records=[];missing=[];flat=[];stacks=[];history=[];stability=[]
    for variant in VARIANTS:
        c,run=protocol(variant)
        for seed in c['training_seeds']:
            try:
                rows,out=model_rows(variant,seed)
                stacks.append(server.read(out/'selection.json')['stack'])
                records.append(dict(variant=variant,seed=seed,**aggregate(rows,c['time_limit'])))
                flat.extend(dict(variant=variant,training_seed=seed,**r) for r in rows)
                updates=server.read(out/'updates.json')
                history.extend(dict(variant=variant,seed=seed,**u) for u in updates)
                actual=[u for u in updates if u.get('updated')]
                stability.append(dict(variant=variant,seed=seed,effective_updates=len(actual),
                    finite_losses=all(np.isfinite(u[k]) for u in actual for k in ('actor_loss','critic_loss','entropy')),
                    peak_gpu_allocated_bytes=max((u.get('peak_cuda_allocated_bytes',0) for u in updates),default=0),
                    peak_gpu_reserved_bytes=max((u.get('peak_cuda_reserved_bytes',0) for u in updates),default=0)))
            except FileNotFoundError as exc:missing.append(str(exc.filename or exc))
    if stacks and any(s!=stacks[0] for s in stacks):raise ValueError('CPU/software differs across evaluated models')
    summary=dict(complete=len(records)==6,completed_models=len(records),missing=missing,per_seed=records,
                 parameters=counts,training_updates=history,training_stability=stability,paired_test_rows=flat)
    if summary['complete']:
        freeze(verify=True)
        names=sorted({r['instance_name'] for r in flat});paired={};rng=np.random.RandomState(20261027)
        for key,field in [('solve_time','end_to_end_wall_seconds'),('nodes','nnodes'),
                          ('fair_nodes','fair_nodes'),('pdi','pdi_horizon_midpoint')]:
            values={v:np.array([np.mean([r[field] for r in flat if r['variant']==v and r['instance_name']==n])
                                for n in names]) for v in VARIANTS}
            deltas=values['bimatching']-values['transformer']
            boots=deltas[rng.randint(0,len(names),size=(5000,len(names)))].mean(1)
            paired[key]=dict(mean_delta=float(deltas.mean()),bootstrap_95_low=float(np.percentile(boots,2.5)),
                bootstrap_95_high=float(np.percentile(boots,97.5)),instances=len(names),
                scope='instance-resampled paired interval conditional on these three training seeds')
        summary['paired_instance_comparison']=paired
    write_json(destination/'comparison.json',summary)
    labels=[('Parameters (Actor)','actor'),('Parameters (Actor+Critic)','total'),
        ('Inference Latency (ms)','inference_mean_ms'),('Inference P50 (ms)','inference_p50_ms'),
        ('Inference P95 (ms)','inference_p95_ms'),('Solve Time (s)','solve_time'),
        ('Worker wall incl. checkpoint load (s)','worker_wall_seconds'),('Nodes','nodes'),
        ('Fair Nodes','fair_nodes'),('Primal-Dual Integral','pdi'),('Final Gap','final_gap'),
        ('Solved Instances (all solver seeds; /66)','solved_instances'),
        ('Solved runs (/198)','solved'),('Timeout runs','timeouts'),
        ('Feature extraction per instance (s)','feature_seconds'),('Branch callback per instance (s)','callback_seconds')]
    lines=['# Transformer 消融结果','',
        '正式对照完成。' if summary['complete'] else '正式对照尚未完成；未测量项不能用于推断性能。',
        '', '| Metric | Transformer + BiMatching | BiMatching Only | Relative Change |',
        '|---|---:|---:|---:|']
    for label,key in labels:
        values=[]
        for variant in VARIANTS:
            if key in ('actor','total') and counts:
                values.append(counts[variant]['actor'] if key=='actor' else sum(counts[variant].values()))
            elif summary['complete']:
                vals=[r[key] for r in records if r['variant']==variant]
                values.append(float(np.mean(vals)) if all(v is not None for v in vals) else None)
            else:values.append(None)
        delta=100*(values[1]/values[0]-1) if None not in values and values[0]!=0 else None
        fmt=lambda v:'未测量' if v is None else '{:.6g}'.format(v)
        lines.append('| {} | {} | {} | {} |'.format(label,fmt(values[0]),fmt(values[1]),
            '未测量' if delta is None else '{:+.2f}%'.format(delta)))
    lines+=['','以实例为统计单位，先平均求解器种子，再按训练种子报告均值/样本标准差。',
        'Solved Instances 为三个求解器种子都在时限内证明完成的不同实例数，Solved runs 为198次任务中的完成数。',
        '超时全部保留；Final Gap 保留 SCIP sentinel，并同时报告未知 gap 数。PDI 为 percent-seconds，保留上下界。',
        '纯网络延迟跨调用汇总，实际访问候选集合可能不同；固定输入微基准另见 benchmark-cpu.json。','']
    if records:
        lines+=['| Variant | Metric | Mean | SD across training seeds |','|---|---|---:|---:|']
        for variant in VARIANTS:
            for key in ('solve_time','nodes','fair_nodes','pdi','final_gap','solved','timeouts','inference_mean_ms','gap_unknown_count'):
                values=[r[key] for r in records if r['variant']==variant and r[key] is not None]
                if values:lines.append('| {} | {} | {:.6g} | {} |'.format(variant,key,np.mean(values),
                    '{:.6g}'.format(statistics.stdev(values)) if len(values)>1 else '未测量'))
    if summary['complete']:
        lines+=['','| Paired metric (BM minus Transformer) | Mean delta | Instance bootstrap 95% interval |',
                '|---|---:|---:|']
        for key,r in summary['paired_instance_comparison'].items():
            lines.append('| {} | {:.6g} | [{:.6g}, {:.6g}] |'.format(key,r['mean_delta'],r['bootstrap_95_low'],r['bootstrap_95_high']))
        lines+=['','节点数必须结合超时、Gap/PDI和吞吐判断；相同时限下节点更少本身不等于策略更优。',
            '本实验同时移除 Actor/Critic Transformer，参数容量及 encoder 内部归一化也减少；不能仅凭总结果归因于交互机制。']
    if stability:
        lines+=['','| Variant | Seed | Effective PPO updates | Finite losses | Peak GPU allocated (GiB) |',
                '|---|---:|---:|---|---:|']
        for row in stability:
            lines.append('| {} | {} | {} | {} | {:.6g} |'.format(row['variant'],row['seed'],row['effective_updates'],
                row['finite_losses'],row['peak_gpu_allocated_bytes']/1024**3))
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if flat:
        fields=['variant','training_seed','instance_name','solver_seed','status','valid','proof_completed',
                'end_to_end_wall_seconds','nnodes','fair_nodes','pdi_horizon_midpoint','gap','gap_finite','inference_mean_ms']
        with (destination/'per_instance.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(complete=summary['complete'],completed_models=len(records),missing=missing),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','online-check','benchmark','smoke','prepare','train','select','freeze','evaluate','report'])
    parser.add_argument('variant',nargs='?',choices=VARIANTS)
    parser.add_argument('seed',nargs='?',type=int,choices=[11,22,33])
    parser.add_argument('--device',default='cpu');parser.add_argument('--repeats',type=int,default=30)
    args=parser.parse_args();check_pair()
    if args.command=='online-check':online_check();return
    if args.command=='benchmark':benchmark(args.device,args.repeats);return
    if args.command=='report':report();return
    if args.command=='freeze':freeze();return
    if args.command in ('prepare','smoke'):
        lock_path=ROOT/'tgppo/runs/transformer-ablation-20261027'/('.'+args.command+'.lock')
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            (prepare if args.command=='prepare' else smoke)()
        return
    if args.variant is None:parser.error('variant required')
    c,run=protocol(args.variant)
    if args.command=='preflight':
        print(json.dumps(server.preflight(ROOT,c,args.device),indent=2));return
    if args.seed is None:parser.error('training seed required')
    # Both selection and evaluation use the same CPU/SCIP/software as normalization.
    server.baseline_bundle(ROOT,c,run)
    if args.command=='train':server.train(ROOT,c,run,'B0',args.seed,True,'cuda')
    elif args.command=='select':
        if (ROOT/'tgppo/runs/transformer-ablation-20261027/frozen_selections.json').exists():
            freeze(verify=True)
        else:server.select(ROOT,c,run,'B0',args.seed)
    elif args.command=='evaluate':
        freeze(verify=True)
        out=server.group_dir(run,'B0',args.seed)
        names=server.read(ROOT/'migration/2026-10-22/server-run/test66.json')
        server.evaluate(ROOT,c,run,out/'actor_best.pt',ROOT/'tgppo/datasets/test_instances',
                        names,'official66',args.seed)


if __name__=='__main__':main()
