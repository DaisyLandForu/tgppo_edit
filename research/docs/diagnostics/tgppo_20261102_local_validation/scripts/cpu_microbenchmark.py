"""Single-thread CPU microbenchmarks on saved states; no SCIP or optimizer."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
from collections import defaultdict

START=time.perf_counter()
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261102)
sys.path.insert(0,str(OUT))
from model_copy.actor import Actor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def q(v):
    return dict(zip(['min','p50','p95','max'],map(float,np.quantile(v,[0,.5,.95,1]))))


def main():
    source=ROOT/'tgppo/experiments/server-20261022/code/project/agent/agent.py'
    source_sha=sha(source);text=source.read_text()
    cls=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='ActorPolicy')
    copied='import time\nimport torch\nfrom torch.distributions import Categorical\n\n'+ast.get_source_segment(text,cls)+'\n'
    policy_path=OUT/'model_copy/policy_path.py';policy_path.write_text(copied)
    from model_copy.policy_path import ActorPolicy
    path=ROOT/'tgppo/runs/server6-20261022/training/B0-s11/episode-0100.pt'
    hash_before=sha(path);load_start=time.perf_counter();cp=torch.load(path,map_location='cpu');cfg=cp['config']
    a=Actor(**cfg['state_dims'],**{k:cfg[k] for k in ['hidden_dim','num_heads','num_layers','dropout']})
    a.load_state_dict(cp['actor']);a.eval();before={k:v.clone() for k,v in a.state_dict().items()};del cp
    policy=ActorPolicy(a);load_seconds=time.perf_counter()-load_start
    obs=pickle.loads((OUT/'observations.pkl').read_bytes())
    bins=[('2-16',2,16),('17-64',17,64),('65-256',65,256),('257+',257,float('inf'))]
    selected=[]
    for label,low,high in bins:
        pool=sorted([(i,key,name,r) for i,(key,name,r) in enumerate(obs) if low<=len(r['cands_state'])<=high],
                    key=lambda x:(len(x[3]['cands_state']),x[1]))
        for index in sorted(set(np.linspace(0,len(pool)-1,min(4,len(pool)),dtype=int).tolist())) if pool else []:
            selected.append((label,pool[index]))
    write('benchmark_selection.csv',[dict(bin=label,index=i,observation=key,instance=name,L=len(r['cands_state']))
                                      for label,(i,key,name,r) in selected])
    raw=[];first=[];stages=[];done=[]
    for label,(i,key,name,r) in selected:
        if time.perf_counter()-START>85:break
        def convert():return policy._inputs(r['cands_state'],r['mip_state'],r['node_state'])
        def full():return policy.act(r['cands_state'],r['mip_state'],r['node_state'])
        t=time.perf_counter_ns();full();initial_ms=(time.perf_counter_ns()-t)/1e6
        first.append(dict(observation=key,instance=name,bin=label,L=len(r['cands_state']),first_state_act_ms=initial_ms))
        prepared=convert();c,m,n,mask=prepared
        with torch.no_grad():
            probs=a(c,n,m,mask)
            def forward():return a(c,n,m,mask)
            def sampling():
                if not torch.isfinite(probs).all():raise ValueError('Nonfinite probabilities')
                dist=torch.distributions.Categorical(probs);action=dist.sample();lp=dist.log_prob(action)
                return int(action.item()),float(lp.item())
            components={'input_conversion_checks':convert,'pure_actor_forward':forward,
                        'distribution_sample_return':sampling,'complete_actor_policy_call':full}
            for _ in range(3):full()
            rng=np.random.RandomState(20261102+i)
            for repeat in range(20):
                for component in rng.permutation(list(components)):
                    t=time.perf_counter_ns();components[component]();elapsed=(time.perf_counter_ns()-t)/1e6
                    raw.append(dict(observation=key,instance=name,bin=label,L=len(r['cands_state']),
                                    repeat=repeat,component=component,milliseconds=elapsed))
            # Separate instrumented forwards; do not mix their overhead into the
            # uninstrumented microbenchmark distributions above.
            starts={};times=defaultdict(list);handles=[]
            modules={'var_embedding':a.var_embedding,'tree_embedding':a.tree_embedding,
                     'fusion_linear':a.global_embedding,'transformer':a.transformer,
                     'bimatching':a.tree_refinement,'treegate_head':a.output_layer}
            for module_name,module in modules.items():
                def pre(mod,inp,name=module_name):starts[name]=time.perf_counter_ns()
                def post(mod,inp,result,name=module_name):times[name].append((time.perf_counter_ns()-starts[name])/1e6)
                handles.append(module.register_forward_pre_hook(pre));handles.append(module.register_forward_hook(post))
            totals=[]
            for _ in range(3):
                t=time.perf_counter_ns();forward();totals.append((time.perf_counter_ns()-t)/1e6)
            for h in handles:h.remove()
            for module_name,times_ms in times.items():
                stages.append(dict(observation=key,bin=label,L=len(r['cands_state']),module=module_name,
                                   mean_milliseconds=float(np.mean(times_ms)),
                                   fraction_of_instrumented_forward=float(np.mean(times_ms)/np.mean(totals))))
        done.append(key)
        write('microbenchmark_raw.csv',raw);write('microbenchmark_first_calls.csv',first)
        write('forward_module_times.csv',stages)
        print('benchmarked',key,'L',len(r['cands_state']),'elapsed',time.perf_counter()-START,flush=True)
    summary_rows=[]
    for label,_,_ in bins:
        for component in ['input_conversion_checks','pure_actor_forward','distribution_sample_return','complete_actor_policy_call']:
            rs=[r for r in raw if r['bin']==label and r['component']==component]
            if not rs:continue
            summary_rows.append(dict(bin=label,component=component,states=len({r['observation'] for r in rs}),calls=len(rs),
                                     mean_ms=float(np.mean([r['milliseconds'] for r in rs])),
                                     **{k+'_ms':v for k,v in q([r['milliseconds'] for r in rs]).items()}))
    write('microbenchmark_summary.csv',summary_rows)
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    assert sha(path)==hash_before and sha(source)==source_sha
    assert not torch.cuda.is_initialized() and 'pyscipopt' not in sys.modules
    summary=dict(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=hash_before,weights_unchanged=True,
                 original_policy_class_source=str(source.relative_to(ROOT)),original_policy_source_sha256=source_sha,
                 extracted_class_exact=True,selected_states=len(selected),completed_states=len(done),
                 warmup_calls_per_state=3,measured_repeats_per_component=20,
                 raw_measurements=len(raw),summary=summary_rows,model_load_seconds=load_seconds,
                 affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
                 torch_interop_threads=torch.get_num_interop_threads(),cuda_initialized=False,new_solver_tasks=0,
                 true_solver_feature_extraction_measured=False,
                 limitation='Conditional warmed replay states; not historical online latency or first-call-per-fresh-process distribution.',
                 elapsed_seconds=time.perf_counter()-START)
    (OUT/'microbenchmark_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
