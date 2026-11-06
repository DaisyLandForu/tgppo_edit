"""Cross-check the local intervention and derive instance-level summaries."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]


def read(name):return list(csv.DictReader((OUT/name).open()))


def median(rows,key):return statistics.median(float(r[key]) for r in rows)


def main():
    d=json.loads((OUT/'local_validation_summary.json').read_text());plan=d['plan']
    rows=read('policy_metrics.csv');pairs=read('paired_intervention.csv');grads=read('score_gradients.csv')
    inv=read('invariance_checks.csv');obs=read('observation_manifest.csv')
    assert len(rows)==len(obs)*8 and len(pairs)==len(obs)*4
    assert len(obs)<=200 and len({r['instance'] for r in obs})==plan['instances']
    assert all(r['split']=='train' and r['overlap_with_previous_pool']=='False' for r in obs)
    assert all(int(r['candidate_count'])>=1 for r in obs)
    assert all(math.isfinite(float(r['logit_spread'])) and math.isfinite(float(r['tv_to_uniform'])) for r in rows)
    assert all(float(r[k])<=1e-6 for r in inv for k in ['repeat_max_error','permutation_max_error','padding_max_error'])
    assert all(float(r['pad_probability'])==0 for r in inv)
    before=json.loads((OUT/'runtime_sources_before.json').read_text());after=json.loads((OUT/'runtime_sources_after.json').read_text())
    assert before==after
    for p,h in before.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    # The independent original actor/module files must be literal source copies.
    for name in ['actor.py','modules.py']:
        source=ROOT/'tgppo/experiments/server-20261022/code/project/policy'/name
        assert source.read_bytes()==(OUT/'model_copy'/name).read_bytes()
    instance_rows=[];cp_rows=[]
    for model in ['B0-s11','B0-s22']:
        for ep in ['100','500']:
            for name in sorted({r['instance'] for r in rows}):
                rr=[r for r in rows if r['model']==model and r['episode']==ep and r['instance']==name and int(r['L'])>1]
                a=[r for r in rr if r['variant']=='original'];b=[r for r in rr if r['variant']=='ln2_affine_only']
                if not a:continue
                instance_rows.append(dict(model=model,episode=ep,instance=name,states=len(a),
                                          original_median_tv_uniform=median(a,'tv_to_uniform'),
                                          intervention_median_tv_uniform=median(b,'tv_to_uniform'),
                                          original_median_logit_spread=median(a,'logit_spread'),
                                          intervention_median_logit_spread=median(b,'logit_spread')))
            pp=[r for r in pairs if r['model']==model and r['episode']==ep]
            aa=[r for r in rows if r['model']==model and r['episode']==ep and r['variant']=='original' and int(r['L'])>1]
            bb=[r for r in rows if r['model']==model and r['episode']==ep and r['variant']=='ln2_affine_only' and int(r['L'])>1]
            cp_rows.append(dict(model=model,episode=ep,states=len(aa),original_median_tv_uniform=median(aa,'tv_to_uniform'),
                                intervention_median_tv_uniform=median(bb,'tv_to_uniform'),
                                intervention_max_tv_uniform=max(float(r['tv_to_uniform']) for r in bb),
                                median_tv_change=median(pp,'tv_original_to_intervention'),
                                max_tv_change=max(float(r['tv_original_to_intervention']) for r in pp),
                                original_relu_all_zero_states=sum(r['head2_all_candidates_zero']=='True' for r in aa),
                                intervention_relu_all_zero_states=sum(r['head2_all_candidates_zero']=='True' for r in bb),
                                original_median_saturation_fraction=median(aa,'normalization_amplitude_ge_099_fraction')))
    for filename,data in [('per_instance_intervention.csv',instance_rows),('checkpoint_intervention_summary.csv',cp_rows)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    grad_summary=[]
    for model in ['B0-s11','B0-s22']:
        for ep in ['100','500']:
            for variant in ['original','ln2_affine_only']:
                for module in ['embedding','transformer','bimatching','treegate']:
                    rs=[r for r in grads if r['model']==model and r['episode']==ep and r['variant']==variant and r['module']==module]
                    grad_summary.append(dict(model=model,episode=ep,variant=variant,module=module,
                                             cases=len(rs),median_l2=median(rs,'l2_norm'),
                                             min_l2=min(float(r['l2_norm']) for r in rs),
                                             max_l2=max(float(r['l2_norm']) for r in rs),
                                             exactly_zero_cases=sum(float(r['l2_norm'])==0 for r in rs)))
    with (OUT/'gradient_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(grad_summary[0]));w.writeheader();w.writerows(grad_summary)
    bench=json.loads((OUT/'microbenchmark_summary.json').read_text());raw=read('microbenchmark_raw.csv')
    assert bench['completed_states']==bench['selected_states'] and len(raw)==bench['completed_states']*20*4
    assert bench['weights_unchanged'] and not bench['cuda_initialized'] and not bench['true_solver_feature_extraction_measured']
    budgets=json.loads((OUT/'compute_budget.json').read_text());prior=sum(r.get('elapsed_seconds',0) for r in budgets['runs'])
    assert prior<590
    validation=dict(status='passed',observations=len(obs),instances=plan['instances'],policy_forward_records=len(rows),
                    intervention_pairs=len(pairs),gradient_cases=len(grads)//4,invariance_cases=len(inv),
                    benchmark_states=bench['completed_states'],benchmark_raw_measurements=len(raw),
                    baseline_source_files_unchanged=len(before),source_copies_match=True,source_and_checkpoint_audit_passed=True,
                    no_optimizer_or_new_solver=True,material_nonuniform_threshold=.01,
                    states_above_nonuniform_threshold=sum(float(r['tv_to_uniform'])>=.01 for r in rows if r['variant']=='ln2_affine_only'),
                    caveat='No training or policy-quality labels; offline intervention does not establish solver benefit.')
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,checkpoints=cp_rows,gradients=grad_summary,bench=bench['summary']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
