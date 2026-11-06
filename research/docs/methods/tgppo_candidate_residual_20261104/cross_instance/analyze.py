"""Read-only frozen-cohort paired analysis; incomplete matrices are not pooled."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/'tgppo/runs/candidate-residual-20261104'
RUN=PARENT/'cross-instance-8'
evidence={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()

def read(path):
    evidence[str(Path(path).relative_to(ROOT))]=sha(path)
    return json.loads(Path(path).read_text())

def save(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def csv_write(name,rows):
    if rows:
        with (HERE/name).open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean(rows,key):return stats.mean(r[key] for r in rows)

def summarize(rows):
    return dict(tasks=len(rows),**{k:mean(rows,k) for k in ('pdi','pdi_lower','pdi_upper','nodes',
        'solve_seconds','worker_seconds')},proof_rate=mean(rows,'proof_within120'),
        actor_calls=sum(r['branch_count'] for r in rows),no_actor_tasks=sum(r['branch_count']==0 for r in rows))

def main():
    status=read(RUN/'status.json');assert status['status']!='running'
    cohort=read(RUN/'cohort.json');manifest=read(RUN/'manifest.json')
    assert sha(RUN/'cohort.json')==manifest['cohort_sha256']
    assert sha(HERE/'run.py')==manifest['runner_sha256']
    assert status['charged_wall_seconds']<=5400
    for m in manifest['models'].values():
        for key in ('actor','frozen_actor','checkpoint'):
            assert sha(m[key])==m['checkpoint_sha256' if key=='checkpoint' else 'actor_sha256']
    package=ROOT/'tgppo/experiments/candidate-residual-20261104'
    for rel,digest in read(package/'docs/bundle_manifest.json')['files'].items():assert sha(package/rel)==digest
    for rel,digest in read(package/'docs/protected_sources.json').items():assert sha(ROOT.parent/rel)==digest
    for rel,digest in read(PARENT/'results/artifact_hashes.json').items():assert sha(PARENT/'results'/rel)==digest
    assert sha(cohort['selection_source'])==cohort['selection_source_sha256']
    for item in cohort['selected']:assert sha(item['path'])==item['sha256']
    rows=[];missing=[]
    for task in manifest['tasks']:
        path=RUN/'raw'/(task['job']['job_id']+'.json')
        if not path.exists():missing.append(dict(index=task['index'],job_id=task['job']['job_id'],status='missing'));continue
        r=read(path)
        if not r.get('valid'):missing.append(dict(index=task['index'],job_id=task['job']['job_id'],status=r['status']));continue
        assert r['sampling_seed']==task['job']['sampling_seed'] and not r['cuda_initialized']
        assert r['cpu_affinity']==manifest['cpu_affinity'] and r['time_limit_seconds']==120
        if r['mode']=='eval':assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
        assert r['pdi_horizon_lower']<=r['pdi_horizon_midpoint']<=r['pdi_horizon_upper']
        rows.append(dict(variant=task['variant'],training_seed=task['training_seed'],instance=r['instance_name'],
            stratum=task['stratum'],solver_seed=r['solver_seed'],pdi=r['pdi_horizon_midpoint'],
            pdi_lower=r['pdi_horizon_lower'],pdi_upper=r['pdi_horizon_upper'],nodes=r['nnodes'],
            solve_seconds=r['scip_solve_time'],worker_seconds=r['worker_wall_seconds'],
            proof_within120=bool(r['proof_completed'] and r['scip_solve_time']<=120),
            branch_count=r.get('branch_count',0),actor_seconds=r.get('callback_seconds',{}).get('actor',0),
            job_id=r['job_id'],source=str(path.relative_to(ROOT))))
    csv_write('evaluation_rows.csv',rows);csv_write('missing_or_invalid.csv',missing)
    if missing:
        save('verification.json',dict(passed=False,complete=False,valid=len(rows),missing_or_invalid=missing,
            charged_wall_seconds=status['charged_wall_seconds']))
        print(json.dumps(dict(complete=False,valid=len(rows),missing_or_invalid=missing)));return
    assert len(rows)==112 and status['status']=='complete'
    names=[r['instance'] for r in cohort['selected']]
    inactive=[n for n in names if all(r['branch_count']==0 for r in rows if r['instance']==n and r['variant']!='native')]
    active=[n for n in names if n not in inactive]
    keys=[('native',None)]+[(v,s) for v in ('b0','score','candidate') for s in (11,22)]
    panels=[];instances=[];paired=[];contributions=[]
    for variant,seed in keys:
        rr=[r for r in rows if r['variant']==variant and r['training_seed']==seed]
        assert len(rr)==16 and {(r['instance'],r['solver_seed']) for r in rr}=={(n,s) for n in names for s in (0,1)}
        for n in names:
            instances.append(dict(variant=variant,training_seed=seed,instance=n,**summarize([r for r in rr if r['instance']==n])))
        for panel,subset in [('all',names),('actor_participating',active)]:
            if subset:panels.append(dict(variant=variant,training_seed=seed,panel=panel,instances=len(subset),
                **summarize([r for r in rr if r['instance'] in subset])))
    for n in names:
        means={v:{k:stats.mean(r[k] for r in instances if r['instance']==n and r['variant']==v)
                  for k in ('pdi','nodes','solve_seconds','worker_seconds','proof_rate')}
               for v in ('native','b0','score','candidate')}
        contributions.append(dict(instance=n,actor_participates=n in active,
            **{v+'_'+k:val for v,m in means.items() for k,val in m.items()},
            cr_vs_b0_pdi_percent=100*(means['candidate']['pdi']/means['b0']['pdi']-1),
            cr_vs_score_pdi_percent=100*(means['candidate']['pdi']/means['score']['pdi']-1),
            contribution_to_all_mean=(means['candidate']['pdi']-means['b0']['pdi'])/8))
        for seed in (11,22):
            values={v:next(r for r in instances if r['instance']==n and r['variant']==v and r['training_seed']==seed)
                    for v in ('b0','score','candidate')}
            for reference in ('b0','score'):
                a,b=values['candidate'],values[reference]
                paired.append(dict(instance=n,training_seed=seed,reference=reference,
                    actor_participates=n in active,pdi_delta=a['pdi']-b['pdi'],
                    pdi_percent=100*(a['pdi']/b['pdi']-1),
                    delta_lower=a['pdi_lower']-b['pdi_upper'],delta_upper=a['pdi_upper']-b['pdi_lower'],
                    nodes_percent=100*(a['nodes']/b['nodes']-1) if b['nodes'] else None,
                    solve_seconds_percent=100*(a['solve_seconds']/b['solve_seconds']-1)))
    differences=[]
    for panel in ('all','actor_participating'):
        for seed in (11,22):
            values={v:next(r for r in panels if r['panel']==panel and r['variant']==v and r['training_seed']==seed)
                    for v in ('b0','score','candidate')}
            for ref in ('b0','score'):
                a,b=values['candidate'],values[ref]
                differences.append(dict(panel=panel,training_seed=seed,reference=ref,
                    pdi_delta=a['pdi']-b['pdi'],pdi_percent=100*(a['pdi']/b['pdi']-1),
                    delta_lower=a['pdi_lower']-b['pdi_upper'],delta_upper=a['pdi_upper']-b['pdi_lower'],
                    proof_rate_delta=a['proof_rate']-b['proof_rate']))
    aggregate={p:{v:stats.mean(r['pdi'] for r in panels if r['panel']==p and r['variant']==v)
                 for v in ('native','b0','score','candidate')} for p in ('all','actor_participating')}
    csv_write('panel_summary.csv',panels);csv_write('instance_model_means.csv',instances)
    csv_write('instance_summary.csv',contributions);csv_write('paired_instances.csv',paired)
    csv_write('paired_panels.csv',differences)
    result=dict(complete=True,instances=names,independent_instances=8,model_tasks=96,native_tasks=16,
        panels=panels,aggregate=aggregate,differences=differences,instance_summary=contributions,
        no_actor_instances=inactive,charged_wall_seconds=status['charged_wall_seconds'],
        summed_worker_wall_seconds=status['summed_worker_wall_seconds'],stack=manifest['stack'],
        scope=cohort['scope'],selection_rule=cohort['selection_rule'])
    save('analysis.json',result)
    save('verification.json',dict(passed=True,complete=True,rows=112,independent_instances=8,
        all_frozen_models_unchanged=True,old_sources_and_results_unchanged=True,actor_only_cpu=True,
        workers=4,training_steps=0,charged_wall_seconds=status['charged_wall_seconds'],
        cohort_fixed_before_current_outcomes=True,source_files_hashed=len(evidence)))
    save('input_hashes.json',evidence)
    print(json.dumps({k:result[k] for k in ('aggregate','differences','no_actor_instances','charged_wall_seconds')},indent=2))

if __name__=='__main__':main()
