"""Read-only paired analysis; retain old/new solver-seed cohorts separately."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/'tgppo/runs/candidate-residual-20261104'
RUN=PARENT/'cpu-retest-seeds23'
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
    if not rows:return
    with (HERE/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    status=read(RUN/'status.json');manifest=read(RUN/'manifest.json')
    records=read(RUN/'completed.json');assert status['status']!='running'
    assert status['charged_wall_seconds']<=600
    for m in manifest['models'].values():
        for key in ('actor','frozen_actor','checkpoint'):
            assert sha(m[key])==m['checkpoint_sha256' if key=='checkpoint' else 'actor_sha256']
    package=ROOT/'tgppo/experiments/candidate-residual-20261104'
    for name,digest in read(package/'docs/bundle_manifest.json')['files'].items():
        assert sha(package/name)==digest
    for name,digest in read(package/'docs/protected_sources.json').items():
        assert sha(ROOT.parent/name)==digest
    for name,digest in read(PARENT/'results/artifact_hashes.json').items():
        assert sha(PARENT/'results'/name)==digest
    rows=[]
    for item in records:
        r=read(item['result']);task=manifest['tasks'][item['index']]
        assert r['valid'] and not r['cuda_initialized'] and r['cpu_affinity']==manifest['cpu_affinity']
        assert r['sampling_seed']==task['job']['sampling_seed'] and r['time_limit_seconds']==120
        if item['variant']!='native':
            assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
        rows.append(dict(variant=item['variant'],training_seed=item['training_seed'],
            instance=r['instance_name'],solver_seed=r['solver_seed'],pdi=r['pdi_horizon_midpoint'],
            pdi_lower=r['pdi_horizon_lower'],pdi_upper=r['pdi_horizon_upper'],nodes=r['nnodes'],
            solve_seconds=r['scip_solve_time'],worker_seconds=r['worker_wall_seconds'],
            orchestration_seconds=item['elapsed_seconds'],proof=r['proof_completed'],
            branch_count=r.get('branch_count',0)))
    pairs=[];model_means=[];instance_means=[]
    for name in manifest['instances']:
        for seed in (11,22):
            for ss in (2,3):
                group={v:[r for r in rows if r['variant']==v and r['instance']==name
                          and r['training_seed']==seed and r['solver_seed']==ss] for v in ('b0','score','candidate')}
                if not all(group.values()):continue
                group={v:r[0] for v,r in group.items()}
                pair=dict(instance=name,training_seed=seed,solver_seed=ss)
                for v,r in group.items():
                    for metric in ('pdi','nodes','solve_seconds','worker_seconds'):pair[v+'_'+metric]=r[metric]
                for ref in ('b0','score'):
                    for metric in ('pdi','nodes','solve_seconds','worker_seconds'):
                        pair['cr_vs_'+ref+'_'+metric+'_percent']=100*(group['candidate'][metric]/group[ref][metric]-1)
                pairs.append(pair)
        for v in ('native','b0','score','candidate'):
            for seed in ((None,) if v=='native' else (11,22)):
                chosen=[r for r in rows if r['variant']==v and r['instance']==name and r['training_seed']==seed]
                if not chosen:continue
                model_means.append(dict(instance=name,variant=v,training_seed=seed,tasks=len(chosen),
                    **{k:stats.mean(r[k] for r in chosen) for k in ('pdi','nodes','solve_seconds','worker_seconds')}))
        means={v:{k:stats.mean(r[k] for r in rows if r['variant']==v and r['instance']==name)
                  for k in ('pdi','nodes','solve_seconds','worker_seconds')}
               for v in ('native','b0','score','candidate') if any(r['variant']==v and r['instance']==name for r in rows)}
        if set(means)=={'native','b0','score','candidate'}:
            metrics={}
            for v,r in means.items():
                for k,value in r.items():metrics[v+'_'+k]=value
            pp=[r for r in pairs if r['instance']==name]
            for ref in ('b0','score'):
                for k in ('pdi','nodes','solve_seconds','worker_seconds'):
                    metrics['cr_vs_'+ref+'_'+k+'_percent']=100*(means['candidate'][k]/means[ref][k]-1)
                    metrics['cr_vs_'+ref+'_'+k+'_wins']=sum(r['candidate_'+k]<r[ref+'_'+k] for r in pp)
            instance_means.append(dict(instance=name,paired_model_repeats=len(pp),**metrics))
    complete=status['status']=='complete' and len(rows)==28
    if complete:
        assert len(pairs)==8 and sum(r['variant']=='native' for r in rows)==4
        assert {(r['instance'],r['solver_seed']) for r in rows if r['variant']=='native'}=={
            (n,s) for n in manifest['instances'] for s in (2,3)}
        assert all(r['proof'] for r in rows)
    csv_write('evaluation_rows.csv',rows);csv_write('paired.csv',pairs)
    csv_write('model_means.csv',model_means);csv_write('instance_means.csv',instance_means)
    result=dict(complete=complete,completed_tasks=len(rows),requested_tasks=28,
        charged_wall_seconds=status['charged_wall_seconds'],training_steps=0,solver_seeds=[2,3],
        instances=manifest['instances'],stack=manifest['stack'],same_cpu_as_previous_evaluation=
        manifest['stack']['cpu_model']==manifest['original_evaluation_stack']['cpu_model'],
        instance_means=instance_means,model_means=model_means)
    save('analysis.json',result)
    save('verification.json',dict(passed=True,complete_matrix=complete,valid_rows=len(rows),
        frozen_models_unchanged=True,package_and_baseline_unchanged=True,old_results_unchanged=True,
        cpu_actor_only=True,single_cpu_affinity=manifest['cpu_affinity'],budget_seconds=600,
        charged_wall_seconds=status['charged_wall_seconds'],training_steps=0,source_files_hashed=len(evidence)))
    save('input_hashes.json',evidence)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
