"""Read-only review of completed method runs; no training or SCIP imports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/candidate-residual-20261104'
PACKAGE=ROOT/'tgppo/experiments/candidate-residual-20261104'
VARIANTS=('b0','score','candidate')
SEEDS=(11,22)
evidence={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    path=Path(path);evidence[str(path.relative_to(ROOT))]=sha(path)
    return json.loads(path.read_text())

def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def csv_write(name,rows):
    with (HERE/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean(rows,key):return stats.mean(r[key] for r in rows)

def summarize(rows):
    names=sorted({r['instance_name'] for r in rows})
    result={k:stats.mean(stats.mean(r[src] for r in rows if r['instance_name']==n) for n in names)
        for k,src in [('pdi','pdi_horizon_midpoint'),('pdi_lower','pdi_horizon_lower'),
            ('pdi_upper','pdi_horizon_upper'),('nodes','nnodes'),('solve_seconds','scip_solve_time'),
            ('worker_seconds','worker_wall_seconds')]}
    result.update(instances=len(names),tasks=len(rows),
        proof_rate=stats.mean(r['proof_completed'] and r['scip_solve_time']<=120 for r in rows),
        no_actor=sum(r.get('branch_count',0)==0 for r in rows))
    return result

def main():
    # Preserve the generated raw report/CSVs: record and independently recompute.
    release=read(PACKAGE/'docs/bundle_manifest.json')
    for rel,digest in release['files'].items():assert sha(PACKAGE/rel)==digest,rel
    flex=read(PACKAGE/'cpu-flex/verification.json')
    for rel,digest in flex['files'].items():assert sha(PACKAGE/'cpu-flex'/rel)==digest,rel
    protected=read(PACKAGE/'docs/protected_sources.json')
    for rel,digest in protected.items():assert sha(ROOT.parent/rel)==digest,rel
    for path,digest in read(RUN/'results/artifact_hashes.json').items():
        assert sha(RUN/'results'/path)==digest,path
        evidence[str((RUN/'results'/path).relative_to(ROOT))]=digest
    manifest=read(RUN/'evaluation/episode-0100/frozen_models.json')
    assert manifest['episode']==100
    names=manifest['development'];solver_seeds=manifest['solver_seeds']
    expected={(n,s) for n in names for s in solver_seeds}
    initialization={};training=[];all_rows={};native=read(RUN/'evaluation/episode-0100/native_rows.json')
    all_rows['native']=native
    for m in manifest['models']:
        for key in ('checkpoint','actor'):
            p=Path(m[key]);assert sha(p)==m[key+'_sha256'],str(p)
            evidence[str(p.relative_to(ROOT))]=m[key+'_sha256']
        v,s=m['variant'],m['seed'];folder=RUN/v/'training'/('B0-s'+str(s))
        status=read(RUN/'jobs'/('{}-s{}.json'.format(v,s)))
        summary=read(folder/'summary.json');history=read(folder/'updates.json')
        init=read(folder/'initialization.json');initialization[v,s]=init
        assert status['status']==summary['status']=='training_complete'
        assert summary['episodes_seen']==summary['trained_episodes']==100
        assert summary['pending_transitions']==0 and len(history)==25
        assert all(h['updated'] for h in history)
        assert sum(h['num_updates'] for h in history)==summary['optimizer_steps']==m['logical_minibatch_updates']
        assert all(math.isfinite(h[k]) for h in history for k in ('actor_loss','critic_loss','entropy'))
        assert sha(folder/'latest.pt')==summary['final_checkpoint_sha256']
        schedule=read(folder/'task_schedule.json')
        assert len(schedule)==100 and len(set(schedule))==20
        assert all(schedule.count(n)==5 for n in set(schedule))
        if v!='b0':assert schedule==read(RUN/'b0/training'/('B0-s'+str(s))/'task_schedule.json')
        preflight=read(folder/'preflight.json')
        if v=='b0' and s==11:context=RUN/'b0'
        else:
            contexts=[p.parent for p in (RUN/'cpu-flex/contexts'/('{}-s{}'.format(v,s))).glob('*/environment.json')]
            assert len(contexts)==1;context=contexts[0]
        normalization=read(context/'normalization/baselines.json')
        assert normalization['stack']==preflight['stack']
        assert normalization['protocol_sha256']==summary['protocol_sha256']
        assert read(context/'gpu-bridge/verification.json')['passed']
        selected=[]
        for p in sorted((folder/'collection').glob('e*/selected_results.json')):selected+=read(p)
        assert len(selected)==100 and len({r['job_id'] for r in selected})==100
        assert all(r['valid'] and not r['cuda_initialized'] for r in selected)
        assert sum(r['num_transitions'] for r in selected)==sum(h['num_samples'] for h in history)
        for index,name in enumerate(schedule):
            prefix='e{:05d}-{}-'.format(index,name)
            assert sum(r['job_id'].startswith(prefix) for r in selected)==1
        raw_count=len(list((folder/'collection').rglob('e*-*.json')))
        assert raw_count==100,(v,s,raw_count)
        training.append(dict(variant=v,training_seed=s,episodes=100,ppo_blocks=25,
            logical_minibatches=summary['optimizer_steps'],transitions=sum(r['num_transitions'] for r in selected),
            max_clip_fraction=max(h['clip_fraction'] for h in history),
            cpu=preflight['stack']['cpu_model'],gpu=preflight['gpu'],
            charged_wall_seconds=status['charged_wall_seconds']))
        all_rows[v,s]=read(RUN/'evaluation/episode-0100'/('{}-s{}'.format(v,s))/'selected_results.json')
        identity=read(RUN/'evaluation/episode-0100'/('{}-s{}'.format(v,s))/'identity.json')
        assert identity['stack']==manifest['stack']
        assert identity['actor_sha256']==m['actor_sha256']
    for s in SEEDS:
        for k in ('common_actor_fingerprint','common_critic_fingerprint','public_torch_rng_sha256','public_cuda_rng_sha256'):
            assert len({initialization[v,s][k] for v in VARIANTS})==1,(s,k)
    for key,rows in all_rows.items():
        assert len(rows)==10 and {(r['instance_name'],r['solver_seed']) for r in rows}==expected,key
        assert all(r['valid'] and r['time_limit_seconds']==120 for r in rows)
        assert all(r['pdi_horizon_lower']<=r['pdi_horizon_midpoint']<=r['pdi_horizon_upper'] for r in rows)
        if key!='native':
            assert all(r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
                and not r['cuda_initialized'] for r in rows)
    inactive=[n for n in names if all(r['branch_count']==0 for key,rows in all_rows.items() if key!='native'
        for r in rows if r['instance_name']==n)]
    active=[n for n in names if n not in inactive]
    models=[];instances=[]
    for key,rows in all_rows.items():
        v,s=('native',None) if key=='native' else key
        for panel,subset in [('all',names),('actor_participating',active)]:
            models.append(dict(variant=v,training_seed=s,panel=panel,**summarize([r for r in rows if r['instance_name'] in subset])))
        for n in names:instances.append(dict(variant=v,training_seed=s,instance=n,**summarize([r for r in rows if r['instance_name']==n])))
    differences=[];contributions=[];instance_means=[];local_pairs=[]
    for n in names:
        means={v:stats.mean(r['pdi'] for r in instances if r['variant']==v and r['instance']==n) for v in ('native',)+VARIANTS}
        instance_means.append(dict(instance=n,actor_participates=n in active,**means,
            candidate_vs_b0_percent=100*(means['candidate']/means['b0']-1)))
        contributions.append(dict(instance=n,actor_participates=n in active,
            contribution_to_all_mean=(means['candidate']-means['b0'])/len(names)))
    for s in SEEDS:
        for panel in ('all','actor_participating'):
            values={v:next(r for r in models if r['variant']==v and r['training_seed']==s and r['panel']==panel) for v in VARIANTS}
            for ref in ('b0','score'):
                differences.append(dict(training_seed=s,panel=panel,reference=ref,
                    pdi_delta=values['candidate']['pdi']-values[ref]['pdi'],
                    percent_delta=100*(values['candidate']['pdi']/values[ref]['pdi']-1),
                    delta_lower=values['candidate']['pdi_lower']-values[ref]['pdi_upper'],
                    delta_upper=values['candidate']['pdi_upper']-values[ref]['pdi_lower']))
        for ss in solver_seeds:
            chosen={v:next(r for r in all_rows[v,s] if r['instance_name']=='dcmulti' and r['solver_seed']==ss) for v in VARIANTS}
            local_pairs.append(dict(training_seed=s,solver_seed=ss,**{
                v+'_'+metric:chosen[v][key] for v in VARIANTS
                for metric,key in [('pdi','pdi_horizon_midpoint'),('nodes','nnodes'),('solve_seconds','scip_solve_time')]}))
    aggregate={panel:{v:stats.mean(r['pdi'] for r in models if r['variant']==v and r['panel']==panel)
        for v in ('native',)+VARIANTS} for panel in ('all','actor_participating')}
    fraction=sum(r['contribution_to_all_mean'] for r in contributions if not r['actor_participates'])/(
        aggregate['all']['candidate']-aggregate['all']['b0'])
    finalization=read(RUN/'finalization.json');assert finalization['complete'] and finalization['episode']==100
    elapsed=sum(r['charged_wall_seconds'] for r in training)+finalization['charged_wall_seconds']
    assert elapsed<16*3600
    raw_native=list((RUN/'cpu-flex/normalization').glob('*/normalization/jobs/attempt-*/native-*.json'))
    raw_native+=list((RUN/'shared/normalization/jobs').glob('attempt-*/native-*.json'))
    native_evaluation=list((RUN/'evaluation/episode-0100/native').glob('attempt-*/dev-*.json'))
    summary=dict(complete=True,common_episodes=100,models=models,training=training,
        aggregate=aggregate,differences=differences,instance_means=instance_means,
        non_actor_instances=inactive,non_actor_share_of_pdi_difference=fraction,
        charged_pipeline_wall_hours=elapsed/3600,new_training_solves=600,
        new_normalization_solves=len(raw_native),new_model_evaluation_solves=60,
        new_native_evaluation_solves=len(native_evaluation),
        new_scip_solves=600+len(raw_native)+60+len(native_evaluation),
        evaluation_stack=manifest['stack'],
        conclusion='local gains, not established broad policy improvement; global raw PDI gain dominated by no-Actor instance')
    csv_write('panel_summary.csv',models);csv_write('paired_panels.csv',differences)
    csv_write('instance_means.csv',instance_means);csv_write('pdi_contributions.csv',contributions)
    csv_write('dcmulti_paired.csv',local_pairs);csv_write('training_summary.csv',training)
    save('analysis.json',summary);save('input_hashes.json',evidence)
    save('verification.json',dict(passed=True,models=6,episodes=600,model_evaluations=60,
        native_references=10,independent_instances=5,all_initializations_paired=True,
        all_training_cpu_normalizers_match=True,all_eval_stacks_match=True,
        pending_transitions=0,raw_artifacts_preserved=True,source_hashes_checked=len(evidence),
        protected_baseline_files_unchanged=len(protected),
        new_training_or_scip_in_this_review=False))
    print(json.dumps({k:summary[k] for k in ('aggregate','non_actor_share_of_pdi_difference',
        'charged_pipeline_wall_hours','new_scip_solves')},indent=2))

if __name__=='__main__':main()
