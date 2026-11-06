"""Freeze an outcome-independent, native-cost-stratified exploration cohort."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/candidate-residual-20261104/cross-instance-8'
SOURCE=ROOT/'tgppo/runs/server6-20261022/stage5-results-20261023/per_task_with_sources.csv'
SALT='tgppo-cr-cross-instance-20261104-v1:'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    rows=list(csv.DictReader(SOURCE.open()))
    data={r['name']:r for r in json.loads((ROOT/'tgppo/datasets/download_manifest.json').read_text())}
    cutoffs=json.loads((ROOT/'tgppo/datasets/cutoffs.json').read_text())
    original=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())['data']
    pool=[]
    for name in sorted(p.name[:-7] for p in (ROOT/'tgppo/datasets/test_instances').glob('*.mps.gz')):
        native=[r for r in rows if r['instance']==name and r['method']=='native']
        assert len(native)==3
        seconds=stats.mean(float(r['solve_seconds']) for r in native)
        proved=sum(r['proof_within_120s']=='True' for r in native)
        exception=any(r['memory_exception']=='True' for r in native)
        stratum=('unfinished' if proved<3 else 'fast' if seconds<=10 else 'medium' if seconds<=60 else 'slow')
        pool.append(dict(instance=name,stratum=stratum,native_mean_solve_seconds=seconds,
            native_proof_count=proved,eligible=not exception,
            exclusion='historical extra-memory protocol; no new memory exception in this pilot' if exception else None,
            selection_hash=hashlib.sha256((SALT+name).encode()).hexdigest()))
    selected=[]
    for group in ('fast','medium','slow','unfinished'):
        chosen=sorted([r for r in pool if r['stratum']==group and r['eligible']],key=lambda r:r['selection_hash'])[:2]
        assert len(chosen)==2
        for r in chosen:
            name=r['instance'];path=ROOT/'tgppo/datasets/test_instances'/(name+'.mps.gz')
            assert name not in original['train']+original['development']
            assert sha(path)==data[name]['sha256']
            assert cutoffs[name]['status']=='optimal' and cutoffs[name]['source']
            selected.append(dict(r,path=str(path),sha256=sha(path)))
    learned=[r for r in rows if r['method']=='B0' and r['instance'] in {v['instance'] for v in selected}]
    native=[r for r in rows if r['method']=='native' and r['instance'] in {v['instance'] for v in selected}]
    # These legacy B0 costs estimate runtime only, after selection is fixed.
    predicted=2*sum(6*stats.mean(float(r['worker_wall_seconds']) for r in learned if r['instance']==v['instance'])+
        stats.mean(float(r['worker_wall_seconds']) for r in native if r['instance']==v['instance']) for v in selected)
    result=dict(schema='frozen-cross-instance-exploration-v1',scope='8-instance subset of previously examined 66; not a pristine holdout',
        authorization='User requested frozen-model small cross-instance evaluation; local CPU, Custom only if needed; no training',
        selection_rule='Two smallest salted SHA256 ranks per native-cost stratum; no current B0/S/CR outcomes used',
        salt=SALT,strata_definitions={'fast':'3/3 native proofs; mean solve <=10s',
            'medium':'3/3 native proofs; 10s<mean solve<=60s','slow':'3/3 native proofs; mean solve>60s',
            'unfinished':'fewer than3/3 native proofs within120s'},
        selection_source=str(SOURCE),selection_source_sha256=sha(SOURCE),pool=pool,selected=selected,
        solver_seeds=[0,1],training_seeds=[11,22],variants=['b0','score','candidate'],
        model_tasks=96,native_tasks=16,total_tasks=112,workers=4,threads_per_worker=1,
        time_limit_seconds=120,watchdog_seconds=180,aggregate_elapsed_wall_cap_seconds=5400,
        legacy_predicted_sum_worker_seconds=predicted,legacy_predicted_4_worker_seconds=predicted/4,
        solve_limit_core_hour_ceiling=112*120/3600,
        success_rule='Report all8 instance-equal PDI and proof rate, CR-vs-B0/S separately for both training seeds, per-instance signs, native reference and Actor participation; broad gain requires consistent seed directions and more than isolated instance wins. No posthoc model selection.',
        stop_rule='Complete112 tasks or90 elapsed minutes, whichever first; preserve invalid/missing tasks, no automatic retries/cohort changes/training or full66 expansion')
    RUN.mkdir(parents=True,exist_ok=True)
    path=RUN/'cohort.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(selected=selected,estimated_worker_seconds=predicted,
        estimated_4_worker_minutes=predicted/240),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
