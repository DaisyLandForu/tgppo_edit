"""Cross-check delivered numbers, source identity, raw tasks, and local links."""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
OLD=ROOT/'tgppo/runs/server6-20261022/stage5-results-20261023'


def readcsv(p):return list(csv.DictReader(p.open()))


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-8),(a,b)


def main():
    reports=['00_inventory_and_protocol.md','01_paper_limitations_evidence_matrix.md',
             '02_training_reward_actor_diagnostics.md','03_runtime_and_solver_diagnostics.md',
             '04_hypotheses_and_low_cost_validation.md','MEETING_BRIEF_20261030.md','README.md']
    broken=[];link_count=0
    for name in reports:
        p=OUT/name;assert p.is_file(),name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https://','http://','#')):continue
            target=target.split('#')[0];link_count+=1
            resolved=(p.parent/target).resolve()
            # These two receipts are written by this validation invocation.
            if not resolved.exists() and resolved not in [OUT/'validation.json',OUT/'delivery_manifest.json']:
                broken.append([name,target])
    assert not broken,broken
    ledger=json.loads((OLD/'source_ledger.json').read_text())['sources']
    codepaths=[p for p in ledger if 'tgppo/experiments/server-20261022/code/' in p]
    for p in codepaths:
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==ledger[p],p
    tasks=readcsv(OLD/'per_task_with_sources.csv')
    assert len(tasks)==len({(r['model'],r['instance'],r['solver_seed']) for r in tasks})==1386
    # Verify CSV numbers against every original task, not only their aggregate.
    for r in tasks:
        raw=json.loads((ROOT/r['source_path']).read_text())
        assert raw['valid'] and raw['task_terminal']
        for csvkey,rawkey in [('pdi','pdi_horizon_midpoint'),('pdi_lower','pdi_horizon_lower'),
                              ('pdi_upper','pdi_horizon_upper'),('nodes','nnodes'),
                              ('solve_seconds','scip_solve_time'),('branch_count','branch_count')]:
            close(r[csvkey],raw.get(rawkey,0))
        assert (r['proof_within_120s']=='True')==(raw['proof_completed'] and raw['scip_solve_time']<=120)
        for csvkey,rawkey in [('actor_seconds','actor'),('base_feature_seconds','features'),
                              ('entire_branch_callback_seconds','callback')]:
            close(r[csvkey],raw.get('callback_seconds',{}).get(rawkey,0))
    summary=json.loads((OLD/'summary.json').read_text())
    for row in summary['combined']:
        method=row['method'];rs=[r for r in tasks if r['method']==method]
        by=defaultdict(list)
        for r in rs:by[r['instance']].append(r)
        assert len(by)==66
        pdi=sum(sum(float(r['pdi']) for r in entries)/len(entries) for entries in by.values())/66
        proof=sum(r['proof_within_120s']=='True' for r in rs)/len(rs)
        par=sum(float(r['solve_seconds']) if r['proof_within_120s']=='True' else 240 for r in rs)/len(rs)
        close(pdi,row['mean_pdi']);close(proof,row['proof_rate']);close(par,row['mean_par2'])
    actor=json.loads((OUT/'actor_offline_summary.json').read_text())
    assert actor['observations']==50 and actor['instances']==17 and not actor['cuda_initialized']
    assert len(readcsv(OUT/'actor_checkpoint_metrics.csv'))==600
    assert len(readcsv(OUT/'actor_checkpoint_comparison.csv'))==300
    assert len(readcsv(OUT/'actor_gradient_metrics.csv'))==144
    assert len(readcsv(OUT/'actor_invariance_checks.csv'))==68
    data=json.loads((OUT/'existing_data_summary.json').read_text())
    assert data['hash_status_counts']=={'match':1665}
    assert sum(r['valid_updates'] for r in data['updates'])==744
    assert sum(r['optimizer_steps'] for r in data['updates'])==22539
    assert data['b0_reward_episode_count']==1000 and data['b0_reward_transitions']==615922
    assert data['both_proven_pairs']==76 and data['fewer_nodes_but_slower_pairs']==8
    for fig in ['01_learning_diagnostics','02_runtime_solver_diagnostics','03_gap_time_examples']:
        for ext in ['png','svg','pdf']:assert (OUT/'figures'/f'{fig}.{ext}').stat().st_size>1000
    result=dict(status='passed',checked_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                reports=len(reports),local_links=link_count,broken_links=broken,
                raw_tasks_checked=1386,method_aggregates_checked=4,runtime_source_files_rechecked=len(codepaths),
                source_ledger_matches=1665,actor_forward_records=600,
                note='Historical source hashes were audited by analyze_existing.py; runtime sources were rechecked here. No solve or training.')
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(algorithm='sha256',files=files),indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
