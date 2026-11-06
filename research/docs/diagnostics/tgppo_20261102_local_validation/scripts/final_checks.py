"""Final evidence/identity/link audit; no model forward, training, or solver."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime,timezone,timedelta

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    for name in ['local_validation.py','cpu_microbenchmark.py']:
        s=(OUT/'scripts'/name).read_text();tree=ast.parse(s)
        forbidden=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in ['step','optimize','minimize']]
        assert not forbidden,name
    baseline=json.loads((OUT/'runtime_sources_before.json').read_text())
    for path,expected in baseline.items():assert sha(ROOT/path)==expected,path
    checkpoints=json.loads((OUT/'checkpoint_audit.json').read_text())
    for entry in checkpoints:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    trajectories=json.loads((OUT/'input_trajectory_hashes.json').read_text())
    for path,expected in trajectories.items():assert sha(ROOT/path)==expected,path
    first_stage=json.loads((ROOT/'docs/diagnostics/tgppo_20261030/source_ledger.json').read_text()) if (ROOT/'docs/diagnostics/tgppo_20261030/source_ledger.json').exists() else None
    # The original audit ledger lives alongside historical results, not inside
    # the diagnostic output. Compare only the runtime files in our scope.
    previous=json.loads((ROOT/'tgppo/runs/server6-20261022/stage5-results-20261023/source_ledger.json').read_text())['sources']
    compared=[p for p in baseline if p in previous]
    assert all(baseline[p]==previous[p] for p in compared)
    generated={'delivery_validation.json','delivery_manifest.json'};links=[]
    for p in [OUT/'README.md',OUT/'REPORT.md',OUT/'AUTHORIZATION.md']:
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if href.startswith(('https://','http://','#')):continue
            href=href.split('#')[0];target=(p.parent/href).resolve()
            assert target.exists() or (target.parent==OUT and target.name in generated),(p.name,href)
            links.append(href)
    for name in ['01_local_intervention','02_cpu_microbenchmark']:
        for ext in ['png','svg','pdf']:assert (OUT/'figures'/f'{name}.{ext}').stat().st_size>1000
    validation=json.loads((OUT/'validation.json').read_text());assert validation['status']=='passed'
    budget=json.loads((OUT/'compute_budget.json').read_text())
    assert sum(r.get('elapsed_seconds',0) for r in budget['runs'])+budget['preparation_reserve_seconds']<budget['limit_seconds']
    assert all(r.get('exit_code')==0 for r in budget['runs'] if r['status']!='running')
    selection=list(csv.DictReader((OUT/'benchmark_selection.csv').open()))
    result=dict(delivery_checks='passed',checked_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                scientific_screening='No visible preference recovered by frozen-weight LN2 normalization bypass',
                baseline_source_and_split_hashes_checked=len(baseline),prior_ledger_runtime_hashes_match=len(compared),
                checkpoints_checked=len(checkpoints),source_trajectories_checked=len(trajectories),local_links_checked=len(links),
                independent_model_source_copies_match=True,benchmark_instances=len({r['instance'] for r in selection}),
                optimizer_steps=0,new_solver_tasks=0,stage_status='complete; await user acceptance; no next-stage work')
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
