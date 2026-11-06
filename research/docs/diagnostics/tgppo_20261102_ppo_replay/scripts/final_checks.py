"""Audit the original sources, exact step cap, artifacts and evidence links."""
import csv
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(name):return list(csv.DictReader((OUT/name).open()))


def main():
    v=json.loads((OUT/'validation.json').read_text());assert v['status']=='passed'
    frozen=json.loads((OUT/'frozen_inputs.json').read_text())
    assert sha(OUT/'frozen_protocol.json')==frozen['protocol_sha256']
    assert sha(OUT.parent/'tgppo_20261102_ppo_replay_plan/PLAN.md')==frozen['source_plan_sha256']
    sources=frozen['original_sources']
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    assert sources==json.loads((OUT/'runtime_sources_after.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['trajectory_and_config_hashes'].items())
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in frozen['checkpoint_specs'])
    assert all(sha(OUT/p)==h for p,h in frozen['model_copy_sha256'].items())
    prior_count=0
    for directory,r in frozen['prior_deliveries'].items():
        folder=OUT.parent/directory
        assert sha(folder/'delivery_manifest.json')==r['manifest_sha256']
        files=json.loads((folder/'delivery_manifest.json').read_text())['files']
        assert all(sha(folder/p)==h for p,h in files.items())
        prior_count+=len(files)
    entries=json.loads((OUT/'optimizer_steps.json').read_text())['entries']
    assert len(entries)==4 and all(r['status']=='completed' for r in entries)
    assert sorted((r['model'],r['network']) for r in entries)==[(m,k) for m in ['B0-s11','B0-s22'] for k in ['actor','critic']]
    seeds=json.loads((OUT/'seed_summaries.json').read_text())
    assert len(seeds)==2
    for s in seeds:
        assert sha(OUT/s['diagnostics_saved'])==s['diagnostic_checkpoint_sha256']
        assert s['preupdate_ratio_max_error']<=1e-4
        assert s['actor_steps']==s['critic_steps']==1 and not s['full_ppo_update_completed']
        assert s['original_learn']['progress']['minibatches_completed']==1
    params=read('parameter_adam_changes.csv')
    assert all(float(r['adam_step_after'])-float(r['adam_step_before'])==1 for r in params)
    batch=read('batch_manifest.csv');responses=read('policy_responses.csv')
    assert len(batch)==512 and len(responses)==840
    assert len(read('trajectory_imports.csv'))==8
    for m in ['B0-s11','B0-s22']:
        indices=json.loads((OUT/'frozen_batches'/f'{m}.json').read_text())['indices']
        assert indices==[int(r['global_index']) for r in batch if r['model']==m]
    # Restore-before-forward identity with the already sealed localization.
    old=list(csv.DictReader((OUT.parent/'tgppo_20261102_signal_localization/policy_metrics.csv').open()))
    old_index={(r['model'],r['observation']):float(r['tv_uniform']) for r in old if r['episode']=='100' and r['variant']=='original'}
    max_before_error=0.
    for r in responses:
        if r['dataset']=='monitor_164':
            delta=abs(float(r['pre_tv_uniform'])-old_index[r['model'],r['state_key']])
            assert delta<=1e-12
            max_before_error=max(max_before_error,delta)
    generated={'delivery_validation.json','delivery_manifest.json'};links=[]
    for path in [OUT/'README.md',OUT/'REPORT.md',OUT/'AUTHORIZATION.md']:
        for href in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if href.startswith(('http://','https://','#')):continue
            dest=(path.parent/href.split('#')[0]).resolve()
            assert dest.exists() or (dest.parent==OUT and dest.name in generated),(path,href)
            links.append(href)
    for ext in ['png','svg','pdf']:assert (OUT/'figures'/f'01_single_step_response.{ext}').stat().st_size>1000
    budget=json.loads((OUT/'compute_budget.json').read_text())
    assert sum(r.get('elapsed_seconds',0) for r in budget['runs'])+budget['preparation_reserve_seconds']<600
    result=dict(status='passed',checked_at_utc=datetime.now(timezone.utc).isoformat(),
                original_source_and_split_hashes_checked=len(sources),original_checkpoints_checked=2,
                trajectory_and_config_hashes_checked=len(frozen['trajectory_and_config_hashes']),
                historical_files_unchanged=prior_count,independent_source_copies_checked=len(frozen['model_copy_sha256']),
                original_policy_monitor_max_error=max_before_error,optimizer_step_ledger_completed=4,
                adam_parameter_step_checks=len(params),new_solver_tasks=0,local_links_checked=len(links),
                cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
                stage_status='complete; stop and await user acceptance')
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
