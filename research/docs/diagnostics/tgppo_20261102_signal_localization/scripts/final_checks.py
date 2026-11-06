"""Evidence identity, record consistency, and local link audit."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
PRIOR = OUT.parent/'tgppo_20261102_local_validation'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read(name):
    return list(csv.DictReader((OUT/name).open()))


def main():
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['status']=='passed' and validation['scope_complete']
    frozen=json.loads((OUT/'frozen_inputs.json').read_text())
    assert sha(OUT/'frozen_protocol.json')==frozen['protocol_sha256']
    sources=json.loads((OUT/'runtime_sources_before.json').read_text())
    assert sources==json.loads((OUT/'runtime_sources_after.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    assert all(sha(ROOT/p)==h for p,h in frozen['source_trajectory_hashes'].items())
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in frozen['checkpoint_specs'])
    for n in ['actor.py','modules.py']:
        assert sha(OUT/'model_copy'/n)==sha(ROOT/'tgppo/experiments/server-20261022/code/project/policy'/n)
    prior_manifest=json.loads((PRIOR/'delivery_manifest.json').read_text())['files']
    assert all(sha(PRIOR/p)==h for p,h in prior_manifest.items()), 'Historical stage artifacts drifted'
    assert sha(OUT/'observations.pkl')==frozen['prior_observations_sha256']==prior_manifest['observations.pkl']
    tree=ast.parse((OUT/'scripts/localize.py').read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','optimize','minimize']]
    policies=read('policy_metrics.csv')
    layers=read('layer_metrics.csv')
    assert len(layers)==164*4*2*33, len(layers)
    assert len(read('effective_ranks.csv'))==17*4*2*10
    assert len(read('score_gradients.csv'))==17*4*2*14
    old=list(csv.DictReader((PRIOR/'policy_metrics.csv').open()))
    old_index={(r['model'],r['episode'],r['observation']):r for r in old if r['variant']=='original'}
    max_policy_difference=0.
    for r in policies:
        if r['variant']=='original':
            previous=old_index[r['model'],r['episode'],r['observation']]
            delta=abs(float(previous['tv_to_uniform'])-float(r['tv_uniform']))
            max_policy_difference=max(max_policy_difference,delta)
            assert delta<=1e-12, (r['model'],r['observation'],delta)
    original={(r['model'],r['episode'],r['observation'],r['layer']):r for r in layers if r['variant']=='original'}
    upstream_checks=0
    for r in layers:
        if r['variant']=='original' or r['layer'] in ['head6.norm','head6.out','head7.linear','logits']:
            continue
        previous=original[r['model'],r['episode'],r['observation'],r['layer']]
        for field in ['centered_rms','total_rms','relative_centered_rms','mean_candidate_range']:
            assert previous[field]==r[field], (r['layer'],field)
        upstream_checks+=1
    summary=json.loads((OUT/'summary.json').read_text())
    assert max(summary['invariance'].values())<=1e-6
    assert max(r['reconstruction_max_error'] for r in summary['bimatching'])<=1e-6
    links=[]
    generated={'delivery_validation.json','delivery_manifest.json'}
    for p in [OUT/'README.md',OUT/'REPORT.md',OUT/'AUTHORIZATION.md']:
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if href.startswith(('http://','https://','#')):
                continue
            dest=(p.parent/href.split('#')[0]).resolve()
            assert dest.exists() or (dest.parent==OUT and dest.name in generated), (p,href)
            links.append(href)
    for name in ['01_layerwise_contrast','02_fixed_scale_intervention']:
        for ext in ['png','svg','pdf']:
            assert (OUT/'figures'/f'{name}.{ext}').stat().st_size>1000
    budget=json.loads((OUT/'compute_budget.json').read_text())
    assert sum(r.get('elapsed_seconds',0) for r in budget['runs'])+budget['preparation_reserve_seconds']<600
    result=dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                baseline_source_and_split_hashes_checked=len(sources), checkpoints_checked=4,
                trajectories_checked=len(frozen['source_trajectory_hashes']),
                prior_delivery_files_unchanged=len(prior_manifest), upstream_identity_checks=upstream_checks,
                original_policy_tv_max_difference_from_prior=max_policy_difference,
                local_links_checked=len(links), activation_scale_preservation=summary['scale_preservation_passed'],
                scientific_result='See REPORT.md; delivery checks do not establish policy improvement',
                cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
                optimizer_steps=0,new_solver_tasks=0,stage_status='complete; stop and await acceptance')
    (OUT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
