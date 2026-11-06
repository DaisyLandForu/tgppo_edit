"""Audit diagnostic coverage and recheck protected sources/checkpoints."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parents[1]
RESEARCH=EXPERIMENT.parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    files={name:json.loads((HERE/name).read_text()) for name in ('causal-results.json','timeline.json',
        'uniform-control.json','cost-attribution.json','trained-benchmark-cpu.json')}
    cause=files['causal-results.json'];timeline=files['timeline.json'];control=files['uniform-control.json']
    checks={name+'_checks_passed':all(payload['checks'].values()) for name,payload in files.items()}
    checks['193_real_causal_states']=len(cause['records'])==193 and len(cause['summaries'])==4
    checks['36_causal_checks']=len(cause['checks'])==36
    checks['24_timeline_checks']=len(timeline['checks'])==24 and len(timeline['records'])==24
    checks['20_online_cases_16_uniform_comparisons']=len(control['rows'])==20 and len(control['comparisons'])==16
    checks['all_online_feature_action_traces_equal_uniform']=all(r['actions_identical'] for r in control['comparisons'])
    checks['12_identical_benchmark_states']=len(files['trained-benchmark-cpu.json']['rows'])==48 and files['trained-benchmark-cpu.json']['states']==12
    checks['2000_training_episode_costs']=sum(r['episodes'] for r in files['cost-attribution.json']['rows'])==2000
    tests=(HERE/'tests.log').read_text()
    checks['six_unit_tests_passed']='Ran 6 tests' in tests and '\nOK\n' in tests and 'FAILED' not in tests
    frozen_paths=['code','configs','env.sh','custom_job.sh']
    completed=subprocess.run(['git','diff','--exit-code','222be16','--']+frozen_paths,cwd=str(EXPERIMENT),capture_output=True)
    checks['production_sources_and_configs_unchanged']=completed.returncode==0
    paths=[EXPERIMENT/'code/project/policy/modules.py',RESEARCH/'tgppo/repo/code/project/policy/modules.py',
           RESEARCH/'tgppo/experiments/server-20261022/code/project/policy/modules.py']
    checks['head_source_identical_to_original']=all(digest(p)=='959605658df62563d6d8145eda28838c61eb0c592188a85caa275450b55344aa' for p in paths)
    checkpoints={}
    for payload in files.values():checkpoints.update(payload.get('protected_checkpoint_sha256',{}))
    checks['all_24_checkpoint_files_still_unchanged']=len(checkpoints)==24 and all(digest(p)==h for p,h in checkpoints.items())
    artifacts={str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='verification.json'}
    receipt=dict(passed=all(checks.values()),checks=checks,protected_checkpoint_sha256=checkpoints,artifact_sha256=artifacts,
        training_executed=False,production_source_modified=False,formal_quality_evaluation_executed=False,
        scope='Read-only causal diagnosis, local bounded CPU controls and cost accounting; no validated trained repair.')
    (HERE/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=receipt['passed'],checks=checks),ensure_ascii=False,indent=2))
    if not receipt['passed']:raise RuntimeError('Diagnostic deliverable verification failed')


if __name__=='__main__':main()
