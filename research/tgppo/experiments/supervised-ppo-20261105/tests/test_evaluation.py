"""Stdlib fixtures for fair selection/gate/partial results; no actual evaluation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / 'evaluation'))
from common import read, sha, specs, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.monotonic()
    fixture = Path(tempfile.mkdtemp(prefix='tgppo-expert-eval-check-'))
    analysis = load('supervised_analysis_fixture', PACKAGE / 'evaluation/analyze.py')
    analysis.OUT, analysis.DOCS, analysis.BASE = fixture / 'out', fixture / 'docs', fixture / 'matrix'
    analysis.TOP = fixture / 'top'
    write(analysis.TOP / 'docs/release.json', dict(fixture_only=True))
    (analysis.TOP / 'docs/PROTOCOL.md').write_text((PACKAGE / 'docs/PROTOCOL.md').read_text())
    analysis.OUT.mkdir()
    analysis.DOCS.mkdir()
    source = read(ROOT / 'tgppo/runs/light-actor-20261105/development-curve-70/evaluation_rows.json')
    template = next(r for r in source if r['mode'] == 'eval')
    rows, tasks = [], []
    for item in specs():
        r = copy.deepcopy(template)
        seed = item['training_seed'] if item['training_seed'] is not None else 11
        sampling = int(hashlib.sha256((str(seed) + ':' + item['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
        score = 80. if item['variant'] == 'native' else (100. if item['variant'] == 'bc' else
            {('control', 20): 95., ('control', 100): 90., ('expert', 20): 80., ('expert', 100): 70.}[(item['variant'], item['episode'])])
        if item['instance'] == 'roi5alpha10n8':
            score = 10000.
        r.update(job_id=item['job_id'], instance_name=item['instance'], solver_seed=item['solver_seed'],
            sampling_seed=sampling, valid=True, worker_cpu_complete=True, cuda_initialized=False,
            mode='native' if item['variant'] == 'native' else 'eval', pdi_horizon_midpoint=score,
            pdi_horizon_lower=score - .01, pdi_horizon_upper=score + .01, proof_completed=True, scip_solve_time=5.,
            branch_count=0 if item['instance'] == 'roi5alpha10n8' or item['variant'] == 'native' else 10,
            callback_seconds=dict(actor=.01, features=.02, callback=.04, critic=0.), fallback_action_count=0)
        rows.append(r)
        tasks.append(dict(item, job=dict(sampling_seed=sampling)))
    pair = {str(s): {v: dict(cpu='FIXTURE', gpu='FIXTURE') for v in ('control', 'expert')} for s in (11, 22)}
    write(analysis.OUT / 'manifest.json', dict(models={}, tasks=tasks, pairings=pair,
        stack=dict(cpu_model='FIXTURE ONLY'), host='fixture', runner_sha256=sha(PACKAGE / 'evaluation/run.py'),
        protocol_sha256=sha(PACKAGE / 'docs/PROTOCOL.md')))
    write(analysis.OUT / 'status.json', dict(status='complete'))
    write(analysis.OUT / 'job_state.json', dict(error=None))
    for v in ('control', 'expert'):
        for s in (11, 22):
            write(analysis.BASE / v / 'training' / ('B0-s' + str(s)) / 'updates.json',
                [dict(updated=True, num_updates=1, expert_batches=int(v == 'expert'),
                    expert_samples=16 * int(v == 'expert'), learning_wall_seconds=1.)])
            write(analysis.BASE / 'jobs' / (v + '-s' + str(s) + '.json'), dict(charged_seconds=60, status='training_complete'))
    write(analysis.OUT / 'evaluation_rows.json', rows)
    assert analysis.main() == 0
    value = read(analysis.DOCS / 'analysis.json')
    assert value['status'] == 'complete' and value['valid_tasks'] == 110 and value['gate']['passed']
    assert len(value['common_actor_instances']) == 4 and value['no_actor_instances'] == ['roi5alpha10n8']
    assert all(x['selected_episode'] == 100 for x in value['selected'])
    # No improvement is a complete experiment, not an execution error or global-method failure.
    neutral = copy.deepcopy(rows)
    for item, r in zip(specs(), neutral):
        if item['variant'] == 'expert' and item['instance'] != 'roi5alpha10n8':
            score = 95. if item['episode'] == 20 else 90.
            r.update(pdi_horizon_midpoint=score, pdi_horizon_lower=score - .01, pdi_horizon_upper=score + .01)
    write(analysis.OUT / 'evaluation_rows.json', neutral)
    assert analysis.main() == 0
    assert not read(analysis.DOCS / 'analysis.json')['gate']['passed']
    write(analysis.OUT / 'evaluation_rows.json', rows[:-1])
    assert analysis.main() == 1 and read(analysis.DOCS / 'analysis.json')['valid_tasks'] == 109
    write(analysis.OUT / 'evaluation_rows.json', rows + [rows[0]])
    assert analysis.main() == 1 and 'duplicate_or_unexpected_rows' in read(analysis.DOCS / 'analysis.json')['errors']
    job = load('supervised_job_fixture', PACKAGE / 'evaluation/job.py')
    job.BASE, job.OUT, job.DOCS = fixture / 'not-trained', fixture / 'not-started', fixture / 'not-docs'
    assert job.run() == 2 and not job.OUT.exists()
    job.BASE = analysis.BASE
    job.OUT.mkdir()
    write(job.OUT / 'bootstrap.json', dict(budget_seconds=3000))
    write(job.OUT / 'job_state.json', dict(status='complete', charged_wall_seconds=10.))
    assert job.run() == 0 and not (job.OUT / 'runner.log').exists()
    assert 'torch' not in sys.modules
    result = dict(passed=True, fixture_only=True, fixture_directory=str(fixture), elapsed_seconds=time.monotonic() - started,
        checks=['110-task matrix', 'equal checkpoint opportunities', 'common/no Actor split', 'prespecified positive gate',
            'valid negative result is complete', 'missing/duplicate fail closed', 'early CPU submission consumes no budget',
            'completed evaluation never restarts', 'persistent JSON does not require node-local raw files'],
        solver_tasks_executed=0, torch_imported=False)
    write(ROOT / 'tgppo/runs/supervised-ppo-20261105/preparation/evaluation_checks.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
