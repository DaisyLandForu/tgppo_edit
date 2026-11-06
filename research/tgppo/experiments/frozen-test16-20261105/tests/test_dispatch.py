"""Stdlib lifecycle/analysis fixtures. No model import, GPU, or SCIP optimization."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import common


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='tgppo-test16-check-'))
    common.RUN, common.DOCS = root / 'run', root / 'reports'
    analyze = module('test16_analysis_fixture', PACKAGE / 'analyze.py')
    # Hashes point to isolated fixture files; nothing is written to real output.
    fake = root / 'package'
    (fake / 'docs').mkdir(parents=True)
    for name in ('run.py', 'analyze.py', 'docs/PROTOCOL.md', 'selection.json', 'release.json'):
        (fake / name).write_text('fixture-only\n')
    analyze.PACKAGE = fake
    source = common.read(common.ROOT / 'tgppo/runs/light-actor-20261105/development-curve-70/evaluation_rows.json')
    template = next(r for r in source if r['mode'] == 'eval')
    no_actor = common.PLAN['selected'][0]['instance']
    zero_pdi = common.PLAN['selected'][1]['instance']
    all_ids = set()
    for shard in range(4):
        tasks, rows = [], []
        for t in common.specs(shard):
            assert t['job_id'] not in all_ids
            all_ids.add(t['job_id'])
            seed = t['training_seed'] if t['training_seed'] is not None else 11
            sampling = int(hashlib.sha256((str(seed) + ':' + t['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
            score = dict(native=70., b0=100., b0_bc=95., mlp=90., mlp_bc=80.)[t['variant']]
            if t['instance'] == no_actor:
                score = 1000.
            if t['instance'] == zero_pdi:
                score = 0.
            r = copy.deepcopy(template)
            r.update(job_id=t['job_id'], instance_name=t['instance'], solver_seed=t['solver_seed'],
                sampling_seed=sampling, valid=True, worker_cpu_complete=True, cuda_initialized=False,
                mode='native' if t['variant'] == 'native' else 'eval', time_limit_seconds=120, pdi_horizon_seconds=120,
                pdi_horizon_midpoint=score, pdi_horizon_lower=max(0, score - .01), pdi_horizon_upper=score + .01,
                proof_completed=True, scip_solve_time=5., branch_count=0 if t['instance'] == no_actor or t['variant'] == 'native' else 10,
                callback_seconds=dict(actor=.01, features=.02, callback=.04, critic=0), fallback_action_count=0)
            tasks.append(dict(t, job=dict(sampling_seed=sampling)))
            rows.append(r)
        out = common.output(shard)
        manifest = dict(tasks=tasks, models={m['label']: dict(m, source=m['actor']) for m in common.PLAN['frozen_models']},
            selected=common.selected(shard), stack=dict(cpu_model='FIXTURE' + str(shard)), host='fixture-' + str(shard),
            runner_sha256=common.sha(fake / 'run.py'), protocol_sha256=common.sha(fake / 'docs/PROTOCOL.md'),
            selection_sha256=common.sha(fake / 'selection.json'))
        common.write(out / 'manifest.json', manifest)
        common.write(out / 'status.json', dict(status='complete', weights_unchanged=True, summed_worker_wall_seconds=72))
        common.write(out / 'job_state.json', dict(status='complete', engine_complete=True, final_input_verification={'passed': True}, error=None, charged_wall_seconds=100))
        common.write(out / 'evaluation_rows.json', rows)
        loaded = [analyze.load_shard(i) for i in range(4)]
        complete = analyze.render(common.DOCS, loaded, whole_matrix=True)
        assert complete == (shard == 3)
        if shard < 3:
            assert not (common.DOCS / 'summary.csv').exists()
    assert len(all_ids) == 288
    result = common.read(common.DOCS / 'analysis.json')
    assert result['status'] == 'complete' and result['model_tasks'] == 256 and result['native_tasks'] == 32
    assert result['no_actor_instances'] == [no_actor] and len(result['common_actor_instances']) == 15
    assert len(result['cohorts']) == 4
    # One invalid/duplicate/missing task never becomes a complete72/288 matrix.
    out = common.output(0)
    original = common.read(out / 'evaluation_rows.json')
    common.write(out / 'evaluation_rows.json', original[:-1])
    assert not analyze.load_shard(0)['complete']
    common.write(out / 'evaluation_rows.json', original + [original[0]])
    assert 'duplicate_or_unexpected_rows' in analyze.load_shard(0)['errors']
    altered = copy.deepcopy(original)
    altered[0]['sampling_seed'] += 1
    common.write(out / 'evaluation_rows.json', altered)
    assert any(m['reason'] == 'canonical_action_rng' for m in analyze.load_shard(0)['missing'])
    common.write(out / 'evaluation_rows.json', original)
    # Neutral/negative method results are still complete, with no post-test selection.
    for shard in range(4):
        out = common.output(shard)
        values = common.read(out / 'evaluation_rows.json')
        for t, r in zip(common.specs(shard), values):
            if t['variant'] == 'mlp_bc' and t['instance'] not in (no_actor, zero_pdi):
                r.update(pdi_horizon_midpoint=120., pdi_horizon_lower=119.99, pdi_horizon_upper=120.01)
        common.write(out / 'evaluation_rows.json', values)
    assert analyze.render(common.DOCS, [analyze.load_shard(i) for i in range(4)], whole_matrix=True)
    assert common.read(common.DOCS / 'analysis.json')['no_model_selection']
    # Final global summary waits for terminal supervisor commit, not just raw files.
    path = common.output(3) / 'job_state.json'
    job_state = common.read(path)
    job_state['status'] = 'analyzing'
    common.write(path, job_state)
    assert not analyze.render(root / 'before-terminal', [analyze.load_shard(i) for i in range(4)], whole_matrix=True)
    supervisor = module('test16_supervisor_fixture', PACKAGE / 'job.py')
    supervisor.SHARD = 0
    supervisor.OUT, supervisor.DOCS = root / 'lock-test', root / 'lock-docs'
    supervisor.OUT.mkdir()
    with (supervisor.OUT / 'custom.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert supervisor.run() == 2
    common.write(supervisor.OUT / 'bootstrap.json', {'budget_seconds': 3000})
    common.write(supervisor.OUT / 'job_state.json', {'status': 'complete', 'charged_wall_seconds': 100})
    assert supervisor.run() == 0 and not (supervisor.OUT / 'runner.log').exists()
    assert 'torch' not in sys.modules
    checks = ['288 unique tasks/four72-task shards', 'same-instance full method/native pairing',
        'partial shards cannot rank models', 'complete matrix auto aggregation', 'native optional callback fields',
        'zero-reference PDI ratios safe', 'common/no Actor split', 'heterogeneous CPU cohort reporting',
        'missing/duplicate/action-RNG mismatch fail closed', 'negative method result remains complete',
        'terminal supervisor required for final global report', 'concurrent duplicate lock', 'completed never restarts']
    common.write(PACKAGE / 'CHECKS.json', dict(passed=True, fixture_only=True, checks=checks,
        elapsed_seconds=time.monotonic() - started, fixture_directory=str(root),
        solver_tasks_executed=0, real_training_episodes=0, torch_imported=False))
    print(json.dumps(dict(passed=True, checks=len(checks), solver_tasks_executed=0, fixture_directory=str(root)), indent=2))


if __name__ == '__main__':
    main()
