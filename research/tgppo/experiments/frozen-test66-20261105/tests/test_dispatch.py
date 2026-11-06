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
from unittest import mock

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
    root = Path(tempfile.mkdtemp(prefix='tgppo-test66-check-'))
    common.RUN, common.DOCS = root / 'run', root / 'reports'
    analyze = module('test66_analysis_fixture', PACKAGE / 'analyze.py')
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
    for shard in common.SHARDS:
        policy = common.resources(shard)
        tasks, rows = [], []
        for t in common.specs(shard):
            row_policy = common.task_resources(t['instance'])
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
                address_space_limit_bytes=row_policy['address_space_limit_bytes'], cpu_affinity=list(range(row_policy['workers'])),
                callback_seconds=dict(actor=.01, features=.02, callback=.04, critic=0), fallback_action_count=0)
            if t['variant'] == 'native':
                for key in ('callback_seconds', 'fallback_action_count'):
                    r.pop(key)
            job = common.apply_resource_policy(dict(job_id=t['job_id'], sampling_seed=sampling, instance_name=t['instance']), shard)
            tasks.append(dict(t, job=job))
            rows.append(r)
        out = common.output(shard)
        manifest = dict(tasks=tasks, models={m['label']: dict(m, source=m['actor']) for m in common.PLAN['frozen_models']},
            selected=common.selected(shard), stack=dict(cpu_model='FIXTURE' + str(shard)), host='fixture-' + str(shard),
            resource_policy=policy, workers=policy['workers'], cpu_affinity=list(range(policy['workers'])),
            runner_sha256=common.sha(fake / 'run.py'), protocol_sha256=common.sha(fake / 'docs/PROTOCOL.md'),
            selection_sha256=common.sha(fake / 'selection.json'))
        common.write(out / 'manifest.json', manifest)
        common.write(out / 'status.json', dict(status='complete', weights_unchanged=True, summed_worker_wall_seconds=common.task_count(shard)))
        common.write(out / 'job_state.json', dict(status='complete', engine_complete=True, final_input_verification={'passed': True}, error=None, charged_wall_seconds=100))
        common.write(out / 'evaluation_rows.json', rows)
        loaded = [analyze.load_shard(i) for i in common.SHARDS]
        complete = analyze.render(common.DOCS, loaded, whole_matrix=True)
        assert complete == (shard == 3)
        if shard < 3:
            assert not (common.DOCS / 'summary.csv').exists()
    assert len(all_ids) == 1188
    result = common.read(common.DOCS / 'analysis.json')
    assert result['status'] == 'complete' and result['model_tasks'] == 1056 and result['native_tasks'] == 132
    assert result['no_actor_instances'] == [no_actor] and len(result['common_actor_instances']) == 65
    assert len(result['cohorts']) == 4 and result['full66_complete']
    # One invalid/duplicate/missing task never becomes a complete1188 matrix.
    out = common.output(0)
    original = common.read(out / 'evaluation_rows.json')
    common.write(out / 'evaluation_rows.json', original[:-1])
    assert not analyze.load_shard(0)['complete']
    assert not analyze.render(common.DOCS, [analyze.load_shard(i) for i in common.SHARDS], whole_matrix=True)
    assert not (common.DOCS / 'summary.csv').exists()
    common.write(out / 'evaluation_rows.json', original + [original[0]])
    assert 'duplicate_or_unexpected_rows' in analyze.load_shard(0)['errors']
    altered = copy.deepcopy(original)
    altered[0]['sampling_seed'] += 1
    common.write(out / 'evaluation_rows.json', altered)
    assert any(m['reason'] == 'canonical_action_rng' for m in analyze.load_shard(0)['missing'])
    common.write(out / 'evaluation_rows.json', original)
    # Neutral/negative method results are still complete, with no post-test selection.
    for shard in common.SHARDS:
        out = common.output(shard)
        values = common.read(out / 'evaluation_rows.json')
        for t, r in zip(common.specs(shard), values):
            if t['variant'] == 'mlp_bc' and t['instance'] not in (no_actor, zero_pdi):
                r.update(pdi_horizon_midpoint=120., pdi_horizon_lower=119.99, pdi_horizon_upper=120.01)
        common.write(out / 'evaluation_rows.json', values)
    assert analyze.render(common.DOCS, [analyze.load_shard(i) for i in common.SHARDS], whole_matrix=True)
    assert common.read(common.DOCS / 'analysis.json')['no_model_selection']
    # Final global summary waits for terminal supervisor commit, not just raw files.
    path = common.output(3) / 'job_state.json'
    job_state = common.read(path)
    job_state['status'] = 'analyzing'
    common.write(path, job_state)
    assert not analyze.render(root / 'before-terminal', [analyze.load_shard(i) for i in common.SHARDS], whole_matrix=True)
    job_state['status'] = 'complete'
    common.write(path, job_state)
    # Large-instance policy must apply equally to all16 models and both native tasks.
    large = common.output(3)
    large_rows = common.read(large / 'evaluation_rows.json')
    large_only = [r for r in large_rows if r['instance_name'] == 'mspp16']
    assert len(large_only) == 18 and {r['address_space_limit_bytes'] for r in large_only} == {24*1024**3}
    assert all(r['cpu_affinity'] == [0] for r in large_only)
    damaged = copy.deepcopy(large_rows)
    next(r for r in damaged if r['instance_name'] == 'mspp16')['address_space_limit_bytes'] = 8*1024**3
    common.write(large / 'evaluation_rows.json', damaged)
    assert any(x['reason'] == 'resource_exception_or_affinity' for x in analyze.load_shard(3)['missing'])
    common.write(large / 'evaluation_rows.json', large_rows)
    over_budget = dict(job_state, charged_wall_seconds=common.BUDGET + 1)
    common.write(path, over_budget)
    assert 'per_shard_budget_exceeded' in analyze.load_shard(3)['errors']
    common.write(path, job_state)
    for name, shard in [('mspp16', 0), (common.selected(0)[0]['instance'], 3)]:
        try:
            common.apply_resource_policy(dict(instance_name=name), shard)
        except ValueError:
            pass
        else:
            raise AssertionError('Resource exception leaked to another instance')
    # Exercise the actual phase dispatcher with fake solves; no ordinary worker
    # remains active when mspp16 starts, and CPU affinity restores on success/error.
    engine = module('test66_engine_fixture', PACKAGE / 'run.py')
    tasks = common.read(large / 'manifest.json')['tasks']
    trace = []
    cores = [0, 1, 2, 3]
    affinity = list(cores)
    def set_affinity(pid, values):
        affinity[:] = values
    class RuntimeFixture:
        def run_jobs(self, jobs, directory, max_workers, watchdog_seconds):
            large_phase = jobs[0]['instance_name'] == 'mspp16'
            assert all((j['instance_name'] == 'mspp16') == large_phase for j in jobs)
            assert max_workers == (1 if large_phase else 4)
            assert watchdog_seconds == (300 if large_phase else 180)
            assert affinity == cores[:max_workers]
            trace.append(('serial' if large_phase else 'ordinary', len(jobs)))
            return [{'job_id': j['job_id']} for j in jobs]
    with mock.patch.object(engine.os, 'sched_setaffinity', side_effect=set_affinity):
        values = engine.evaluate_batches(RuntimeFixture(), tasks, root / 'phase-test', cores, {})
        assert trace == [('ordinary', 234), ('serial', 18)] and len(values) == 252 and affinity == cores
        class FailingRuntime(RuntimeFixture):
            def run_jobs(self, *args, **kwargs):
                raise RuntimeError('Injected failure')
        try:
            engine.evaluate_batches(FailingRuntime(), tasks, root / 'phase-error', cores, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Dispatcher swallowed worker failure')
        assert affinity == cores
    supervisor = module('test66_supervisor_fixture', PACKAGE / 'job.py')
    supervisor.SHARD = 0
    supervisor.OUT, supervisor.DOCS = root / 'lock-test', root / 'lock-docs'
    supervisor.OUT.mkdir()
    with (supervisor.OUT / 'custom.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert supervisor.run() == 2
    common.write(supervisor.OUT / 'bootstrap.json', {'budget_seconds': common.BUDGET})
    common.write(supervisor.OUT / 'job_state.json', {'status': 'complete', 'charged_wall_seconds': 100})
    assert supervisor.run() == 0 and not (supervisor.OUT / 'runner.log').exists()
    assert 'torch' not in sys.modules
    checks = ['1188 unique tasks/four shards covering exactly66 instances', 'same-instance full method/native pairing',
        'partial shards cannot rank models', 'complete matrix auto aggregation', 'native optional callback fields',
        'zero-reference PDI ratios safe', 'common/no Actor split', 'heterogeneous CPU cohort reporting',
        'missing/duplicate/action-RNG mismatch fail closed', 'negative method result remains complete',
        'terminal supervisor required for final global report', 'concurrent duplicate lock', 'completed never restarts',
        'mspp16 all18 tasks24GiB/single CPU and exceptions cannot leak', 'budget overrun cannot pass complete',
        'incomplete reanalysis removes stale complete ranking', 'actual phase dispatcher serializes mspp16 and restores CPU affinity']
    common.write(PACKAGE / 'CHECKS.json', dict(passed=True, fixture_only=True, checks=checks,
        elapsed_seconds=time.monotonic() - started, fixture_directory=str(root),
        solver_tasks_executed=0, real_training_episodes=0, torch_imported=False))
    print(json.dumps(dict(passed=True, checks=len(checks), solver_tasks_executed=0, fixture_directory=str(root)), indent=2))


if __name__ == '__main__':
    main()
