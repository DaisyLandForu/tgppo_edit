"""Read-only result audit and derivation; no Torch import or solver execution."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = ROOT / 'tgppo/experiments/dev-curve-20261105'
sys.path.insert(0, str(PACKAGE))
from common import BASE, BUDGET, OUT, read, sha, specs, verify, write


def main():
    checks = verify()
    state = read(OUT / 'job_state.json')
    engine = read(OUT / 'status.json')
    manifest = read(OUT / 'manifest.json')
    analysis = read(HERE / 'analysis.json')
    rows = read(OUT / 'evaluation_rows.json')
    assert state['status'] == engine['status'] == analysis['status'] == 'complete'
    assert state['error'] is None and state['analysis_exit_code'] == 0
    assert state['charged_wall_seconds'] <= BUDGET and state['persistent_rows'] == len(rows) == 70
    assert engine['weights_unchanged'] and not engine['gpu'] and engine['training_steps'] == 0
    expected = specs()
    assert {r['job_id'] for r in rows} == {t['job_id'] for t in expected}
    assert len({r['job_id'] for r in rows}) == 70
    mapping = {r['job_id']: r for r in rows}
    tasks = {t['job_id']: t for t in manifest['tasks']}
    for item in expected:
        row, task = mapping[item['job_id']], tasks[item['job_id']]
        assert all(item[k] == task[k] for k in item)
        assert row['valid'] and row['worker_cpu_complete'] and not row['cuda_initialized']
        assert row['instance_name'] == item['instance'] and row['solver_seed'] == item['solver_seed']
        assert row['time_limit_seconds'] == row['pdi_horizon_seconds'] == 120
        assert set(row['cpu_affinity']).issubset(set(manifest['cpu_affinity']))
        seed = item['training_seed'] if item['training_seed'] is not None else 11
        sampling = int(hashlib.sha256((str(seed) + ':' + item['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
        assert row['sampling_seed'] == task['job']['sampling_seed'] == sampling
        if item['episode'] is not None:
            assert row['callback_seconds']['critic'] == row['fallback_action_count'] == 0
    for model in manifest['models'].values():
        assert sha(model['source']) == model['source_sha256']
        assert sha(model['frozen_actor']) == model['actor_sha256']
    with tarfile.open(OUT / 'raw_results.tar') as archive:
        members = archive.getmembers()
        assert len(members) == 70
        for member in members:
            row = json.load(archive.extractfile(member))
            assert row == mapping[row['job_id']]
    fork = read(OUT / 'cpu_fork_verification.json')
    assert fork['passed'] and fork['child']['threads'] == 1 and not fork['child']['cuda_initialized']
    for name, digest in analysis['source_sha256'].items():
        assert sha(name) == digest, name
    prior_count = 0
    for stage, key in [('continuation', 'source_output_sha256'), ('bc_only_ablation_20261105', 'source_sha256')]:
        prior = read(HERE.parent / stage / 'FINAL_AUDIT.json')
        for name, digest in prior[key].items():
            assert sha(ROOT / name) == digest, name
            prior_count += 1
    summary = list(csv.DictReader((HERE / 'summary.csv').open()))
    panels = {(r['episode'], r['training_seed'], r['panel']): r for r in summary}
    contrasts = []
    for tag in ('all', 'common_actor'):
        native = panels[('native', '', tag)]
        bc = panels[('0', 'pooled', tag)]
        for episode in ('0', '20', '100'):
            p = panels[(episode, 'pooled', tag)]
            contrasts.append(dict(panel=tag, episode=int(episode),
                pdi=float(p['pdi']), vs_bc_pdi_percent=100 * (float(p['pdi']) / float(bc['pdi']) - 1),
                vs_native_pdi_percent=100 * (float(p['pdi']) / float(native['pdi']) - 1),
                vs_bc_wall_percent=100 * (float(p['wall']) / float(bc['wall']) - 1),
                vs_native_wall_percent=100 * (float(p['wall']) / float(native['wall']) - 1)))
    grouped = {}
    for t in expected:
        grouped.setdefault(t['episode'], []).append(mapping[t['job_id']])
    base = grouped[0]
    decompositions = []
    for episode in (20, 100):
        for name in ('vpm2', 'dcmulti', 'eil33-2', 'neos-476283', 'roi5alpha10n8'):
            before = st.mean(r['pdi_horizon_midpoint'] for r in base if r['instance_name'] == name)
            after = st.mean(r['pdi_horizon_midpoint'] for r in grouped[episode] if r['instance_name'] == name)
            decompositions.append(dict(before_episode=0, after_episode=episode, instance=name,
                bc_pdi=before, after_pdi=after, instance_delta=after - before,
                contribution_to_full_mean_delta=(after - before) / 5,
                no_actor=all(r.get('branch_count', 0) == 0 for r in rows if r['instance_name'] == name and r['mode'] == 'eval')))
    with (HERE / 'mean_decomposition.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decompositions[0]))
        writer.writeheader()
        writer.writerows(decompositions)
    write(HERE / 'REVIEW_DATA.json', dict(contrasts=contrasts, mean_decomposition=decompositions,
        fact='PPO incremental solver gains are mixed across seeds; no convergence or failure-cause inference',
        extra_solver_tasks=0, extra_training_steps=0, gpu=False))
    paths = [OUT / 'job_state.json', OUT / 'status.json', OUT / 'manifest.json', OUT / 'evaluation_rows.json',
        OUT / 'raw_results.tar', OUT / 'cpu_fork_verification.json', PACKAGE / 'release.json',
        HERE / 'REPORT.md', HERE / 'analysis.json', HERE / 'RESULTS_REVIEW.md', HERE / 'NEXT_STAGE_PLAN.md',
        HERE / 'summary.csv', HERE / 'instances.csv', HERE / 'paired.csv', HERE / 'selection.csv',
        HERE / 'REVIEW_DATA.json', HERE / 'mean_decomposition.csv', Path(__file__).resolve()]
    audit = dict(verified_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        passed=True, model_tasks=60, native_tasks=10, valid_tasks=70,
        charged_wall_seconds=state['charged_wall_seconds'], authorized_seconds=BUDGET,
        input_checks=checks, archived_raw_records_identical=70, source_actors_unchanged=6,
        workers_cuda_uninitialized=True, critic_seconds_zero=True, policy_fallbacks_zero=True,
        canonical_development_rng_verified=True, actual_fork_probability_action_check_passed=True,
        prior_audited_sources_unchanged=prior_count, extra_training_steps=0, extra_solver_tasks=0,
        next_stage_started=False, source_output_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})
    write(HERE / 'FINAL_AUDIT.json', audit)
    assert 'torch' not in sys.modules
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_output_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
