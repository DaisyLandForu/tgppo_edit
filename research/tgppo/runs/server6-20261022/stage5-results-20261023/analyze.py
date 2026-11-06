#!/usr/bin/env python3
"""Read-only audit/aggregation of the frozen stage-5 server experiments.

Run with server-run/env.sh and Python 3.8. No solver or training is launched.
All writes are confined to this new results directory.
"""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from collections import Counter

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
LAUNCH = ROOT / 'migration/2026-10-22/server-run'
sys.path.insert(0, str(LAUNCH / 'mspp16-recovery'))
import recover as r

SOURCES = {}


def source(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = str(path.relative_to(ROOT))
    if relative in SOURCES:
        assert SOURCES[relative] == digest, relative
    SOURCES[relative] = digest
    return relative


def read(path):
    source(path)
    return json.loads(Path(path).read_text())


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    assert rows
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    # Each instance has the same number of repeats; still explicitly weight instances equally.
    names = sorted({x['instance_name'] for x in rows})
    def equal_mean(key):
        return float(np.mean([np.mean([x[key] for x in rows if x['instance_name'] == n]) for n in names]))
    proof = sum(x['proof_completed'] and x['scip_solve_time'] <= 120. for x in rows)
    par2 = float(np.mean([np.mean([x['scip_solve_time'] if x['proof_completed'] and x['scip_solve_time'] <= 120.
                                  else 240. for x in rows if x['instance_name'] == n]) for n in names]))
    return dict(instances=len(names), runs=len(rows), mean_pdi=equal_mean('pdi_horizon_midpoint'),
                mean_pdi_lower=equal_mean('pdi_horizon_lower'), mean_pdi_upper=equal_mean('pdi_horizon_upper'),
                mean_par2=par2, mean_nodes=equal_mean('nnodes'), proof_count=proof,
                proof_rate=proof/len(rows), valid_worker_cpu_seconds=sum(x['worker_cpu_seconds'] for x in rows),
                valid_worker_wall_seconds_sum=sum(x['worker_wall_seconds'] for x in rows),
                actor_callback_seconds=sum(x.get('callback_seconds', {}).get('actor', 0.) for x in rows))


def main():
    r.p.torch.set_num_threads(1)
    c = r.p.server.plan(ROOT, LAUNCH / 'protocol.json')
    run = ROOT / c['run_directory']
    work = run / 'test/mspp16-recovery-20261023'
    policy, audited, frozen = r.load_policy(ROOT, c, run, work)
    r.p.validate_dataset(ROOT, audited, 'development')
    r.p.validate_dataset(ROOT, audited, 'test66')
    for path in [LAUNCH/'protocol.json', work/'policy.json', run/'post-training-20261023/training_audit.json',
                 run/'post-training-20261023/frozen_selection.json', ROOT/'tgppo/datasets/cutoffs.json',
                 ROOT/'tgppo/runs/stage4-calibration/final_train_only_statistics.json',
                 ROOT/'tgppo/configs/paper_hparams.json']:
        source(path)
    for item in policy['protected_files']:
        assert r.p.sha256(ROOT/item['path']) == item['sha256']
        source(ROOT/item['path'])
    for path in sorted((ROOT/'tgppo/experiments/server-20261022/code').rglob('*.py')):
        source(path)
    names = sorted(audited['datasets']['test66'])
    reference_report = read(work/'reports/20261023T065206075850Z/comparison.json')
    assert reference_report['complete66'] and len(names) == 66
    original_count = recovered_count = 0
    by_model, by_method = {}, {}
    task_table, failures, receipts = [], [], []
    minimum_task_mtime = float('inf')
    for model in policy['models']:
        key = r.model_name(model['method'], model['seed'])
        original = r.original_rows(ROOT, model)
        recovered = r.recovered_rows(ROOT, c, work, model)
        assert recovered is not None
        rows = original + recovered
        r.valid_rows(rows, names, c['solver_seeds'], 'native' if key == 'native' else 'eval')
        original_count += len(original)
        recovered_count += len(recovered)
        by_model[key] = rows
        by_method.setdefault(model['method'], []).extend(rows)
        jobs = {}
        for item in [model['identity']] + model['manifests']:
            source(ROOT/item['path'])
        for item in model['manifests']:
            for job in read(ROOT/item['path'])['jobs']:
                jobs[job['job_id']] = job
        original_sources = {x['job_id']: x for x in model['valid_sources']}
        base = work/'tasks'/key
        receipt = read(base/'complete.json')
        receipts.append(dict(model=key, finished_at_utc=receipt['finished_at_utc'],
                             host=receipt['preflight']['host'], cpu=receipt['preflight']['stack']['cpu_model'],
                             effective_cpu_count=receipt['preflight']['effective_cpu_count'],
                             container_memory_bytes=receipt['preflight']['memory_limit_bytes']))
        recovered_sources = {}
        for directory in sorted(base.glob('attempt-*')):
            manifest = read(directory/'manifest.json')
            for job in manifest['jobs']:
                path = directory/(job['job_id']+'.json')
                value = read(path)
                if value.get('valid'):
                    recovered_sources[job['job_id']] = dict(path=str(path.relative_to(ROOT)), sha256=r.p.sha256(path))
        assert set(recovered_sources) == {x['job_id'] for x in recovered}
        for row in rows:
            job = jobs[row['job_id']]
            assert row['instance_name'] == job['instance_name'] and row['solver_seed'] == job['solver_seed']
            assert row['sampling_seed'] == job['sampling_seed']
            assert row['cutoff_applied'] and row['cutoff_verified'] and job['cutoff_provenance']['verified']
            expected_cutoff = audited['datasets']['test66'][row['instance_name']]['cutoff']['cutoff']
            assert job['cutoff'] == expected_cutoff
            assert key == 'native' or row['fallback_action_count'] == 0
            # Native floating-point snapshots can differ by ~2e-12 at 12000.
            assert row['pdi_horizon_lower']-1e-8 <= row['pdi_horizon_midpoint'] <= row['pdi_horizon_upper']+1e-8
            assert row['pdi_horizon_lower'] >= -1e-6 and row['pdi_horizon_upper'] <= 12000.000001
            is_retry = row['instance_name'] == 'mspp16'
            ref = (recovered_sources if is_retry else original_sources)[row['job_id']]
            path = ROOT/ref['path']
            assert r.p.sha256(path) == ref['sha256']
            source(path)
            minimum_task_mtime = min(minimum_task_mtime, path.stat().st_mtime)
            cb = row.get('callback_seconds', {})
            task_table.append(dict(model=key, method=model['method'], training_seed='' if key == 'native' else model['seed'],
                instance=row['instance_name'], solver_seed=row['solver_seed'], sampling_seed=row['sampling_seed'],
                pdi=row['pdi_horizon_midpoint'], pdi_lower=row['pdi_horizon_lower'], pdi_upper=row['pdi_horizon_upper'],
                proof_within_120s=bool(row['proof_completed'] and row['scip_solve_time'] <= 120.),
                proof_completed=row['proof_completed'], cutoff_proof=row['cutoff_proof'], original_optimal=row['original_optimal'],
                status=row['status'], solve_seconds=row['scip_solve_time'], nodes=row['nnodes'],
                worker_cpu_seconds=row['worker_cpu_seconds'], worker_wall_seconds=row['worker_wall_seconds'],
                load_seconds=row['load_wall_seconds'], optimize_wall_seconds=row['optimize_wall_seconds'],
                actor_seconds=cb.get('actor', 0.), base_feature_seconds=cb.get('features', 0.),
                entire_branch_callback_seconds=cb.get('callback', 0.), branch_count=row.get('branch_count', 0),
                peak_rss_kib=row['peak_rss_kib'], memory_exception=is_retry,
                address_space_limit_bytes=row['address_space_limit_bytes'], cutoff=job['cutoff'],
                source_path=ref['path'], source_sha256=ref['sha256']))
        for item in model['failures']:
            old = read(ROOT/item['source']['path'])
            failures.append(dict(model=key, job_id=item['job_id'], instance='mspp16',
                solver_seed=old.get('solver_seed', int(item['job_id'][-1])), status=item['status'], exitcode=item['exitcode'],
                measured_wall_seconds=item['wall_seconds'], worker_cpu_seconds=old.get('worker_cpu_seconds'),
                source_path=item['source']['path'], source_sha256=item['source']['sha256']))
    assert original_count == 1365 and recovered_count == 21 and len(task_table) == 1386
    assert len({(x['model'], x['instance'], x['solver_seed']) for x in task_table}) == 1386
    assert len(failures) == 21
    freeze_timestamp = dt.datetime.fromisoformat(frozen['frozen_at_utc']).timestamp()
    assert freeze_timestamp < minimum_task_mtime
    combined = [dict(method=m, **metrics(by_method[m])) for m in ['native', 'B0', 'R0', 'Rjoint']]
    for value in combined:
        expected = next(x for x in reference_report['combined_training_seeds'] if x['method'] == value['method'])
        for field in value:
            if field == 'method': continue
            assert math.isclose(value[field], expected[field], rel_tol=1e-11, abs_tol=1e-8), field
        value['pdi_change_percent_vs_B0'] = 100*(value['mean_pdi']/metrics(by_method['B0'])['mean_pdi'] - 1)
    per_seed = [dict(model=k, method=k.split('-s')[0] if k != 'native' else 'native',
                     training_seed='' if k == 'native' else int(k.split('-s')[1]), **metrics(v))
                for k, v in by_model.items()]
    instance_rows, im = [], {}
    for method in ['native', 'B0', 'R0', 'Rjoint']:
        im[method] = {}
        for name in names:
            value = metrics([x for x in by_method[method] if x['instance_name'] == name])
            im[method][name] = value
            instance_rows.append(dict(method=method, instance=name, **value))
    paired_table = []
    for name in names:
        row = dict(instance=name)
        for method in ['native', 'B0', 'R0', 'Rjoint']:
            row[method+'_pdi'] = im[method][name]['mean_pdi']
        for method in ['native', 'R0', 'Rjoint']:
            row[method+'_delta_vs_B0'] = row[method+'_pdi']-row['B0_pdi']
            row[method+'_delta_lower'] = im[method][name]['mean_pdi_lower']-im['B0'][name]['mean_pdi_upper']
            row[method+'_delta_upper'] = im[method][name]['mean_pdi_upper']-im['B0'][name]['mean_pdi_lower']
        paired_table.append(row)
    resamples = np.random.RandomState(20261023).randint(0, 66, size=(20000, 66))
    paired_stats = []
    for method in ['native', 'R0', 'Rjoint']:
        deltas = np.asarray([x[method+'_delta_vs_B0'] for x in paired_table])
        lower = np.asarray([x[method+'_delta_lower'] for x in paired_table])
        upper = np.asarray([x[method+'_delta_upper'] for x in paired_table])
        low, high = np.percentile(deltas[resamples].mean(1), [2.5, 97.5])
        paired_stats.append(dict(method=method, reference='B0', mean_delta=float(deltas.mean()),
            conditional_instance_bootstrap_95_low=float(low), conditional_instance_bootstrap_95_high=float(high),
            measurement_mean_delta_lower=float(lower.mean()), measurement_mean_delta_upper=float(upper.mean()),
            instances_point_lower=int((deltas < -1e-9).sum()), instances_point_equal=int((abs(deltas) <= 1e-9).sum()),
            instances_point_higher=int((deltas > 1e-9).sum()),
            instances_resolved_lower=int((upper < -1e-9).sum()), instances_resolved_higher=int((lower > 1e-9).sum()),
            instances_measurement_intervals_overlap=int(((upper >= -1e-9) & (lower <= 1e-9)).sum())))
    training, dev_curves, models, dev_cost = [], [], [], []
    for audit_job in audited['jobs']:
        r.p.validate_training_files(ROOT, audit_job)
        for relative in audit_job['files']:
            SOURCES[relative] = audit_job['files'][relative]
        group, seed = audit_job['group'], audit_job['seed']
        key = r.model_name(group, seed)
        base = run/'training'/key
        updates = read(base/'updates.json')
        effective = [x for x in updates if x['updated']]
        assert len(effective) == audit_job['complete_ppo_updates']
        assert audit_job['episodes'] == 500 and audit_job['pending_transitions'] == 0
        training.append(dict(model=key, episodes=500, effective_ppo_updates=len(effective),
            optimizer_steps=audit_job['optimizer_steps'], transitions=audit_job['transitions'],
            training_wall_hours=audit_job['wall_seconds']/3600,
            worker_cpu_core_hours=audit_job['worker_cpu_seconds']/3600,
            learner_wall_hours=audit_job['learner_update_wall_seconds']/3600,
            cuda_event_hours=sum(x.get('gpu_event_seconds', 0.) or 0. for x in effective)/3600,
            peak_cuda_allocated_gib=audit_job['peak_cuda_allocated_bytes']/1024**3,
            peak_cuda_reserved_gib=audit_job['peak_cuda_reserved_bytes']/1024**3,
            skipped_batches=json.dumps(audit_job['skipped_batches'], separators=(',', ':'))))
        selection = read(base/'selection.json')
        chosen = min(selection['all_checkpoints'], key=lambda x: (x['mean_pdi'], x['episode']))
        assert chosen == selection['best']
        export = next(x for x in frozen['models'] if x['group'] == group and x['seed'] == seed)
        models.append(dict(model=key, selected_episode=chosen['episode'], development_mean_pdi=chosen['mean_pdi'],
            actor_best_path=export['exports']['best']['path'], actor_best_sha256=export['exports']['best']['sha256'],
            actor_final_path=export['exports']['final']['path'], actor_final_sha256=export['exports']['final']['sha256']))
        for value in selection['all_checkpoints']:
            dev_curves.append(dict(model=key, method=group, training_seed=seed, **value))
            rows = read(base/'development'/('episode-{:04d}'.format(value['episode']))/'selected_results.json')
            r.valid_rows(rows, selection['selection_instances'], c['solver_seeds'], 'eval')
            assert math.isclose(metrics(rows)['mean_pdi'], value['mean_pdi'], rel_tol=1e-11)
            dev_cost.append(dict(model=key, episode=value['episode'], runs=len(rows),
                                 worker_cpu_seconds=sum(x['worker_cpu_seconds'] for x in rows),
                                 worker_wall_seconds_sum=sum(x['worker_wall_seconds'] for x in rows)))
        for path in [base/'actor_best.pt', base/'actor_final.pt']:
            source(path)
    deployment = []
    for method, rows in by_method.items():
        actor = sum(x.get('callback_seconds', {}).get('actor', 0.) for x in rows)
        branches = sum(x.get('branch_count', 0) for x in rows)
        wall = sum(x['worker_wall_seconds'] for x in rows)
        optimize = sum(x['optimize_wall_seconds'] for x in rows)
        callback = sum(x.get('callback_seconds', {}).get('callback', 0.) for x in rows)
        deployment.append(dict(method=method, runs=len(rows), branches=branches, actor_seconds=actor,
            actor_milliseconds_per_branch=1000*actor/branches if branches else None,
            actor_fraction_of_worker_wall=actor/wall,
            base_feature_seconds=sum(x.get('callback_seconds', {}).get('features', 0.) for x in rows),
            branch_action_seconds=sum(x.get('callback_seconds', {}).get('branch', 0.) for x in rows),
            entire_branch_callback_seconds=callback, callback_fraction_of_optimize_wall=callback/optimize,
            optimize_wall_seconds=optimize, worker_wall_seconds_sum=wall,
            critic_seconds=sum(x.get('callback_seconds', {}).get('critic', 0.) for x in rows)))
    # Keep all logs in the ledger. Log windows are a proxy, not scheduler accounting.
    log_inventory = []
    for path in sorted((run/'custom-logs').glob('*.log')):
        text = path.read_text(errors='replace')
        source(path)
        matches = re.findall(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:\+00:00|Z)', text[:1500])
        elapsed = None
        if matches:
            started = dt.datetime.fromisoformat(matches[0].replace('Z', '+00:00')).timestamp()
            elapsed = max(0., path.stat().st_mtime-started)
        is_target = '"cpu_model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz"' in text
        log_inventory.append(dict(path=str(path.relative_to(ROOT)), log_window_seconds_proxy=elapsed,
            gold_preflight_present=is_target, cpu_mismatch='CPU/software differs from training' in text,
            lock_rejection='already running' in text.lower() or 'another process' in text.lower(),
            recovery_passed='MSPP16_RECOVERED' in text, memory_error='Insufficient memory' in text,
            final_lines=' | '.join(text.splitlines()[-2:])[:500]))
    aux = []
    for path in sorted((run/'aux-gpu-training').glob('*/summary.json')):
        value = read(path)
        aux.append(dict(path=str(path.relative_to(ROOT)), **value))
    failed_wall = sum(x['measured_wall_seconds'] or 0. for x in failures)
    costs = dict(training_gpu_allocation_process_wall_hours=sum(x['training_wall_hours'] for x in training),
        training_collection_cpu_core_hours=sum(x['worker_cpu_core_hours'] for x in training),
        training_learner_wall_hours=sum(x['learner_wall_hours'] for x in training),
        training_cuda_event_hours=sum(x['cuda_event_hours'] for x in training),
        development_valid_runs=sum(x['runs'] for x in dev_cost),
        development_valid_worker_cpu_core_hours=sum(x['worker_cpu_seconds'] for x in dev_cost)/3600,
        test_valid_runs=len(task_table), test_valid_worker_cpu_core_hours=sum(x['worker_cpu_seconds'] for x in task_table)/3600,
        test_recovery_valid_worker_cpu_core_hours=sum(x['worker_cpu_seconds'] for x in task_table if x['memory_exception'])/3600,
        old_mspp16_failed_runs=len(failures), old_mspp16_failed_worker_wall_seconds_sum=failed_wall,
        old_mspp16_failed_worker_cpu_core_hours=None,
        auxiliary_synthetic_gpu_training=aux,
        accounting_limit='Worker CPU core-hours, summed worker wall and allocated GPU wall are different quantities. '
          'SCIP/Actor development and test use no CUDA. Crater GPU allocation/billing, queue/container overhead, '
          'failed worker CPU counters and full preparation costs are not available in these records; do not report these as total billed cost.')
    mspp = [x for x in task_table if x['memory_exception']]
    assert len(mspp) == 21 and all(x['pdi'] == 12000. and x['branch_count'] == 0 and x['actor_seconds'] == 0. for x in mspp)
    audit = dict(observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_protocol_sha256=r.p.fingerprint(c), frozen_selection_sha256=r.p.sha256(run/'post-training-20261023/frozen_selection.json'),
        recovery_policy_sha256=r.p.sha256(work/'policy.json'),
        checks=dict(valid_unique_final_test_records=1386, original_valid_records=1365, recovered_valid_records=21,
            preserved_original_failed_records=21, instances=66, solver_seeds=[0,1,2], training_seeds=[11,22],
            groups=['B0','R0','Rjoint'], same_gold_software_stack=True, same_cutoffs_and_action_seeds=True,
            source_hashes_checked=True, actor_best_and_final_hashes_checked=True, training_audit_files_hashes_checked=True,
            frozen_before_all_final_task_result_files=True, all_critics_disabled=True, all_search_extensions_disabled=True,
            all_cuda_disabled_in_evaluation=True, all_actor_fallback_action_counts_zero=True,
            solver_limit_seconds=120, valid_development_records=450, independent_compatible_dataset_supplied=False),
        stack=audited['stack'], resource_exception=policy['resource_exception'], recovery_receipts=receipts,
        final_test_status_counts=dict(Counter(x['status'] for x in task_table)),
        old_failure_status_counts=dict(Counter(x['status'] for x in failures)),
        bootstrap=dict(unit='paired instance; both fixed training seeds and three solver seeds averaged within instance',
            instances=66, resamples=20000, numpy_random_state_seed=20261023,
            limitation='Conditional on six trained and dev-selected Actors; does not estimate variability over new training seeds. '
                'Exploratory percentile interval on PDI midpoints; distinct from deterministic measurement bounds.'),
        stage5_status='Authorized six-model test matrix complete with disclosed resource exception; delivery awaits user acceptance.',
        stage6_started=False)
    write_json('audit.json', audit)
    write_json('summary.json', dict(combined=combined, per_training_seed=per_seed, paired_instance_comparisons=paired_stats,
                                  selected_models=models, training=training, development_cost=dev_cost,
                                  deployment=deployment, costs=costs, development_curves=dev_curves))
    for name, rows in [('main_results.csv',combined), ('per_training_seed.csv',per_seed), ('per_instance.csv',instance_rows),
                       ('paired_instances.csv',paired_table), ('paired_uncertainty.csv',paired_stats),
                       ('per_task_with_sources.csv',task_table), ('preserved_failures.csv',failures),
                       ('selected_models.csv',models), ('training_cost.csv',training), ('development_curves.csv',dev_curves),
                       ('deployment_cost.csv',deployment), ('custom_log_inventory.csv',log_inventory)]:
        write_csv(name, rows)
    write_json('source_ledger.json', dict(algorithm='sha256', sources=SOURCES,
        note='Audited raw inputs/weights/logs are read-only; output hashes are recorded separately after reporting.'))
    print(json.dumps(dict(audit=audit['checks'], main=combined, paired=paired_stats, costs=costs, deployment=deployment), indent=2))


if __name__ == '__main__':
    main()
