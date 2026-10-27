"""Produce paired, instance-weighted stage-3 tables without dropping failures."""
import csv
import json
import math
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import torch
import numpy as np
import pyscipopt
import argparse
from project.runtime import load_agent, load_actor
from project.utils.protocol import sha256, write_json

ROOT = Path('/root/research')
RUN = ROOT / 'tgppo/runs/stage3'
DOC = ROOT / 'docs/stage3'


def read(path):
    return json.loads(path.read_text())


def csv_table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def decorate(row, method):
    r = dict(row)
    r['method'] = method
    r['proof_by_120'] = bool(r.get('proof_completed') and r.get('scip_solve_time', math.inf) <= 120.)
    r['par2_120'] = r['scip_solve_time'] if r['proof_by_120'] else 240.
    # Sensitivity bound for failed tasks, distinct from the valid-only primary mean.
    r['pdi_with_failure_worstcase'] = r['pdi_horizon_midpoint'] if r.get('valid') else 12000.
    return r


def mean(values):
    return statistics.mean(values) if values else None


def main():
    global RUN,DOC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--report-dir',type=Path,default=DOC)
    args=parser.parse_args()
    RUN,DOC=args.run_root.resolve(),args.report_dir.resolve()
    DOC.mkdir(parents=True,exist_ok=True)
    config = read(RUN / 'execution_config.json')
    native = [decorate(r, 'relpscost') for r in read(RUN / 'native/jobs/results.json')]
    evaluation = [decorate(r, 'B0') for r in read(RUN / 'evaluation/jobs/results.json')]
    collection = read(RUN / 'pilot/collection_results.json')
    updates = read(RUN / 'pilot/updates.json')
    native_summary = read(RUN / 'native/summary.json')
    original_pilot_summary = read(RUN / 'pilot/summary.json')
    pilot_summary = original_pilot_summary
    if (RUN / 'pilot-repaired/summary.json').exists():
        pilot_summary=read(RUN / 'pilot-repaired/summary.json')
        updates+=read(RUN / 'pilot-repaired/updates.json')
    eval_summary = read(RUN / 'evaluation/summary.json')
    assert len(native) == 75 and len(evaluation) == 15
    assert len({(r['instance_name'], r['solver_seed']) for r in native}) == 75
    assert len({(r['instance_name'], r['solver_seed']) for r in evaluation}) == 15
    source = {(r['instance_name'],r['solver_seed']):r for r in native}
    comparisons = []
    for b in evaluation:
        a = source[(b['instance_name'],b['solver_seed'])]
        good = bool(a.get('valid') and b.get('valid'))
        row = dict(instance=b['instance_name'], solver_seed=b['solver_seed'],
            paired_valid=good, native_status=a['status'], b0_status=b['status'],
            native_proof_by_120=a['proof_by_120'], b0_proof_by_120=b['proof_by_120'],
            native_par2=a['par2_120'], b0_par2=b['par2_120'],
            native_pdi=a.get('pdi_horizon_midpoint'), b0_pdi=b.get('pdi_horizon_midpoint'),
            native_gap_raw=a.get('gap'), b0_gap_raw=b.get('gap'),
            native_nodes=a.get('nnodes'), b0_nodes=b.get('nnodes'),
            native_worker_wall=a.get('worker_wall_seconds'), b0_worker_wall=b.get('worker_wall_seconds'),
            native_pdi_uncertainty=a.get('pdi_horizon_uncertainty'),
            b0_pdi_uncertainty=b.get('pdi_horizon_uncertainty'))
        if good:
            row.update(pdi_difference_b0_minus_native=b['pdi_horizon_midpoint']-a['pdi_horizon_midpoint'],
                pdi_difference_lower=b['pdi_horizon_lower']-a['pdi_horizon_upper'],
                pdi_difference_upper=b['pdi_horizon_upper']-a['pdi_horizon_lower'])
        comparisons.append(row)
    csv_table(DOC / 'development_paired.csv', comparisons)
    groups = []
    for name in config['development']:
        pairs = [r for r in comparisons if r['instance']==name]
        good = [r for r in pairs if r['paired_valid']]
        groups.append(dict(instance=name, paired_coverage=len(good), expected_pairs=3,
            native_pdi_mean=mean([r['native_pdi'] for r in good]),
            b0_pdi_mean=mean([r['b0_pdi'] for r in good]),
            pdi_difference_mean=mean([r['pdi_difference_b0_minus_native'] for r in good]),
            pdi_difference_lower=mean([r['pdi_difference_lower'] for r in good]),
            pdi_difference_upper=mean([r['pdi_difference_upper'] for r in good]),
            native_par2_mean=mean([r['native_par2'] for r in pairs]),
            b0_par2_mean=mean([r['b0_par2'] for r in pairs]),
            native_proofs=sum(r['native_proof_by_120'] for r in pairs),
            b0_proofs=sum(r['b0_proof_by_120'] for r in pairs)))
    csv_table(DOC / 'development_by_instance.csv', groups)
    flatkeys = ['method','instance_name','solver_seed','status','valid','proof_by_120','par2_120',
        'pdi_horizon_midpoint','pdi_horizon_lower','pdi_horizon_upper','pdi_horizon_uncertainty',
        'pdi_native_raw_before_flush','pdi_native_complete_at_stop','pdi_tail_completed_readonly',
        'pdi_with_failure_worstcase','gap','gap_finite','primal_bound','dual_bound',
        'has_feasible_solution','nnodes','scip_solve_time','optimize_wall_seconds',
        'worker_wall_seconds','worker_cpu_seconds','peak_rss_kib','measurement_callback_seconds',
        'scip_time_overshoot_seconds']
    csv_table(DOC / 'all_solver_runs.csv', [{k:r.get(k) for k in flatkeys} for r in native+evaluation])
    trajectory_stats=[]
    for row in collection:
        assert row['instance_name'] in config['train']
        if row.get('valid'):
            assert row['num_transitions']==row['branch_count']==row['reward_count']
            assert row['num_terminal_transitions']==int(row['branch_count']>0)
    for directory in sorted((RUN / 'pilot').glob('batch-*')):
        for path in sorted(directory.glob('*.pkl')):
            with path.open('rb') as stream:
                payload=pickle.load(stream)
            for e in payload['episodes']:
                lengths=[len(t['cands_state']) for t in e['transitions']]
                trajectory_stats.append(dict(episode_id=e['episode_id'], policy_version=e['policy_version'],
                    transitions=len(lengths), max_candidates=max(lengths,default=0),
                    mean_candidates=mean(lengths), terminal_transitions=sum(t['done'] for t in e['transitions']),
                    actual_state_bytes=sum(sum(t[k].nbytes for k in ('cands_state','mip_state','node_state'))
                                           for t in e['transitions']),
                    trajectory_file=str(path)))
    csv_table(DOC / 'trajectory_statistics.csv', trajectory_stats)
    actor, critic = [], []
    checkpoint=Path(read(RUN / 'selected_checkpoint.json')['checkpoint'])
    agent, payload=load_agent(checkpoint)
    assert sha256(checkpoint)==eval_summary['checkpoint_sha256']
    assert payload['memory']['num_transitions']==0
    initial=torch.load(RUN/'pilot/initial.pt',map_location='cpu')
    for key in ('actor','critic'):
        assert any(not torch.equal(v,initial[key][name]) for name,v in payload[key].items())
    restored_actor, _=load_actor(checkpoint)
    for name,value in agent.actor_network.state_dict().items():
        assert torch.isfinite(value).all()
        assert torch.equal(value,restored_actor.actor_network.state_dict()[name])
    assert all(torch.isfinite(v).all() for v in agent.critic_network.state_dict().values())
    shared=len({id(p) for p in agent.actor_network.parameters()} & {id(p) for p in agent.critic_network.parameters()})
    assert shared==0
    assert not hasattr(restored_actor,'critic_network')
    assert pilot_summary['completed_updates']>=1
    assert all(r.get('callback_seconds',{}).get('critic',0.)==0. for r in evaluation)
    original_hashes=read(ROOT/'docs/stage1/upstream_file_hashes.json')
    for relative,expected in original_hashes.items():
        assert sha256(ROOT/'tgppo/repo'/relative)==expected
    unchanged=['code/project/policy/actor.py','code/project/policy/critic.py',
        'code/project/policy/modules.py','code/project/reward/reward.py','code/project/utils/settings.py']
    for p in unchanged:
        assert sha256(ROOT/'tgppo/experiments/stage3'/p)==original_hashes[p]
    widths=[r['pdi_horizon_uncertainty'] for r in native+evaluation if r.get('valid')]
    good_groups=[g for g in groups if g['paired_coverage']==3]
    aggregate=dict(instances_with_complete_pairs=len(good_groups), expected_instances=5,
        native_pdi_instance_equal_mean=mean([g['native_pdi_mean'] for g in good_groups]),
        b0_pdi_instance_equal_mean=mean([g['b0_pdi_mean'] for g in good_groups]),
        paired_pdi_difference_mean=mean([g['pdi_difference_mean'] for g in good_groups]),
        native_par2_mean=mean([g['native_par2_mean'] for g in groups]),
        b0_par2_mean=mean([g['b0_par2_mean'] for g in groups]),
        native_proofs=sum(g['native_proofs'] for g in groups),
        b0_proofs=sum(g['b0_proofs'] for g in groups),
        robust_b0_better_instances=sum(g['pdi_difference_upper']<0 for g in good_groups),
        robust_b0_worse_instances=sum(g['pdi_difference_lower']>0 for g in good_groups))
    if good_groups:
        differences=np.asarray([g['pdi_difference_mean'] for g in good_groups])
        rng=np.random.RandomState(20261021)
        resamples=differences[rng.randint(0,len(differences),(10000,len(differences)))].mean(axis=1)
        aggregate['paired_difference_instance_bootstrap_95_percentile_ci']=[float(x) for x in np.percentile(resamples,[2.5,97.5])]
        aggregate['bootstrap_unit']='instance; 10000 resamples; 5-instance exploratory interval, no training-seed uncertainty'
    dev_native=[r for r in native if r['instance_name'] in config['development']]
    aggregate['native_pdi_failure_worstcase_mean']=mean([r['pdi_with_failure_worstcase'] for r in dev_native])
    aggregate['b0_pdi_failure_worstcase_mean']=mean([r['pdi_with_failure_worstcase'] for r in evaluation])
    tests=(ROOT/'docs/stage3/tests-final.log').read_text()
    passed=(all(r.get('valid') for r in native+evaluation) and native_summary['within_budget']
        and all(r.get('valid') for r in collection)
        and pilot_summary['stop_reason'] not in ('learning_error_last_valid_checkpoint_retained',
                  'invalid_collection_batch_retained_no_update','repair_error_last_complete_checkpoint_retained')
        and pilot_summary['within_budget'] and 'Ran 25 tests' in tests and tests.rstrip().endswith('OK'))
    report=dict(passed=passed, purpose='stage3_reduced_budget_B0_pilot', stage4_started=False,
        solver_horizon=120, native_jobs=len(native), native_valid=sum(bool(r.get('valid')) for r in native),
        development_jobs=len(evaluation), development_valid=sum(bool(r.get('valid')) for r in evaluation),
        aggregate=aggregate, native_budget=native_summary,pilot=pilot_summary,
        original_pilot_with_recorded_memory_failure=original_pilot_summary,evaluation=eval_summary,
        forecast_budget_guard=(read(RUN/'pilot-repaired/forecast_budget_guard.json')
                               if (RUN/'pilot-repaired/forecast_budget_guard.json').exists() else None),
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint), shared_parameter_objects=shared,
        model_architecture=payload['config'], updates=updates,
        collection_transitions=sum(r.get('num_transitions',0) for r in collection),
        max_candidates_observed=max(r['max_candidates'] for r in trajectory_stats),
        native_pdi_distinct_values=len({round(r['pdi_horizon_midpoint'],6) for r in native if r.get('valid')}),
        pdi_uncertainty_max=max(widths), pdi_uncertainty_mean=mean(widths),
        pdi_uncertainty_relative_full_range_max=max(widths)/12000.,
        measurement_versions=sorted({r.get('measurement_version') for r in native+evaluation}),
        binding=str(pyscipopt.__file__),python=sys.version,torch=torch.__version__,
        scip_shared_library_sha256=sha256(ROOT/'tgppo/solver/lib/libscip.so.6.0.1.0'),
        actor_only_evaluation=True, source_network_reward_settings_unchanged=unchanged)
    write_json(DOC/'verification.json',report)
    write_json(DOC/'development_summary.json',dict(aggregate=aggregate,instances=groups))
    print(json.dumps(dict(passed=passed,aggregate=aggregate,pilot=pilot_summary),indent=2))


if __name__=='__main__':
    main()
