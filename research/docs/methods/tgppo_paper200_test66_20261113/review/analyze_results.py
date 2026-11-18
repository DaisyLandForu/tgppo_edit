"""Read-only review of the completed frozen200 evaluation; no torch or SCIP."""
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'tgppo/experiments').is_dir())
PACKAGE = ROOT / 'tgppo/experiments/paper200-test66-20261113'
DOCS = HERE.parent
sys.path.insert(0, str(PACKAGE))
from common import LABELS, RUN, SHARDS, plan, read, scan, sha, verify, write
from analyze import paired, statistics, sgm


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    verified = verify()
    protected = [DOCS / name for name in ('REPORT.md', 'summary.csv', 'paired.csv',
        'independent_effects.csv', 'paper_subset_metrics.csv', 'raw_rows.csv', 'results.json')]
    for model in plan()['models']:
        assert sha(ROOT / model['checkpoint']) == model['checkpoint_sha256']
        protected.extend(ROOT / model[key] for key in ('checkpoint', 'source_actor', 'frozen_actor'))
    before = {str(p): sha(p) for p in protected}
    source_outputs = [DOCS / name for name in ('results.json', 'summary.csv', 'paired.csv',
        'independent_effects.csv', 'paper_subset_metrics.csv', 'raw_rows.csv', 'REPORT.md')]
    before = {str(p): sha(p) for p in source_outputs}
    rows, shards, failed = [], [], []
    for shard in SHARDS:
        folder = RUN / ('shard-' + str(shard))
        with (folder / 'execution.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            panels, cohorts = scan(shard)
            state = read(folder / 'status.json')
            assert state['status'] == 'complete'
            rows.extend(r for values in panels.values() for r in values)
            for p in (folder / 'attempts').glob('*/raw/*.json'):
                value = read(p)
                if isinstance(value, dict) and 'job_id' in value and not value.get('valid'):
                    failed.append(str(p))
            shards.append(dict(shard=shard, valid_tasks=5*len(panels), state=state,
                environments={cid: c['environment'] for cid, c in cohorts.items()}))
    assert len(rows) == len({(r['label'], r['instance_name'], r['solver_seed']) for r in rows}) == 330
    assert all(sum(r['label'] == label for r in rows) == 66 for label in LABELS)
    summaries = {label: statistics([r for r in rows if r['label'] == label], label) for label in LABELS}
    original = {r['label']: r for r in read(DOCS / 'results.json')['summary']}
    for label, summary in summaries.items():
        for key, value in summary.items():
            expected = original[label][key]
            if isinstance(value, (int, float)):
                assert abs(value-expected) <= 1e-8*max(1, abs(expected)), (label, key)
            else:
                assert value == expected
    effects = paired(rows)
    main_pairs = [('mlp_bc_s11', 'mlp_s11', 'BC_seed11'),
        ('mlp_bc_s22', 'b0_new_s22', 'combined_seed22')]
    subset = read(ROOT / 'tgppo/experiments/paper-aligned-20261106/docs/author_subset_audit.json')
    easy = set(subset['finished_instances.xlsx']['instances'])
    hard = set(subset['timelimit_instances.xlsx']['instances'])
    comparisons, contributions = [], []
    bootstrap = np.random.RandomState(20261116).randint(0, 66, size=(10000, 66))
    for method, reference, effect in main_pairs:
        group = sorted((e for e in effects if e['effect'] == effect), key=lambda e: e['instance'])
        a = np.array([e['method_pdi'] for e in group])
        b = np.array([e['reference_pdi'] for e in group])
        d = a-b
        sampled_difference = d[bootstrap].mean(axis=1)
        sampled_reference = b[bootstrap].mean(axis=1)
        sampled_relative = 100*sampled_difference/sampled_reference
        gains = [e['instance'] for e in group if e['method_proved'] and not e['reference_proved']]
        losses = [e['instance'] for e in group if e['reference_proved'] and not e['method_proved']]
        best = int(np.argmin(d))
        kept = np.arange(66) != best
        m, r = summaries[method], summaries[reference]
        result = dict(effect=effect, method=method, reference=reference, instances=66,
            relative_pdi_change_percent=100*(a.mean()/b.mean()-1),
            absolute_mean_pdi_change=float(d.mean()),
            midpoint_wins=int((d < 0).sum()), midpoint_losses=int((d > 0).sum()),
            separated_wins=sum(e['pdi_difference_upper'] < 0 for e in group),
            separated_losses=sum(e['pdi_difference_lower'] > 0 for e in group),
            method_proofs=m['proofs'], reference_proofs=r['proofs'],
            proof_gains=gains, proof_losses=losses,
            optimize_wall_change_percent=100*(m['mean_optimize_seconds']/r['mean_optimize_seconds']-1),
            all66_node_sgm_change_percent=100*(m['sgm_nodes_shift100']/r['sgm_nodes_shift100']-1),
            observed_actor_decision_time_ratio=r['actor_mean_ms']/m['actor_mean_ms'],
            exploratory_mean_difference_bootstrap95=np.quantile(sampled_difference, [.025, .975]).tolist(),
            exploratory_relative_change_bootstrap95=np.quantile(sampled_relative, [.025, .975]).tolist(),
            largest_improvement_instance=group[best]['instance'],
            largest_improvement_fraction_of_net_gain=float(d[best]/d.sum()),
            relative_pdi_change_without_largest_improvement_percent=100*(a[kept].mean()/b[kept].mean()-1))
        for names, field, metric in [(easy, 'easy33_node_sgm_change_percent', 'nnodes'),
                (hard, 'hard32_pdi_sgm_change_percent', 'pdi_horizon_midpoint')]:
            aa = [row[metric] for row in rows if row['label'] == method and row['instance_name'] in names]
            bb = [row[metric] for row in rows if row['label'] == reference and row['instance_name'] in names]
            shift = 100 if metric == 'nnodes' else 0
            result[field] = 100*(sgm(aa, shift)/sgm(bb, shift)-1)
        comparisons.append(result)
        for e in group:
            contributions.append(dict(effect=effect, instance=e['instance'],
                method_pdi=e['method_pdi'], reference_pdi=e['reference_pdi'],
                pdi_change=e['mean_pdi_difference'], net_benefit_contribution=-e['mean_pdi_difference']/66,
                proof_gain=int(e['method_proved'])-int(e['reference_proved'])))
    assert not failed
    assert all(sha(p) == digest for p, digest in before.items())
    assert all(sha(r['result_path']) == r['result_sha256'] for r in rows)
    old = read(ROOT / 'docs/methods/tgppo_paper100_test66_20261110/results.json')
    old_native = next(r for r in old['summary'] if r['label'] == 'native')
    current_native = summaries['native']
    result = dict(passed=True, reviewed_beijing=datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))).isoformat(), verification=verified,
        valid_tasks=330, invalid_results=failed, complete_instances=66, summaries=summaries,
        comparisons=comparisons, shards=shards,
        all_four_actors_used_on_all66=all(r['branch_count'] > 0 for r in rows if r['label'] != 'native'),
        baseline_mean_pdi_change_vs_old100_percent=100*(current_native['mean_pdi']/old_native['mean_pdi']-1),
        bootstrap=dict(scope='exploratory, unadjusted instance percentile bootstrap; not preregistered',
            replicates=10000, seed=20261116, unit='instance', confidence=.95,
            pdi_measurement_bounds_are_not_confidence_intervals=True),
        original_output_hashes_unchanged=before,
        source_checkpoints_unchanged=True, new_training_steps=0, new_solver_calls=0)
    write(HERE / 'REVIEW.json', result)
    csv_write(HERE / 'instance_contributions.csv', sorted(contributions,
        key=lambda e: (e['effect'], -e['net_benefit_contribution'])))
    print(json.dumps(dict(valid_tasks=330, comparisons=comparisons,
        native_change_percent=result['baseline_mean_pdi_change_vs_old100_percent']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
