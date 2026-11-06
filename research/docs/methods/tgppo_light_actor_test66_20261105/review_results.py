"""Read-only verification of completed66-instance records; no model/solver import."""
import csv
import datetime
import json
import math
from pathlib import Path
import statistics as st
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'tgppo/experiments/frozen-test66-20261105'
sys.path.insert(0, str(PACKAGE))
import common as c
import analyze


def aggregate(rows):
    proof = [bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in rows]
    calls = sum(r.get('branch_count', 0) for r in rows)
    timing = {key: sum(r.get('callback_seconds', {}).get(key, 0) for r in rows)
              for key in ('actor', 'features', 'callback')}
    wall = sum(r['optimize_wall_seconds'] for r in rows)
    return dict(tasks=len(rows), proof_count=sum(proof), proof_rate=st.mean(proof),
        pdi=st.mean(r['pdi_horizon_midpoint'] for r in rows),
        lower=st.mean(r['pdi_horizon_lower'] for r in rows), upper=st.mean(r['pdi_horizon_upper'] for r in rows),
        par2=st.mean(r['scip_solve_time'] if done else 240 for r, done in zip(rows, proof)),
        wall=wall / len(rows), nodes=st.mean(r['nnodes'] for r in rows), actor_calls=calls,
        actor_ms=timing['actor'] * 1000 / calls if calls else None,
        feature_ms=timing['features'] * 1000 / calls if calls else None,
        callback_ms=timing['callback'] * 1000 / calls if calls else None,
        actor_wall_fraction=timing['actor'] / wall, callback_wall_fraction=timing['callback'] / wall)


def delta(a, b):
    return dict(pdi_delta=a['pdi'] - b['pdi'],
        pdi_percent=100 * (a['pdi'] / b['pdi'] - 1) if b['pdi'] else None,
        lower=a['lower'] - b['upper'], upper=a['upper'] - b['lower'],
        proof_delta=a['proof_rate'] - b['proof_rate'], wall_percent=100 * (a['wall'] / b['wall'] - 1))


def main():
    inputs = c.verify()
    loaded = [analyze.load_shard(s) for s in c.SHARDS]
    assert all(l['complete'] and l['job']['status'] == 'complete' for l in loaded)
    source_paths = [PACKAGE / 'release.json', HERE / 'REPORT.md', HERE / 'analysis.json',
                    HERE / 'summary.csv', HERE / 'effects.csv', HERE / 'cohorts.csv']
    original_analysis = c.read(HERE / 'analysis.json')
    for path, digest in original_analysis['source_sha256'].items():
        assert c.sha(path) == digest, path
    shards = []
    for l in loaded:
        folder, manifest = c.output(l['shard']), l['manifest']
        expected = {r['job_id']: r for _, r in l['accepted']}
        with tarfile.open(folder / 'raw_results.tar', 'r') as archive:
            raw = {}
            for member in archive.getmembers():
                assert member.isfile() and member.name.startswith('t') and member.name.endswith('.json')
                row = json.load(archive.extractfile(member))
                assert row['job_id'] not in raw
                raw[row['job_id']] = row
            assert raw == expected
        for model in manifest['models'].values():
            assert c.sha(model['source']) == c.sha(model['frozen_actor']) == model['actor_sha256']
        probe = c.read(folder / 'cpu_fork_verification.json')
        assert probe['passed'] and probe['parent_threads'] == probe['child']['threads'] == 1
        assert not probe['child']['cuda_initialized']
        source_paths += [folder / n for n in ('job_state.json', 'status.json', 'manifest.json',
            'evaluation_rows.json', 'raw_results.tar', 'cpu_fork_verification.json')]
        shards.append(dict(shard=l['shard'], valid_tasks=len(expected), cpu=manifest['stack']['cpu_model'],
            host=manifest['host'], charged_wall_seconds=l['job']['charged_wall_seconds'],
            started_epoch=l['job']['started_epoch'], summed_worker_wall_seconds=l['engine']['summed_worker_wall_seconds']))
    accepted = [pair for l in loaded for pair in l['accepted']]
    assert len(accepted) == 1188
    names = sorted({t['instance'] for t, _ in accepted})
    old8 = {'bell5', 'bell3a', 'neos-1215259', 'neos-612125', 'sp98ir', 'neos-512201', 'n3seq24', 'bc1'}
    panels = dict(all=names, common_actor=original_analysis['common_actor_instances'],
        no_actor=original_analysis['no_actor_instances'], mixed_actor=original_analysis['mixed_actor_instances'],
        old8_same_run=sorted(old8), other58_same_run=sorted(set(names) - old8),
        without_ns1208400_sensitivity=[n for n in names if n != 'ns1208400'])
    groups, effects = {}, {}
    def select(variant, seed, subset):
        return [r for t, r in accepted if t['variant'] == variant and t['instance'] in subset
                and (seed == 'pooled' or t['training_seed'] == seed)]
    pairs = [('structure_noBC', 'mlp', 'b0'), ('structure_BC', 'mlp_bc', 'b0_bc'),
             ('BC_within_MLP', 'mlp_bc', 'mlp'), ('combined', 'mlp_bc', 'b0')]
    for label, subset in panels.items():
        groups[label] = {'native': {'pooled': aggregate(select('native', 'pooled', subset))}}
        for variant in c.VARIANTS:
            groups[label][variant] = {str(s): aggregate(select(variant, s, subset)) for s in [11, 22, 'pooled']}
        effects[label] = {kind: {str(s): delta(groups[label][a][str(s)], groups[label][b][str(s)])
                               for s in [11, 22, 'pooled']} for kind, a, b in pairs}
        effects[label]['versus_native'] = {v: delta(groups[label][v]['pooled'], groups[label]['native']['pooled'])
                                           for v in c.VARIANTS}
    # Independently recomputed columns must agree with the original automatic table.
    with (HERE / 'summary.csv').open() as stream:
        for row in csv.DictReader(stream):
            seed = row['training_seed'] or 'pooled'
            own = groups[row['panel']][row['variant']][seed]
            for key in ('pdi', 'lower', 'upper', 'proof_rate', 'par2', 'nodes', 'wall', 'actor_ms', 'feature_ms', 'callback_ms'):
                if row[key]:
                    assert math.isclose(own[key], float(row[key]), rel_tol=1e-11, abs_tol=1e-9), (row, key)
    instance_effects = []
    for kind, a, b in pairs:
        for name in names:
            entry = dict(effect=kind, instance=name, common_actor=name in panels['common_actor'],
                no_actor=name in panels['no_actor'], by_seed={})
            for seed in [11, 22, 'pooled']:
                entry['by_seed'][str(seed)] = delta(aggregate(select(a, seed, [name])), aggregate(select(b, seed, [name])))
            entry['contribution_to_full_mean_delta'] = entry['by_seed']['pooled']['pdi_delta'] / 66
            instance_effects.append(entry)
        assert math.isclose(sum(e['contribution_to_full_mean_delta'] for e in instance_effects if e['effect'] == kind),
                            effects['all'][kind]['pooled']['pdi_delta'], abs_tol=1e-8)
    signs = {}
    for kind, _, _ in pairs:
        values = [e for e in instance_effects if e['effect'] == kind]
        signs[kind] = dict(
            both_seeds_midpoint_improve=sum(all(e['by_seed'][str(s)]['pdi_delta'] < -1e-8 for s in c.SEEDS) for e in values),
            both_seeds_midpoint_regress=sum(all(e['by_seed'][str(s)]['pdi_delta'] > 1e-8 for s in c.SEEDS) for e in values),
            both_seeds_measurement_improve=sum(all(e['by_seed'][str(s)]['upper'] < 0 for s in c.SEEDS) for e in values),
            both_seeds_measurement_regress=sum(all(e['by_seed'][str(s)]['lower'] > 0 for s in c.SEEDS) for e in values))
    details = []
    for t, r in accepted:
        if t['instance'] in ('ns1208400', 'mspp16'):
            details.append(dict(label=t['label'], instance=t['instance'], solver_seed=t['solver_seed'],
                **{k: r.get(k) for k in ('status', 'proof_completed', 'original_optimal', 'scip_solve_time',
                    'scip_time_overshoot_seconds', 'branch_count', 'nnodes', 'pdi_horizon_midpoint',
                    'pdi_horizon_lower', 'pdi_horizon_upper', 'peak_rss_kib')}))
    supervised = []
    sup = ROOT / 'tgppo/runs/supervised-ppo-20261105'
    for variant in ('control', 'expert'):
        for seed in (11, 22):
            path = sup / 'jobs' / ('{}-s{}.json'.format(variant, seed))
            job = c.read(path)
            summary_path = sup / variant / 'training' / ('B0-s' + str(seed)) / 'summary.json'
            summary = c.read(summary_path)
            supervised.append(dict(variant=variant, seed=seed, status=job['status'],
                next_episode=job['next_episode'], complete_ppo_updates=summary['complete_ppo_updates'],
                gpu=summary['gpu'], optimizer_steps=summary['optimizer_steps']))
            source_paths += [path, summary_path]
    data = dict(verified_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        valid_tasks=1188, frozen_input_verification=inputs, all_raw_tar_records_equal=True,
        all_source_and_frozen_actors_unchanged=True, summaries_independently_recomputed=True,
        new_solver_tasks=0, new_training_episodes=0, torch_imported='torch' in sys.modules,
        shards=shards, groups=groups, effects=effects, panels=panels, instance_signs=signs,
        instance_effects=instance_effects, tail_details=details, supervised_training_snapshot=supervised,
        supervised_development_started=(sup / 'development-110/bootstrap.json').exists(),
        summed_job_hours=sum(s['charged_wall_seconds'] for s in shards) / 3600,
        parallel_elapsed_hours=(max(s['started_epoch'] + s['charged_wall_seconds'] for s in shards)
                                - min(s['started_epoch'] for s in shards)) / 3600,
        source_sha256={str(p.relative_to(ROOT)): c.sha(p) for p in source_paths})
    assert not data['torch_imported']
    c.write(HERE / 'REVIEW_DATA.json', data)
    print(json.dumps(dict(valid_tasks=1188, reports_verified=True, raw_tar_equal=True,
        new_solver_tasks=0, summed_job_hours=data['summed_job_hours'],
        main_effects=effects['all'], signs=signs), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
