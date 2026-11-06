"""Read existing server6 artifacts; never import SCIP or launch a solve.

Run from research/: python docs/diagnostics/tgppo_20261030/scripts/analyze_existing.py
"""
import csv
import hashlib
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'tgppo/runs/server6-20261022'
OLD = RUN / 'stage5-results-20261023'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def readcsv(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def mean(v):
    return float(np.mean(v)) if len(v) else None


def quantiles(v):
    return dict(zip(['min', 'p50', 'p95', 'max'], map(float, np.quantile(v, [0, .5, .95, 1])))) if len(v) else {}


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) and np.std(y) else None


def main():
    start = time.perf_counter()
    ledger = json.loads((OLD / 'source_ledger.json').read_text())['sources']
    hashes = []
    for path, expected in ledger.items():
        p = ROOT / path
        actual = sha(p) if p.is_file() else None
        hashes.append(dict(path=path, expected_sha256=expected, actual_sha256=actual,
                           status='match' if actual == expected else 'missing' if actual is None else 'mismatch'))
    writecsv('source_hash_audit.csv', hashes)
    print('hashes', dict(Counter(r['status'] for r in hashes)), flush=True)
    tasks = readcsv(OLD / 'per_task_with_sources.csv')
    assert len(tasks) == len({(r['model'], r['instance'], r['solver_seed']) for r in tasks}) == 1386
    for r in tasks:
        for k in ['pdi', 'pdi_lower', 'pdi_upper', 'nodes', 'solve_seconds', 'worker_wall_seconds',
                  'actor_seconds', 'base_feature_seconds', 'entire_branch_callback_seconds',
                  'optimize_wall_seconds', 'branch_count']:
            r[k] = float(r[k])
        r['proof'] = r['proof_within_120s'] == 'True'
    by = defaultdict(list)
    for r in tasks:
        by[r['method'], r['instance']].append(r)
    matrix = []
    for name in sorted({r['instance'] for r in tasks}):
        n, b = by['native', name], by['B0', name]
        npdi, bpdi = mean([r['pdi'] for r in n]), mean([r['pdi'] for r in b])
        row = dict(instance=name, training_covered=False, problem_family='unknown',
                   native_proof_rate=mean([r['proof'] for r in n]), b0_proof_rate=mean([r['proof'] for r in b]),
                   native_pdi=npdi, b0_pdi=bpdi, b0_minus_native=bpdi-npdi,
                   measurement_delta_low=mean([r['pdi_lower'] for r in b])-mean([r['pdi_upper'] for r in n]),
                   measurement_delta_high=mean([r['pdi_upper'] for r in b])-mean([r['pdi_lower'] for r in n]),
                   native_nodes=mean([r['nodes'] for r in n]), b0_nodes=mean([r['nodes'] for r in b]),
                   native_nodes_per_optimize_second=sum(r['nodes'] for r in n)/sum(r['optimize_wall_seconds'] for r in n),
                   b0_nodes_per_optimize_second=sum(r['nodes'] for r in b)/sum(r['optimize_wall_seconds'] for r in b),
                   b0_branches=sum(r['branch_count'] for r in b),
                   b0_actor_fraction_optimize=sum(r['actor_seconds'] for r in b)/sum(r['optimize_wall_seconds'] for r in b),
                   b0_callback_fraction_optimize=sum(r['entire_branch_callback_seconds'] for r in b)/sum(r['optimize_wall_seconds'] for r in b),
                   b0_mean_ms_per_branch=1000*sum(r['actor_seconds'] for r in b)/sum(r['branch_count'] for r in b) if sum(r['branch_count'] for r in b) else None,
                   native_statuses=';'.join(f'{k}:{v}' for k,v in sorted(Counter(r['status'] for r in n).items())),
                   b0_statuses=';'.join(f'{k}:{v}' for k,v in sorted(Counter(r['status'] for r in b).items())))
        matrix.append(row)
    writecsv('per_instance_diagnostic_matrix.csv', matrix)
    cohorts = []
    for label, rows in [('all', matrix), ('native_all_proven', [r for r in matrix if r['native_proof_rate']==1]),
                        ('native_some_proven', [r for r in matrix if 0<r['native_proof_rate']<1]),
                        ('native_none_proven', [r for r in matrix if r['native_proof_rate']==0]),
                        ('exclude_mspp16', [r for r in matrix if r['instance']!='mspp16'])]:
        cohorts.append(dict(cohort=label, instances=len(rows), native_mean_pdi=mean([r['native_pdi'] for r in rows]),
                            b0_mean_pdi=mean([r['b0_pdi'] for r in rows]),
                            b0_minus_native=mean([r['b0_minus_native'] for r in rows]),
                            b0_point_wins=sum(r['b0_minus_native']<0 for r in rows),
                            b0_measurement_resolved_wins=sum(r['measurement_delta_high']<0 for r in rows),
                            native_measurement_resolved_wins=sum(r['measurement_delta_low']>0 for r in rows)))
    writecsv('difficulty_cohorts.csv', cohorts)
    native = {(r['instance'],r['solver_seed']):r for r in tasks if r['method']=='native'}
    pairs=[]
    for b in tasks:
        if b['method']!='B0': continue
        n=native[b['instance'],b['solver_seed']]
        if not (b['proof'] and n['proof']): continue
        pairs.append(dict(instance=b['instance'], training_seed=b['training_seed'], solver_seed=b['solver_seed'],
                          b0_nodes=b['nodes'], native_nodes=n['nodes'], b0_optimize_seconds=b['optimize_wall_seconds'],
                          native_optimize_seconds=n['optimize_wall_seconds'], b0_pdi=b['pdi'],native_pdi=n['pdi'],
                          fewer_nodes=b['nodes']<n['nodes'],slower=b['optimize_wall_seconds']>n['optimize_wall_seconds']))
    writecsv('both_proven_pairs.csv',pairs)
    curves = []
    for r in tasks:
        # Retain real bound traces for all methods of the six largest absolute
        # instance-level gaps, plus mspp16. Illustration only, no model selection.
        selected={x['instance'] for x in sorted(matrix,key=lambda q:abs(q['b0_minus_native']),reverse=True)[:6]}|{'mspp16'}
        if r['instance'] not in selected: continue
        d=json.loads((ROOT/r['source_path']).read_text())
        for p in d.get('bounds_trace',[]):
            pb, db=p['primal'],p['dual']
            gap=0. if abs(pb-db)<=1e-9 else 100. if pb*db<=0 or max(abs(pb),abs(db))>=1e20 else min(100.,100*abs(pb-db)/max(abs(pb),abs(db)))
            curves.append(dict(instance=r['instance'],model=r['model'],solver_seed=r['solver_seed'],
                               time=p['time'],percent_bound_gap=gap,native_previous_percent_gap=p['native_previous_percent_gap'],
                               reason=p['reason'],proof_completed=d['proof_completed']))
    writecsv('gap_time_examples.csv',curves)
    updates=[]
    train_summary=[]
    coverage=[]
    reward_episode=[]
    logp_deviation=[]
    for model in sorted((RUN/'training').iterdir()):
        us=json.loads((model/'updates.json').read_text()); valid=[u for u in us if u['updated']]
        for u in us:
            updates.append(dict(model=model.name,updated=u['updated'],episodes_seen=u['episodes_seen'],
                                clip_fraction=u.get('clip_fraction'),entropy=u.get('entropy'),
                                initial_ratio_max_error=u.get('initial_ratio_max_error'),num_samples=u['num_samples'],num_updates=u.get('num_updates')))
        train_summary.append(dict(model=model.name, update_records=len(us),valid_updates=len(valid),
                                   clip_nonzero=sum(u['clip_fraction']!=0 for u in valid),
                                   initial_ratio_max_error=max(u['initial_ratio_max_error'] for u in valid),
                                   optimizer_steps=sum(u['num_updates'] for u in valid)))
        # JSON metadata covers all 3,000 episodes without loading all observations.
        metas=[]
        for p in sorted((model/'collection').rglob('e*.json')):
            d=json.loads(p.read_text()); metas.append(d)
        assert len(metas)==500
        for name in sorted({m['instance_name'] for m in metas}):
            ds=[m for m in metas if m['instance_name']==name]
            coverage.append(dict(model=model.name,instance=name,episodes=len(ds),
                                 transitions=sum(d['num_transitions'] for d in ds),
                                 mean_pdi=mean([d['pdi_horizon_midpoint'] for d in ds]),
                                 mean_total_reward=mean([d['total_reward'] for d in ds]),
                                 mean_proof_rate=mean([d['proof_completed'] for d in ds])))
        if not model.name.startswith('B0'): continue
        for p in sorted((model/'collection').rglob('*.pkl')):
            payload=pickle.loads(p.read_bytes())
            for ep in payload['episodes']:
                rows=ep['transitions']; n=len(rows)
                assert ep['task_terminal'] and not ep['collector_truncated'] and ep['bootstrap_value']==0
                assert (not n) or (rows[-1]['done'] and not any(r['done'] for r in rows[:-1]))
                rewards=[r['reward'] for r in rows]
                lengths=[len(r['cands_state']) for r in rows]
                deviations=[r['log_prob']+math.log(l) for r,l in zip(rows,lengths)]
                logp_deviation.extend(deviations)
                d=json.loads(p.with_suffix('.json').read_text())
                reward_episode.append(dict(model=model.name,episode_id=ep['episode_id'],instance=d['instance_name'],
                                            transitions=n,total_reward=sum(rewards),terminal_reward=rewards[-1] if n else None,
                                            nonterminal_reward_mean=mean(rewards[:-1]),
                                            step_clip_boundary_count=sum(abs(x)>=1-1e-8 for x in rewards[:-1]),
                                            min_candidates=min(lengths) if n else None,max_candidates=max(lengths) if n else None,
                                            mean_candidates=mean(lengths),pdi=d['pdi_horizon_midpoint'],
                                            proof=d['proof_completed'],status=d['status']))
        print('trajectories read',model.name,flush=True)
    writecsv('ppo_updates.csv',updates)
    writecsv('ppo_update_summary.csv',train_summary)
    writecsv('training_instance_coverage.csv',coverage)
    writecsv('b0_reward_episode_metrics.csv',reward_episode)
    reward_by_instance=[]
    for model in ['B0-s11','B0-s22']:
        rs=[r for r in coverage if r['model']==model]
        reward_by_instance.append(dict(model=model,instances=len(rs),
                                       pearson_mean_return_vs_pdi=corr([r['mean_total_reward'] for r in rs],[r['mean_pdi'] for r in rs])))
    summary=dict(hash_status_counts=dict(Counter(r['status'] for r in hashes)),
                 tasks=len(tasks),instances=len(matrix),cohorts=cohorts,updates=train_summary,
                 recorded_chosen_action_logp_minus_uniform=quantiles(logp_deviation),
                 recorded_chosen_action_abs_logp_minus_uniform=quantiles(np.abs(logp_deviation)),
                 b0_reward_episode_count=len(reward_episode),b0_reward_transitions=sum(r['transitions'] for r in reward_episode),
                 b0_step_clip_boundary_count=sum(r['step_clip_boundary_count'] for r in reward_episode),
                 reward_instance_correlations=reward_by_instance,
                 both_proven_pairs=len(pairs),both_proven_instances=len({r['instance'] for r in pairs}),
                 fewer_nodes_but_slower_pairs=sum(r['fewer_nodes'] and r['slower'] for r in pairs),
                 fewer_nodes_but_slower_instances=sorted({r['instance'] for r in pairs if r['fewer_nodes'] and r['slower']}),
                 zero_actor_b0_tasks=sum(r['method']=='B0' and r['branch_count']==0 for r in tasks),
                 zero_actor_b0_instances=sorted({r['instance'] for r in tasks if r['method']=='B0' and r['branch_count']==0}),
                 task_mean_actor_ms_quantiles=quantiles([1000*r['actor_seconds']/r['branch_count'] for r in tasks if r['method']=='B0' and r['branch_count']>0]),
                 elapsed_seconds=time.perf_counter()-start)
    save('existing_data_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
