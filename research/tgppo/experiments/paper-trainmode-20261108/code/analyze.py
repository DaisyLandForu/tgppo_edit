"""Full-matrix, per-instance paired reporting. Never ranks partial experiments."""
import csv
import math
from collections import defaultdict
from paper import RUN, ROOT, PACKAGE, VARIANTS, SEEDS, read, write, frozen_models, eval_instances


def mean(xs):
    return sum(xs)/len(xs)


def sgm(xs, shift):
    if any(x < 0 or not math.isfinite(x) for x in xs):
        raise ValueError('Invalid SGM input')
    if shift == 0 and any(x == 0 for x in xs):
        return 0.0
    return math.exp(mean([math.log(x+shift) for x in xs]))-shift


def summarize(c):
    models = frozen_models(c)
    rows = []
    missing = []
    for shard in range(c['evaluation_shards']):
        for name in eval_instances(c, shard):
            p = RUN/'evaluation'/('shard-'+str(shard))/name/'results.json'
            values = read(p) if p.exists() else []
            if len(values) != 35 or any(not x.get('valid') for x in values):
                missing.append(dict(instance=name, valid=sum(bool(x.get('valid')) for x in values)))
                continue
            for r in values:
                label = r['job_id'].split('-'+name+'-s')[0][len('eval-'):]
                rows.append(dict(r, label=label))
    docs = ROOT/'docs/methods/tgppo_paper_trainmode_20261108/results'
    if missing:
        write(docs/'progress.json', dict(complete=False, valid_complete_instance_records=len(rows),
                                       expected=2310, missing=missing, model_ranking=None))
        print('Incomplete matrix: {} instances missing. No overall ranking.'.format(len(missing)), flush=True)
        return
    assert len(rows) == 2310 and len({(r['label'], r['instance_name'], r['solver_seed']) for r in rows}) == 2310
    expected = {(label, n, s) for label in ['native']+[m['label'] for m in models] for n in c['test'] for s in range(5)}
    assert {(r['label'], r['instance_name'], r['solver_seed']) for r in rows} == expected
    by_label = defaultdict(list)
    for r in rows:
        by_label[r['label']].append(r)
    table = []
    for label, rs in sorted(by_label.items()):
        proofs = [r['proof_completed'] and r['scip_solve_time'] <= c['time_limit'] for r in rs]
        calls = sum(r.get('branch_count', 0) for r in rs)
        actor_time = sum(r.get('callback_seconds', {}).get('actor', 0.) for r in rs)
        table.append(dict(label=label, tasks=len(rs), mean_pdi=mean([r['pdi_horizon_midpoint'] for r in rs]),
             mean_pdi_lower=mean([r['pdi_horizon_lower'] for r in rs]),
             mean_pdi_upper=mean([r['pdi_horizon_upper'] for r in rs]),
             sgm_pdi_shift0=sgm([r['pdi_horizon_midpoint'] for r in rs], 0),
             sgm_nodes_shift100=sgm([r['nnodes'] for r in rs], 100),
             proof_rate=mean(proofs), mean_optimize_seconds=mean([r['optimize_wall_seconds'] for r in rs]),
             par2=mean([r['scip_solve_time'] if proved else 2*c['time_limit'] for r, proved in zip(rs, proofs)]),
             actor_mean_ms=1000*actor_time/calls if calls else None))
    effects = []
    indexed = {(r['label'], r['instance_name'], r['solver_seed']): r for r in rows}
    for seed in SEEDS:
        for a, b, effect in [('mlp', 'b0', 'structure'), ('mlp_bc', 'mlp', 'BC'), ('mlp_bc', 'b0', 'combined'),
                             ('b0', 'native', 'baseline_vs_native'), ('mlp', 'native', 'MLP_vs_native'), ('mlp_bc', 'native', 'MLP_BC_vs_native')]:
            la = a+'-s'+str(seed)
            lb = b if b == 'native' else b+'-s'+str(seed)
            for name in c['test']:
                aa = [indexed[la, name, ss] for ss in range(5)]
                bb = [indexed[lb, name, ss] for ss in range(5)]
                pa = sgm([r['pdi_horizon_midpoint'] for r in aa], 0)
                pb = sgm([r['pdi_horizon_midpoint'] for r in bb], 0)
                effects.append(dict(effect=effect, training_seed=seed, instance=name, method=la, reference=lb,
                    mean_pdi_difference=mean([r['pdi_horizon_midpoint'] for r in aa])-mean([r['pdi_horizon_midpoint'] for r in bb]),
                    pdi_difference_lower=mean([r['pdi_horizon_lower'] for r in aa])-mean([r['pdi_horizon_upper'] for r in bb]),
                    pdi_difference_upper=mean([r['pdi_horizon_upper'] for r in aa])-mean([r['pdi_horizon_lower'] for r in bb]),
                    seed_sgm_pdi_method=pa, seed_sgm_pdi_reference=pb,
                    seed_sgm_pdi_win=pa < pb,
                    seed_sgm_nodes_method=sgm([r['nnodes'] for r in aa], 100),
                    seed_sgm_nodes_reference=sgm([r['nnodes'] for r in bb], 100)))
    def csvwrite(name, values):
        docs.mkdir(parents=True, exist_ok=True)
        with (docs/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    csvwrite('summary.csv', table)
    csvwrite('paired.csv', effects)
    common_actor, no_actor, mixed = [], [], []
    for name in c['test']:
        participation = [indexed[m['label'], name, ss].get('branch_count', 0) > 0
                         for m in models for ss in range(5)]
        (common_actor if all(participation) else no_actor if not any(participation) else mixed).append(name)
    panels = dict(all66=c['test'], common_actor=common_actor, no_actor=no_actor, mixed=mixed)
    write(docs/'actor_participation.json', panels)
    effect_summary = []
    for effect in sorted({e['effect'] for e in effects}):
        for seed in SEEDS:
            for panel, names in panels.items():
                chosen = [e for e in effects if e['effect'] == effect and e['training_seed'] == seed and e['instance'] in names]
                if not chosen:
                    continue
                effect_summary.append(dict(effect=effect, training_seed=seed, panel=panel, instances=len(chosen),
                     mean_pdi_difference=mean([e['mean_pdi_difference'] for e in chosen]),
                     pdi_difference_lower=mean([e['pdi_difference_lower'] for e in chosen]),
                     pdi_difference_upper=mean([e['pdi_difference_upper'] for e in chosen]),
                     separated_wins=sum(e['pdi_difference_upper'] < 0 for e in chosen),
                     separated_losses=sum(e['pdi_difference_lower'] > 0 for e in chosen),
                     seed_sgm_pdi_wins=sum(e['seed_sgm_pdi_win'] for e in chosen)))
    csvwrite('independent_effects.csv', effect_summary)
    subset = read(PACKAGE/'docs/author_subset_audit.json')
    subsets = {'author_easy33': subset['finished_instances.xlsx']['instances'],
               'author_hard32': subset['timelimit_instances.xlsx']['instances'],
               'author_unlisted1': subset['missing_test_instances']}
    subgroup = []
    for label, rs in sorted(by_label.items()):
        for group, names in subsets.items():
            chosen = [r for r in rs if r['instance_name'] in names]
            subgroup.append(dict(label=label, subset=group, instances=len(names), tasks=len(chosen),
                 sgm_pdi_shift0=sgm([r['pdi_horizon_midpoint'] for r in chosen], 0),
                 sgm_nodes_shift100=sgm([r['nnodes'] for r in chosen], 100),
                 mean_pdi=mean([r['pdi_horizon_midpoint'] for r in chosen])))
    csvwrite('author_subsets.csv', subgroup)
    write(docs/'results.json', dict(complete=True, tasks=len(rows), models=models, summary=table,
          easy_hard='Author spreadsheets:33easy+32hard; opm2-z7-s2 absent; report separately without inventing33rd hard member',
          zero_pdi_sgm='shift0 geometric mean is0 if any observation0; arithmetic mean and bounds retained',
          no_test_model_selection=True, historical_test_exposure=True, measurement_bounds_are_not_CI=True,
          actor_participation_counts={k: len(v) for k, v in panels.items()}))
    lines = ['# 论文主协议对齐评测', '', '2310/2310有效；固定500episode、3600秒、66实例×5solver seeds。', '',
             '| 方法／训练种子 | 平均PDI | 证明率 | Actor ms |', '|---|---:|---:|---:|']
    for t in table:
        lines.append('| {} | {:.3f} | {:.2%} | {} |'.format(t['label'], t['mean_pdi'], t['proof_rate'],
                     '{:.3f}'.format(t['actor_mean_ms']) if t['actor_mean_ms'] is not None else '—'))
    lines += ['', '结构、BC和整体增量见paired.csv／independent_effects.csv。主表不删例；共同Actor、全无Actor、混合参与面板另列，不能将无Actor变化归因于模型。',
              'PDI测量界并非统计置信区间；solver seeds是实例内重复。',
              '作者发布的表格实际33 easy＋32 hard，缺opm2-z7-s2，与正文33/33不符。按固定作者名单另报子集，完整66主表不删例、不按本轮输赢重划。',
              '未重跑Optuna；保留正确性修复与历史测试暴露边界。']
    (docs/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(str(docs/'REPORT.md'), flush=True)
