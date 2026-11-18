"""No ranking until all330 rows form complete same-environment panels."""
import csv
import fcntl
import math
from collections import defaultdict

from common import DOCS, LABELS, PACKAGE, ROOT, RUN, SHARDS, plan, read, scan, selected, write


def mean(values):
    return sum(values) / len(values)


def sgm(values, shift=0):
    assert values and all(v >= 0 and math.isfinite(v) for v in values)
    if shift == 0 and any(v == 0 for v in values):
        return 0.
    return math.exp(mean([math.log(v+shift) for v in values]))-shift


def statistics(rows, label):
    proved = [r['proof_completed'] and r['scip_solve_time'] <= 3600 for r in rows]
    calls = sum(r.get('branch_count', 0) for r in rows)
    return dict(label=label, tasks=len(rows), mean_pdi=mean([r['pdi_horizon_midpoint'] for r in rows]),
        mean_pdi_lower=mean([r['pdi_horizon_lower'] for r in rows]),
        mean_pdi_upper=mean([r['pdi_horizon_upper'] for r in rows]),
        sgm_pdi_shift0=sgm([r['pdi_horizon_midpoint'] for r in rows]),
        sgm_nodes_shift100=sgm([r['nnodes'] for r in rows], 100),
        proof_rate=mean(proved), proofs=sum(proved),
        mean_optimize_seconds=mean([r['optimize_wall_seconds'] for r in rows]),
        par2=mean([r['scip_solve_time'] if p else 7200 for r, p in zip(rows, proved)]),
        branch_calls=calls, actor_mean_ms=1000*sum(r.get('callback_seconds', {}).get('actor', 0) for r in rows)/calls if calls else None,
        summed_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows))


def paired(rows):
    idx = {(r['label'], r['instance_name']): r for r in rows}
    # Only compare method increments within the same training seed.
    # There is no fixed200 B0-s11 or MLP-s22 in this frozen evaluation.
    effects = [('mlp_bc_s11', 'mlp_s11', 'BC_seed11'),
        ('mlp_bc_s22', 'b0_new_s22', 'combined_seed22')]
    effects += [(label, 'native', label+'_vs_relpscost') for label in LABELS if label != 'native']
    names = sorted({r['instance_name'] for r in rows})
    result = []
    for a, b, effect in effects:
        for name in names:
            aa, bb = idx[a, name], idx[b, name]
            result.append(dict(effect=effect, instance=name, method=a, reference=b,
                mean_pdi_difference=aa['pdi_horizon_midpoint']-bb['pdi_horizon_midpoint'],
                pdi_difference_lower=aa['pdi_horizon_lower']-bb['pdi_horizon_upper'],
                pdi_difference_upper=aa['pdi_horizon_upper']-bb['pdi_horizon_lower'],
                method_pdi=aa['pdi_horizon_midpoint'], reference_pdi=bb['pdi_horizon_midpoint'],
                method_proved=aa['proof_completed'] and aa['scip_solve_time'] <= 3600,
                reference_proved=bb['proof_completed'] and bb['scip_solve_time'] <= 3600,
                same_environment=aa['cohort_id'] == bb['cohort_id']))
    assert all(r['same_environment'] for r in result)
    return result


def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    DOCS.mkdir(parents=True, exist_ok=True)
    # Four finishing jobs can call this concurrently; publish a coherent snapshot.
    with (DOCS / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, missing, hardware, progress = [], [], {}, []
        for shard in SHARDS:
            panels, cohorts = scan(shard)
            rows.extend(r for values in panels.values() for r in values)
            hardware.update({cid: c['environment'] for cid, c in cohorts.items()})
            missing.extend(i['name'] for i in selected(shard) if i['name'] not in panels)
            progress.append(dict(shard=shard, complete_instances=len(panels), total_instances=len(selected(shard)),
                completed_paired_tasks=5*len(panels),
                best_cohort_valid_tasks=sum(max([sum(r['instance_name'] == i['name'] for r in c['rows'].values())
                    for c in cohorts.values()] or [0]) for i in selected(shard))))
        snapshot = dict(complete=not missing, requested_tasks=330, completed_paired_tasks=len(rows),
            missing_instances=missing, shards=progress, ranking=None if missing else 'summary.csv')
        write(DOCS / 'progress.json', snapshot)
        if missing:
            print('Complete paired panels: {}/330 tasks; {} instances pending. No partial ranking.'.format(len(rows), len(missing)), flush=True)
            return snapshot
        assert len(rows) == 330
        assert {(r['label'], r['instance_name'], r['solver_seed']) for r in rows} == {
            (label, i['name'], 0) for label in LABELS for i in plan()['instances']}
        table = [statistics([r for r in rows if r['label'] == label], label) for label in LABELS]
        effects = paired(rows)
        participation = defaultdict(list)
        for i in plan()['instances']:
            flags = [r['branch_count'] > 0 for r in rows if r['instance_name'] == i['name'] and r['label'] != 'native']
            participation['common_actor' if all(flags) else 'no_actor' if not any(flags) else 'mixed'].append(i['name'])
        panels = dict(all66=[i['name'] for i in plan()['instances']], **participation)
        effect_summary = []
        for effect in sorted({e['effect'] for e in effects}):
            for panel, names in panels.items():
                group = [e for e in effects if e['effect'] == effect and e['instance'] in names]
                if not group:
                    continue
                m = mean([e['method_pdi'] for e in group])
                b = mean([e['reference_pdi'] for e in group])
                effect_summary.append(dict(effect=effect, panel=panel, instances=len(group),
                    method_mean_pdi=m, reference_mean_pdi=b, relative_change_percent=100*(m/b-1) if b else None,
                    mean_pdi_difference=mean([e['mean_pdi_difference'] for e in group]),
                    pdi_difference_lower=mean([e['pdi_difference_lower'] for e in group]),
                    pdi_difference_upper=mean([e['pdi_difference_upper'] for e in group]),
                    separated_wins=sum(e['pdi_difference_upper'] < 0 for e in group),
                    separated_losses=sum(e['pdi_difference_lower'] > 0 for e in group),
                    midpoint_wins=sum(e['mean_pdi_difference'] < 0 for e in group),
                    proof_rate_difference=mean([int(e['method_proved'])-int(e['reference_proved']) for e in group])))
        csvwrite(DOCS / 'summary.csv', table)
        csvwrite(DOCS / 'paired.csv', effects)
        csvwrite(DOCS / 'independent_effects.csv', effect_summary)
        csvwrite(DOCS / 'raw_rows.csv', [dict(label=r['label'], instance=r['instance_name'], solver_seed=0,
            cohort_id=r['cohort_id'], pdi=r['pdi_horizon_midpoint'], pdi_lower=r['pdi_horizon_lower'],
            pdi_upper=r['pdi_horizon_upper'], proof_completed=r['proof_completed'],
            scip_solve_time=r['scip_solve_time'], optimize_wall_seconds=r['optimize_wall_seconds'],
            nnodes=r['nnodes'], branch_count=r.get('branch_count', 0), result_path=r['result_path'],
            result_sha256=r['result_sha256']) for r in rows])
        from common import BASE
        subset_audit = read(BASE / 'docs/author_subset_audit.json')
        paper_subsets = dict(author_easy33=subset_audit['finished_instances.xlsx']['instances'],
            author_hard32=subset_audit['timelimit_instances.xlsx']['instances'],
            author_unassigned=subset_audit['missing_test_instances'])
        paper_tables = [dict(subset=subset, **statistics(
            [r for r in rows if r['label'] == label and r['instance_name'] in names], label))
            for subset, names in paper_subsets.items() for label in LABELS]
        csvwrite(DOCS / 'paper_subset_metrics.csv', paper_tables)
        write(DOCS / 'results.json', dict(complete=True, tasks=330, summary=table,
            independent_effects=effect_summary, actor_participation=panels, environments=hardware,
            frozen_models=plan()['models'], training_seeds=[11, 22], solver_seeds=[0], episodes=200,
            unbalanced_training_seed_coverage=True, structure_increment_available=False,
            eligible_increment_comparisons=plan()['eligible_increment_comparisons'],
            paper_subset_metrics=paper_tables,
            historical_test_exposure=True, test_used_for_selection=False, measurement_bounds_are_not_CI=True))
        lines = ['# 固定200 episode／完整66实例：可用权重评测', '',
            '330/330项有效；四份已到200权重，solver seed0，求解上限3600秒。各权重与原生relpscost按实例同环境配对。', '',
            '| 方法 | 平均PDI ↓ | 证明率 ↑ | Actor ms ↓ |', '|---|---:|---:|---:|']
        for t in table:
            lines.append('| {} | {:.3f} | {:.2%} | {} |'.format(t['label'], t['mean_pdi'], t['proof_rate'],
                '{:.3f}'.format(t['actor_mean_ms']) if t['actor_mean_ms'] is not None else '—'))
        lines += ['', '同种子可报告BC（seed11）与结构＋BC整体（seed22）；各权重对relpscost另报。见independent_effects.csv和paired.csv。',
            '新B0-s11与MLP-s22尚未到200，本轮不能识别同种子的纯结构增量，也不按不齐全的种子直接汇总方法均值。',
            '作者固定easy33／hard32与未分组1例的SGM另见paper_subset_metrics.csv；不按本轮结果重划easy/hard。',
            '完整66为主表；共同调用Actor、均未调用Actor及混合面板另外报告，无Actor变化不能归因于网络。',
            '这是固定200的中途测试，solver仅seed0；不是最终500episode×5solver seeds。训练种子覆盖不齐，不能证明跨种子稳定收益；不使用测试结果挑权重。',
            'MLP类权重来自已记录的GPU→CPU续训分支；B0来自CUDA分支。训练设备差异保留，不宣称跨设备逐位等价。',
            '这些测试实例有历史暴露。PDI上下界是测量界，不是统计置信区间。不同实例可能来自不同CPU，比较始终在同一实例环境内配对。',
            '本轮评测完成后提交分析并停止，不自动扩训、调参或追加种子。']
        (DOCS / 'REPORT.md').write_text('\n'.join(lines)+'\n')
        print(str(DOCS / 'REPORT.md'), flush=True)
        return snapshot
