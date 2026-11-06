"""Read committed results only. No Torch, training, solver, or /tmp dependency."""
import csv
import hashlib
import statistics as st
import sys

from common import BUDGET, DOCS, EPISODES, NAMES, OUT, PACKAGE, SEEDS, read, sha, specs, write


def csv_save(name, rows):
    if not rows:
        return
    path = DOCS / name
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def panel(rows):
    calls = sum(r.get('branch_count', 0) for r in rows)
    timing = {k: sum(r.get('callback_seconds', {}).get(k, 0) for r in rows)
              for k in ('actor', 'features', 'callback', 'critic')}
    wall = sum(r['optimize_wall_seconds'] for r in rows)
    return dict(pdi=st.mean(r['pdi_horizon_midpoint'] for r in rows),
        lower=st.mean(r['pdi_horizon_lower'] for r in rows), upper=st.mean(r['pdi_horizon_upper'] for r in rows),
        proof_rate=st.mean(bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in rows),
        nodes=st.mean(r['nnodes'] for r in rows), wall=wall / len(rows), actor_calls=calls,
        actor_ms=1000 * timing['actor'] / calls if calls else None,
        feature_ms=1000 * timing['features'] / calls if calls else None,
        actor_fraction_of_wall=timing['actor'] / wall if wall else None,
        callback_fraction_of_wall=timing['callback'] / wall if wall else None,
        critic_seconds=timing['critic'])


def effect(before, after):
    delta = after['pdi'] - before['pdi']
    return dict(pdi_delta=delta, pdi_percent=100 * delta / before['pdi'] if before['pdi'] else None,
        measurement_delta_lower=after['lower'] - before['upper'],
        measurement_delta_upper=after['upper'] - before['lower'],
        proof_delta=after['proof_rate'] - before['proof_rate'], wall_delta=after['wall'] - before['wall'],
        direction='improved' if after['upper'] < before['lower'] else
            ('regressed' if after['lower'] > before['upper'] else 'measurement_overlap'))


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    engine = read(OUT / 'status.json') if (OUT / 'status.json').exists() else dict(status='not_started')
    job_state = read(OUT / 'job_state.json') if (OUT / 'job_state.json').exists() else {}
    manifest = read(OUT / 'manifest.json') if (OUT / 'manifest.json').exists() else None
    rows = read(OUT / 'evaluation_rows.json') if (OUT / 'evaluation_rows.json').exists() else []
    found = {r['job_id']: r for r in rows}
    expected = specs()
    missing, tasks, records, native = [], [], {}, []
    source_errors = []
    manifest_tasks = {t['job_id']: t for t in manifest['tasks']} if manifest else {}
    if len(found) != len(rows) or set(found) - {t['job_id'] for t in expected}:
        source_errors.append('duplicate_or_unexpected_result_ids')
    if manifest:
        if sha(PACKAGE / 'run.py') != manifest['runner_sha256'] or sha(DOCS / 'PLAN.md') != manifest['plan_sha256']:
            source_errors.append('runner_or_plan_changed')
        for model in manifest['models'].values():
            if sha(model['source']) != model['source_sha256'] or sha(model['frozen_actor']) != model['actor_sha256']:
                source_errors.append('source_or_frozen_actor_changed')
    for item in expected:
        row = found.get(item['job_id'])
        reason = None
        task = manifest_tasks.get(item['job_id'])
        if row is None:
            reason = 'missing'
        elif not row.get('valid'):
            reason = 'invalid:' + str(row.get('status', row.get('error')))
        elif not task:
            reason = 'manifest_task_missing'
        elif not row['worker_cpu_complete'] or row['cuda_initialized']:
            reason = 'CPU_contract_failed'
        elif row['instance_name'] != item['instance'] or row['solver_seed'] != item['solver_seed']:
            reason = 'instance_or_solver_seed_mismatch'
        else:
            seed = item['training_seed'] if item['training_seed'] is not None else 11
            canonical = item['canonical_sampling_job_id']
            sampling = int(hashlib.sha256((str(seed) + ':' + canonical).encode()).hexdigest()[:8], 16)
            if row['sampling_seed'] != sampling or row['sampling_seed'] != task['job']['sampling_seed']:
                reason = 'canonical_action_rng_mismatch'
            elif item['label'] != 'native' and (row['callback_seconds']['critic'] != 0 or row['fallback_action_count'] != 0):
                reason = 'critic_called_or_policy_fallback'
        if reason:
            missing.append(dict(job_id=item['job_id'], reason=reason))
            continue
        values = panel([row])
        tasks.append(dict(job_id=item['job_id'], episode=item['episode'], training_seed=item['training_seed'],
            instance=item['instance'], solver_seed=item['solver_seed'], condition=item['label'], **values))
        if item['label'] == 'native':
            native.append(row)
        else:
            records.setdefault((item['episode'], item['training_seed']), []).append(row)
    csv_save('tasks.csv', tasks)
    # Always write missing.csv, including its header for a complete attempt.
    with (DOCS / 'missing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['job_id', 'reason'])
        writer.writeheader()
        writer.writerows(missing)
    if missing or source_errors or engine['status'] != 'complete' or job_state.get('error'):
        result = dict(status='incomplete', valid_tasks=len(tasks), expected_tasks=70, missing=missing,
            source_errors=source_errors, engine_status=engine['status'], supervisor_error=job_state.get('error'),
            training_steps=0, gpu=False, extra_tasks_started=0)
        write(DOCS / 'analysis.json', result)
        (DOCS / 'REPORT.md').write_text('# 开发集学习曲线：本轮未完整\n\n'
            '有效{}/70项，engine状态{}。缺失/失败见missing.csv，原始数据和预算保留；'
            '不以部分结果挑选权重，不自动重试或延长预算。\n\n'
            'CPU Custom的持久日志：{}；最终累计墙时见{}。\n\n错误：{}\n'.format(
                len(tasks), engine['status'], OUT / 'runner.log', OUT / 'job_state.json',
                source_errors or job_state.get('error') or engine.get('error', 'see log')))
        print('Incomplete; report saved: ' + str(DOCS / 'REPORT.md'))
        return 1
    assert len(tasks) == 70 and len(native) == 10 and len(records) == 6
    active = [n for n in NAMES if all(r.get('branch_count', 0) > 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    inactive = [n for n in NAMES if all(r.get('branch_count', 0) == 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    summaries, per_instance, panel_effects, pairs, selections = [], [], [], [], []
    by_panel = {}
    for tag, names in [('all', NAMES), ('common_actor', active)]:
        if not names:
            continue
        nr = panel([r for r in native if r['instance_name'] in names])
        summaries.append(dict(episode='native', training_seed=None, panel=tag, **nr))
        for episode in EPISODES:
            for seed in list(SEEDS) + ['pooled']:
                rs = records[(episode, seed)] if seed != 'pooled' else [r for s in SEEDS for r in records[(episode, s)]]
                values = panel([r for r in rs if r['instance_name'] in names])
                by_panel[(episode, seed, tag)] = values
                summaries.append(dict(episode=episode, training_seed=seed, panel=tag, **values))
        for seed in list(SEEDS) + ['pooled']:
            for before, after in ((0, 20), (20, 100), (0, 100)):
                panel_effects.append(dict(before_episode=before, after_episode=after, training_seed=seed, panel=tag,
                    **effect(by_panel[(before, seed, tag)], by_panel[(after, seed, tag)])))
            chosen = min(EPISODES, key=lambda e: (by_panel[(e, seed, tag)]['pdi'], e))
            selections.append(dict(training_seed=seed, panel=tag, descriptive_min_pdi_episode=chosen,
                pdi=by_panel[(chosen, seed, tag)]['pdi'], proof_rate=by_panel[(chosen, seed, tag)]['proof_rate']))
    for name in NAMES:
        for seed in SEEDS:
            values = {}
            for episode in EPISODES:
                values[episode] = panel([r for r in records[(episode, seed)] if r['instance_name'] == name])
                per_instance.append(dict(instance=name, episode=episode, training_seed=seed,
                    common_actor=name in active, no_actor=name in inactive, **values[episode]))
            for before, after in ((0, 20), (20, 100), (0, 100)):
                pairs.append(dict(instance=name, training_seed=seed, before_episode=before, after_episode=after,
                    common_actor=name in active, no_actor=name in inactive, **effect(values[before], values[after])))
        per_instance.append(dict(instance=name, episode='native', training_seed=None,
            common_actor=False, no_actor=name in inactive, **panel([r for r in native if r['instance_name'] == name])))
    for name, data in [('summary.csv', summaries), ('instances.csv', per_instance),
                       ('paired.csv', pairs), ('panel_effects.csv', panel_effects), ('selection.csv', selections)]:
        csv_save(name, data)
    result = dict(status='complete', valid_tasks=70, model_tasks=60, native_tasks=10,
        common_actor_instances=active, no_actor_instances=inactive, selections=selections,
        training_steps=0, gpu=False, budget_seconds=BUDGET,
        wall_seconds_at_analysis=job_state.get('charged_wall_seconds'),
        final_cost_record=str(OUT / 'job_state.json'),
        source_sha256={str(p): sha(p) for p in (OUT / 'manifest.json', OUT / 'status.json',
            OUT / 'evaluation_rows.json', PACKAGE / 'release.json', PACKAGE / 'analyze.py', DOCS / 'PLAN.md')})
    write(DOCS / 'analysis.json', result)
    lines = ['# MLP＋BC：开发集0／20／100 episode学习曲线', '',
        '70/70有效完成：60模型＋10同机新原生。两个训练种子11/22、solver0/1，'
        '5个既有开发实例。没有新增训练/GPU或访问66测试实例。', '',
        'CPU：{}；主机{}。截至汇总累计{:.2f}/2400秒，包含启动/导入/校验/求解/保存；'
        '最终记账见job_state.json。'.format(manifest['stack']['cpu_model'], manifest['host'], job_state.get('charged_wall_seconds', 0)), '',
        'BC第0个PPO episode是实际训练采用的BC第100步初始化；原BC预算1000步，'
        '不是重新训练或改选BC权重。PPO20/100分别逐张量核对原checkpoint与Actor导出；'
        '部署不构建Critic/优化器。', '',
        '| PPO episode | PDI s11↓ | PDI s22↓ | 合并PDI↓ | 证明率↑ | optimize秒↓ | Actor ms/调用↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    nr = panel(native)
    lines.append('| relpscost | — | — | {:.3f} | {:.1%} | {:.3f} | — |'.format(nr['pdi'], nr['proof_rate'], nr['wall']))
    for episode in EPISODES:
        a, b, p = (by_panel[(episode, seed, 'all')] for seed in (11, 22, 'pooled'))
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.1%} | {:.3f} | {} |'.format(
            episode, a['pdi'], b['pdi'], p['pdi'], p['proof_rate'], p['wall'],
            '{:.3f}'.format(p['actor_ms']) if p['actor_ms'] is not None else '—'))
    lines += ['', 'PDI是固定120秒百分比gap积分。每个实例等权，solver重复先在实例内平均；'
        '证明完成还要求SCIP计时≤120秒。原生没有训练种子。', '',
        '共同Actor实例：{}。无Actor实例：{}。后者的变化不归因于神经决策。'.format(
            '、'.join(active) or '无', '、'.join(inactive) or '无'), '',
        '| 比较 | 训练种子 | 全5实例PDI变化 | 共同Actor PDI变化 | 全体证明率变化 |',
        '|---|---:|---:|---:|---:|']
    for before, after in ((0, 20), (20, 100), (0, 100)):
        for seed in SEEDS:
            all_effect = effect(by_panel[(before, seed, 'all')], by_panel[(after, seed, 'all')])
            ap = effect(by_panel[(before, seed, 'common_actor')], by_panel[(after, seed, 'common_actor')]) if active else None
            lines.append('| {}→{} | {} | {} | {} | {:+.1f}pp |'.format(before, after, seed,
                '{:+.2f}%'.format(all_effect['pdi_percent']) if all_effect['pdi_percent'] is not None else '—',
                '{:+.2f}%'.format(ap['pdi_percent']) if ap and ap['pdi_percent'] is not None else '—',
                100 * all_effect['proof_delta']))
    lines += ['', '各实例/种子的三段变化与积分测量上下界在paired.csv；部署分解在summary.csv。'
        '上下界是积分截尾测量范围，不是统计置信区间。', '',
        '开发集均值最低点（严格并列选早）为：{}。这只是同开发集的描述性选模，'
        '同时检查证明率；未覆盖原权重或导出新的“最佳模型”。'.format('；'.join(
            '{}种子{}→episode{}'.format(s['panel'], s['training_seed'], s['descriptive_min_pdi_episode']) for s in selections)), '',
        'E1 已确认事实：以上仅来自本轮70项同机评估。E2 源码机制：网络/廉价特征/SCIP/PDI/采样语义沿用冻结版，'
        'BC包装张量与原训练初始化等值，源文件SHA前后校验。E3 待验证：更长PPO能否改善、奖励目标/信用分配是否需改，'
        '本轮均不能直接确认。', '',
        '这是5开发实例的预算诊断，不是独立测试、收敛证明或统计显著性检验。'
        'PPO100若相对BC/20退步，不自动拉长300/500；若两种子/共同Actor实例表现一致，'
        '再提交固定下一阶段训练预算与选模规则供验收。当前阶段结束后停止。']
    (DOCS / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Complete: 70/70. Report saved: ' + str(DOCS / 'REPORT.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
