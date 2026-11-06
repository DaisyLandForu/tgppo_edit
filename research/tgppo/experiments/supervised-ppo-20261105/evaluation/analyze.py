"""CPU result aggregation and prespecified supervision gate; never runs a solver."""
import csv
import hashlib
import statistics as st
import sys
from common import BASE, DOCS, NAMES, OUT, PACKAGE, ROOT, SEEDS, TOP, read, sha, specs, write


def csv_save(name, rows):
    if not rows:
        return
    with (DOCS / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        callback_ms=1000 * timing['callback'] / calls if calls else None,
        actor_fraction_of_wall=timing['actor'] / wall if wall else None, critic_seconds=timing['critic'])


def contrast(a, b):
    return dict(pdi_delta=a['pdi'] - b['pdi'],
        pdi_percent=100 * (a['pdi'] / b['pdi'] - 1) if b['pdi'] else None,
        measurement_delta_lower=a['lower'] - b['upper'], measurement_delta_upper=a['upper'] - b['lower'],
        proof_delta=a['proof_rate'] - b['proof_rate'], wall_delta=a['wall'] - b['wall'],
        direction='improved' if a['upper'] < b['lower'] else ('regressed' if a['lower'] > b['upper'] else 'measurement_overlap'))


def evaluate_gate(records, active):
    selections = []
    chosen = {}
    for variant in ('control', 'expert'):
        for seed in SEEDS:
            options = {0: records[('bc', 0, seed)], 20: records[(variant, 20, seed)], 100: records[(variant, 100, seed)]}
            episode = min(options, key=lambda e: (panel(options[e])['pdi'], e))
            chosen[(variant, seed)] = options[episode]
            selections.append(dict(variant=variant, training_seed=seed, selected_episode=episode, **panel(options[episode])))
    differences, wins, reasons = [], [], []
    if len(active) < 2:
        reasons.append('fewer_than_two_common_actor_instances')
    for seed in SEEDS:
        for reference in ('control', 'bc'):
            for tag, names in [('all', NAMES), ('common_actor', active)]:
                if not names:
                    continue
                a = panel([r for r in chosen[('expert', seed)] if r['instance_name'] in names])
                ref = chosen[('control', seed)] if reference == 'control' else records[('bc', 0, seed)]
                b = panel([r for r in ref if r['instance_name'] in names])
                diff = contrast(a, b)
                differences.append(dict(variant='expert_selected', reference=reference, training_seed=seed, panel=tag, **diff))
                if a['upper'] >= b['lower']:
                    reasons.append('{}:{}:{}:PDI'.format(seed, reference, tag))
                if a['proof_rate'] < b['proof_rate']:
                    reasons.append('{}:{}:{}:proof'.format(seed, reference, tag))
    for name in active:
        passed = True
        for seed in SEEDS:
            a = panel([r for r in chosen[('expert', seed)] if r['instance_name'] == name])
            for ref in (chosen[('control', seed)], records[('bc', 0, seed)]):
                b = panel([r for r in ref if r['instance_name'] == name])
                passed &= a['upper'] < b['lower']
        if passed:
            wins.append(name)
    if len(wins) < 2:
        reasons.append('fewer_than_two_cross_seed_joint_instance_wins')
    return dict(passed=not reasons, failed_conditions=reasons, common_instance_wins=wins), selections, differences, chosen


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    engine = read(OUT / 'status.json') if (OUT / 'status.json').exists() else dict(status='not_started')
    supervisor = read(OUT / 'job_state.json') if (OUT / 'job_state.json').exists() else {}
    manifest = read(OUT / 'manifest.json') if (OUT / 'manifest.json').exists() else None
    rows = read(OUT / 'evaluation_rows.json') if (OUT / 'evaluation_rows.json').exists() else []
    found = {r['job_id']: r for r in rows}
    tasks = {t['job_id']: t for t in manifest['tasks']} if manifest else {}
    errors = []
    expected = specs()
    if len(rows) != len(found) or set(found) - {t['job_id'] for t in expected}:
        errors.append('duplicate_or_unexpected_rows')
    if manifest:
        if sha(PACKAGE / 'run.py') != manifest['runner_sha256'] or sha(TOP / 'docs/PROTOCOL.md') != manifest['protocol_sha256']:
            errors.append('runner_or_protocol_changed')
        for model in manifest['models'].values():
            if sha(model['source']) != model['source_sha256'] or sha(model['checkpoint']) != model['checkpoint_sha256']:
                errors.append('source_weight_changed')
            if sha(model['frozen_actor']) != model['actor_sha256']:
                errors.append('frozen_weight_changed')
    missing, flat, native, records = [], [], [], {}
    for item in expected:
        r, t = found.get(item['job_id']), tasks.get(item['job_id'])
        reason = None
        if r is None:
            reason = 'missing'
        elif not r.get('valid'):
            reason = 'invalid:' + str(r.get('status', r.get('error')))
        elif t is None or not r['worker_cpu_complete'] or r['cuda_initialized']:
            reason = 'manifest_or_CPU_contract'
        elif r['instance_name'] != item['instance'] or r['solver_seed'] != item['solver_seed']:
            reason = 'instance_or_solver_seed'
        else:
            seed = item['training_seed'] if item['training_seed'] is not None else 11
            sampling = int(hashlib.sha256((str(seed) + ':' + item['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
            if r['sampling_seed'] != sampling or r['sampling_seed'] != t['job']['sampling_seed']:
                reason = 'canonical_action_rng'
            elif item['variant'] != 'native' and (r['callback_seconds']['critic'] or r['fallback_action_count']):
                reason = 'critic_or_fallback'
        if reason:
            missing.append(dict(job_id=item['job_id'], reason=reason))
            continue
        flat.append(dict(variant=item['variant'], episode=item['episode'], training_seed=item['training_seed'],
            instance=item['instance'], solver_seed=item['solver_seed'], **panel([r])))
        if item['variant'] == 'native':
            native.append(r)
        else:
            records.setdefault((item['variant'], item['episode'], item['training_seed']), []).append(r)
    csv_save('tasks.csv', flat)
    with (DOCS / 'missing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['job_id', 'reason'])
        writer.writeheader()
        writer.writerows(missing)
    if missing or errors or engine['status'] != 'complete' or supervisor.get('error'):
        result = dict(status='incomplete', valid_tasks=len(flat), missing=missing, errors=errors,
            supervisor_error=supervisor.get('error'), next_stage_allowed=False)
        write(DOCS / 'analysis.json', result)
        (DOCS / 'REPORT.md').write_text('# 持续专家监督：评估未完整\n\n有效{}/110；状态{}。不以子集选模或扩大训练，'
            '失败见missing.csv／analysis.json及{}。保留成本与结果，无自动重试。\n'.format(len(flat), engine['status'], OUT / 'runner.log'))
        return 1
    assert len(flat) == 110 and len(native) == 10 and len(records) == 10
    active = [n for n in NAMES if all(r.get('branch_count', 0) > 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    inactive = [n for n in NAMES if all(r.get('branch_count', 0) == 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    gate, selections, differences, chosen = evaluate_gate(records, active)
    summaries, instances = [], []
    for (variant, episode, seed), rs in records.items():
        for tag, names in [('all', NAMES), ('common_actor', active)]:
            if names:
                summaries.append(dict(variant=variant, episode=episode, training_seed=seed, panel=tag,
                    **panel([r for r in rs if r['instance_name'] in names])))
        for name in NAMES:
            instances.append(dict(variant=variant, episode=episode, training_seed=seed,
                instance=name, common_actor=name in active, no_actor=name in inactive,
                **panel([r for r in rs if r['instance_name'] == name])))
    fixed = []
    for seed in SEEDS:
        for episode in (20, 100):
            for tag, names in [('all', NAMES), ('common_actor', active)]:
                if names:
                    a = panel([r for r in records[('expert', episode, seed)] if r['instance_name'] in names])
                    b = panel([r for r in records[('control', episode, seed)] if r['instance_name'] in names])
                    fixed.append(dict(episode=episode, training_seed=seed, panel=tag, **contrast(a, b)))
    for name, data in [('summary.csv', summaries), ('instances.csv', instances), ('selection.csv', selections),
                       ('selected_effects.csv', differences), ('fixed_budget_effects.csv', fixed)]:
        csv_save(name, data)
    training = []
    for v in ('control', 'expert'):
        for seed in SEEDS:
            source = BASE / v / 'training' / ('B0-s' + str(seed))
            updates = read(source / 'updates.json')
            actual = [u for u in updates if u['updated']]
            steps = sum(u['num_updates'] for u in actual)
            expert_batches = sum(u['expert_batches'] for u in actual)
            assert expert_batches == (steps if v == 'expert' else 0)
            training.append(dict(variant=v, training_seed=seed, updates=len(actual), optimizer_steps=steps,
                expert_batches=expert_batches, expert_samples=sum(u['expert_samples'] for u in actual),
                learning_seconds=sum(u['learning_wall_seconds'] for u in updates),
                charged_job_seconds=read(BASE / 'jobs' / (v + '-s' + str(seed) + '.json'))['charged_seconds']))
    csv_save('training.csv', training)
    hardware = {s: (p['control']['cpu'] == p['expert']['cpu'] and p['control']['gpu'] == p['expert']['gpu'])
        for s, p in manifest['pairings'].items()}
    decision = dict(status='complete', valid_tasks=110, gate=gate, next_stage_allowed=gate['passed'],
        next_stage_started=False, selected=selections, common_actor_instances=active, no_actor_instances=inactive,
        paired_training_cpu_gpu_models_equal=hardware, training_hardware_limit=not all(hardware.values()),
        evaluation_cpu=manifest['stack']['cpu_model'], training_steps_in_evaluation=0, gpu_in_evaluation=False,
        source_sha256={str(p): sha(p) for p in (OUT / 'manifest.json', OUT / 'evaluation_rows.json',
            OUT / 'status.json', TOP / 'docs/release.json', TOP / 'docs/PROTOCOL.md', PACKAGE / 'analyze.py')})
    write(DOCS / 'analysis.json', decision)
    lines = ['# 持续专家监督：完整开发结果', '', '110/110有效：100模型＋10同机新relpscost。'
        '固定轻量MLP与既有BC初始化；两个训练种子11/22，5开发实例×solver0/1，未访问66测试。', '',
        '方法增量：Actor损失增加0.01×实例均衡训练专家CE/log(K)，每PPO逻辑minibatch16专家状态；'
        '原奖励/Critic/PPO优化步骤及部署结构保持。BC-only不重训。', '',
        '| 方法 | 合并PDI↓ | 证明率↑ | optimize秒↓ | Actor ms/调用↓ |', '|---|---:|---:|---:|---:|']
    rows_by_method = {'relpscost': native,
        'BC-only': records[('bc', 0, 11)] + records[('bc', 0, 22)]}
    for v in ('control', 'expert'):
        rows_by_method[v + '-PPO100'] = records[(v, 100, 11)] + records[(v, 100, 22)]
        rows_by_method[v + '-dev-selected'] = chosen[(v, 11)] + chosen[(v, 22)]
    for label, rs in rows_by_method.items():
        p = panel(rs)
        lines.append('| {} | {:.3f} | {:.1%} | {:.3f} | {} |'.format(label, p['pdi'], p['proof_rate'], p['wall'],
            '{:.3f}'.format(p['actor_ms']) if p['actor_ms'] is not None else '—'))
    lines += ['', '选择：' + '；'.join('{} s{}→{}'.format(s['variant'], s['training_seed'], s['selected_episode']) for s in selections), '',
        '预写筛选结果：**{}**。失败条件{}；跨两种子同时优于对照及BC的共同Actor实例{}。'.format(
            '通过，值得提交后续预算' if gate['passed'] else '本次持续监督增量尚未过筛选',
            gate['failed_conditions'], gate['common_instance_wins']), '',
        '共同Actor实例{}；无Actor实例{}。全体与共同Actor面板、逐种子、等episode比较见CSV。'
        '无Actor项计时变化不归因于神经决策，积分测量上下界不是统计置信区间。'.format(active, inactive), '',
        '训练配对CPU/GPU型号一致性{}；不同硬件时保留训练条件混杂边界，不能把同机评估解释为训练也完全同机。'.format(hardware), '',
        'E1：本轮测量只回答持续监督的独立增量。E2：实际配对初始化与CPU/CUDA流、源SHA和监督次数被检查；'
        'Actor-only部署不加载专家数据或Critic。E3：收益原因、泛化和更长训练效果仍需验证。', '',
        '当前旧轻量结构＋BC的已有结果保持；本次监督增量未过筛选不等于原整体方法失败。'
        '筛选通过也不自动扩训或测试，需交付验收；没有自动300/500、CL、改奖励或树信用分配。', '',
        '累计评估墙时及预算见{}；节点/tmp消失后仍可从共享evaluation_rows.json重新汇总。'.format(OUT / 'job_state.json')]
    (DOCS / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Complete110 tasks; supervision gate=' + str(gate['passed']) + '; report=' + str(DOCS / 'REPORT.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
