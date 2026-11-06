"""Shard/matrix analysis of committed CPU records; no Torch or solver imports."""
import argparse
import csv
import fcntl
import hashlib
import math
import statistics as st
import sys
from common import (BUDGET, DOCS, PACKAGE, PLAN, RUN, SEEDS, SHARDS, TOTAL_TASKS, VARIANTS,
    output, read, resources, selected, sha, specs, task_count, task_resources, write)


def csv_save(folder, name, rows):
    if not rows:
        return
    with (folder / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel(rows):
    calls = sum(r.get('branch_count', 0) for r in rows)
    timing = {k: sum(r.get('callback_seconds', {}).get(k, 0) for r in rows)
        for k in ('actor', 'features', 'callback', 'critic')}
    wall = sum(r['optimize_wall_seconds'] for r in rows)
    proof = [bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in rows]
    return dict(pdi=st.mean(r['pdi_horizon_midpoint'] for r in rows),
        lower=st.mean(r['pdi_horizon_lower'] for r in rows), upper=st.mean(r['pdi_horizon_upper'] for r in rows),
        proof_rate=st.mean(proof), par2=st.mean(r['scip_solve_time'] if done else 240 for r, done in zip(rows, proof)),
        nodes=st.mean(r['nnodes'] for r in rows), wall=wall / len(rows), actor_calls=calls,
        actor_ms=1000 * timing['actor'] / calls if calls else None,
        feature_ms=1000 * timing['features'] / calls if calls else None,
        callback_ms=1000 * timing['callback'] / calls if calls else None,
        actor_fraction_of_wall=timing['actor'] / wall if wall else None,
        callback_fraction_of_wall=timing['callback'] / wall if wall else None,
        critic_seconds=timing['critic'])


def difference(candidate, reference):
    return dict(pdi_delta=candidate['pdi'] - reference['pdi'],
        pdi_percent=100 * (candidate['pdi'] / reference['pdi'] - 1) if reference['pdi'] else None,
        measurement_delta_lower=candidate['lower'] - reference['upper'],
        measurement_delta_upper=candidate['upper'] - reference['lower'],
        proof_delta=candidate['proof_rate'] - reference['proof_rate'], wall_delta=candidate['wall'] - reference['wall'],
        wall_percent=100 * (candidate['wall'] / reference['wall'] - 1) if reference['wall'] else None,
        direction='improved' if candidate['upper'] < reference['lower'] else
            ('regressed' if candidate['lower'] > reference['upper'] else 'measurement_overlap'))


def load_shard(shard):
    folder = output(shard)
    engine = read(folder / 'status.json') if (folder / 'status.json').exists() else dict(status='not_started')
    job = read(folder / 'job_state.json') if (folder / 'job_state.json').exists() else dict(status='not_started')
    manifest = read(folder / 'manifest.json') if (folder / 'manifest.json').exists() else None
    rows = read(folder / 'evaluation_rows.json') if (folder / 'evaluation_rows.json').exists() else []
    found = {r['job_id']: r for r in rows}
    tasks = {t['job_id']: t for t in manifest['tasks']} if manifest else {}
    expected = specs(shard)
    policy = resources(shard)
    errors = []
    if job.get('charged_wall_seconds', 0) > BUDGET:
        errors.append('per_shard_budget_exceeded')
    if len(rows) != len(found) or set(found) - {t['job_id'] for t in expected}:
        errors.append('duplicate_or_unexpected_rows')
    if manifest:
        for key, path in [('runner_sha256', PACKAGE / 'run.py'), ('protocol_sha256', PACKAGE / 'docs/PROTOCOL.md'),
                          ('selection_sha256', PACKAGE / 'selection.json')]:
            if manifest[key] != sha(path):
                errors.append(key + '_changed')
        proposed = {m['label']: m for m in PLAN['frozen_models']}
        if set(manifest['models']) != set(proposed):
            errors.append('model_inventory_changed')
        for label, model in manifest['models'].items():
            if label not in proposed or model['source'] != proposed[label]['actor'] or model['actor_sha256'] != proposed[label]['actor_sha256']:
                errors.append('frozen_model_identity_changed')
        if manifest['selected'] != selected(shard):
            errors.append('instance_selection_changed')
        if manifest.get('resource_policy') != policy or manifest.get('workers') != policy['workers']:
            errors.append('resource_policy_changed')
    missing, accepted = [], []
    for item in expected:
        row, task = found.get(item['job_id']), tasks.get(item['job_id'])
        row_policy = task_resources(item['instance'])
        reason = None
        if row is None:
            reason = 'missing'
        elif not row.get('valid'):
            reason = 'invalid:' + str(row.get('status', row.get('error')))
        elif task is None or any(task[k] != item[k] for k in item):
            reason = 'task_identity'
        elif not row['worker_cpu_complete'] or row['cuda_initialized']:
            reason = 'CPU_contract'
        elif row['instance_name'] != item['instance'] or row['solver_seed'] != item['solver_seed']:
            reason = 'instance_or_solver_seed'
        elif row['mode'] != ('native' if item['variant'] == 'native' else 'eval'):
            reason = 'mode'
        elif row['time_limit_seconds'] != 120 or row['pdi_horizon_seconds'] != 120:
            reason = 'horizon'
        elif (row.get('address_space_limit_bytes') != row_policy['address_space_limit_bytes']
                or row.get('cpu_affinity') != manifest.get('cpu_affinity', [])[:row_policy['workers']]
                or len(row.get('cpu_affinity', [])) != row_policy['workers']
                or task['job'].get('address_space_limit_bytes') != row_policy['address_space_limit_bytes']):
            reason = 'resource_exception_or_affinity'
        else:
            seed = item['training_seed'] if item['training_seed'] is not None else 11
            sampling = int(hashlib.sha256((str(seed) + ':' + item['canonical_sampling_job_id']).encode()).hexdigest()[:8], 16)
            if row['sampling_seed'] != task['job']['sampling_seed'] or row['sampling_seed'] != sampling:
                reason = 'canonical_action_rng'
            elif item['variant'] != 'native' and (row['callback_seconds']['critic'] or row['fallback_action_count']):
                reason = 'critic_called_or_fallback'
            elif not all(math.isfinite(row[k]) for k in ('pdi_horizon_midpoint', 'pdi_horizon_lower', 'pdi_horizon_upper')):
                reason = 'nonfinite_PDI'
        if reason:
            missing.append(dict(shard=shard, job_id=item['job_id'], reason=reason))
        else:
            accepted.append((item, row))
    complete = (not missing and not errors and len(accepted) == task_count(shard) and engine['status'] == 'complete'
        and engine.get('weights_unchanged') and job.get('engine_complete') and job.get('final_input_verification')
        and not job.get('error'))
    return dict(shard=shard, complete=bool(complete), accepted=accepted, missing=missing, errors=errors,
        manifest=manifest, engine=engine, job=job)


def render(folder, loaded, whole_matrix=False):
    folder.mkdir(parents=True, exist_ok=True)
    expected_count = TOTAL_TASKS if whole_matrix else sum(task_count(r['shard']) for r in loaded)
    accepted = [pair for result in loaded for pair in result['accepted']]
    flat = [dict(shard=t['shard'], variant=t['variant'], training_seed=t['training_seed'], instance=t['instance'],
        solver_seed=t['solver_seed'], job_id=t['job_id'], **panel([r])) for t, r in accepted]
    csv_save(folder, 'tasks.csv', flat)
    missing = [m for result in loaded for m in result['missing']]
    with (folder / 'missing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['shard', 'job_id', 'reason'])
        writer.writeheader()
        writer.writerows(missing)
    complete = len(accepted) == expected_count and all(result['complete'] for result in loaded)
    # The final global report additionally requires terminal supervisor commits.
    if whole_matrix:
        complete &= all(result['job'].get('status') == 'complete' for result in loaded)
    cohort = []
    for result in loaded:
        m = result['manifest']
        cohort.append(dict(shard=result['shard'], status=result['job']['status'], valid_tasks=len(result['accepted']),
            cpu=m['stack']['cpu_model'] if m else None, host=m['host'] if m else None,
            instances=';'.join(x['instance'] for x in selected(result['shard'])),
            workers=resources(result['shard'])['workers'],
            ordinary_address_space_gib=8, serial_mspp16=resources(result['shard'])['has_serial_mspp16'],
            watchdog_seconds=resources(result['shard'])['watchdog_seconds'],
            charged_wall_seconds=result['job'].get('charged_wall_seconds'),
            worker_wall_seconds=result['engine'].get('summed_worker_wall_seconds')))
    csv_save(folder, 'cohorts.csv', cohort)
    if not complete:
        # Never leave an older complete ranking next to a newer incomplete status.
        for name in ('summary.csv', 'instances.csv', 'effects.csv', 'paired.csv', 'mean_decomposition.csv'):
            stale = folder / name
            if stale.exists():
                stale.unlink()
        value = dict(status='incomplete', expected_tasks=expected_count, valid_tasks=len(accepted),
            complete_shards=[x['shard'] for x in loaded if x['complete']], cohorts=cohort,
            errors={str(x['shard']): x['errors'] for x in loaded}, missing_tasks=len(missing),
            new_training=False, no_partial_model_ranking=True)
        write(folder / 'analysis.json', value)
        (folder / 'REPORT.md').write_text('# 冻结66实例测试：当前范围尚未完整\n\n有效{}/{}项。分片状态见cohorts.csv，缺项/失败见missing.csv，'
            '未按完整矩阵做方法排名，也未替换实例或自动重试。\n\n本阶段要求原测试目录全部66实例、4分片1188项齐全；未齐不称全66完成。历史数据暴露边界仍保留。'
            '全部作业完成后自动更新，或以summarize只读重汇总。\n'.format(len(accepted), expected_count))
        return False
    names = [x['instance'] for result in loaded for x in selected(result['shard'])]
    records, native = {}, []
    for task, row in accepted:
        if task['variant'] == 'native':
            native.append(row)
        else:
            records.setdefault((task['variant'], task['training_seed']), []).append(row)
    active = [n for n in names if all(r.get('branch_count', 0) > 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    inactive = [n for n in names if all(r.get('branch_count', 0) == 0 for rs in records.values() for r in rs if r['instance_name'] == n)]
    mixed = sorted(set(names) - set(active) - set(inactive))
    summaries, instances, effects, pairs, decompositions = [], [], [], [], []
    by_panel = {}
    for tag, subset in [('all', names), ('common_actor', active)]:
        if not subset:
            continue
        nr = panel([r for r in native if r['instance_name'] in subset])
        by_panel[('native', 'pooled', tag)] = nr
        summaries.append(dict(variant='native', training_seed=None, panel=tag, **nr))
        for variant in VARIANTS:
            for seed in list(SEEDS) + ['pooled']:
                rs = records[(variant, seed)] if seed != 'pooled' else [r for s in SEEDS for r in records[(variant, s)]]
                p = panel([r for r in rs if r['instance_name'] in subset])
                by_panel[(variant, seed, tag)] = p
                summaries.append(dict(variant=variant, training_seed=seed, panel=tag, **p))
                effects.append(dict(effect='versus_native', candidate=variant, reference='native', training_seed=seed,
                    panel=tag, **difference(p, nr)))
        for label, a, b in [('structure_noBC', 'mlp', 'b0'), ('structure_BC', 'mlp_bc', 'b0_bc'),
                             ('BC_within_MLP', 'mlp_bc', 'mlp'), ('combined', 'mlp_bc', 'b0')]:
            for seed in list(SEEDS) + ['pooled']:
                effects.append(dict(effect=label, candidate=a, reference=b, training_seed=seed, panel=tag,
                    **difference(by_panel[(a, seed, tag)], by_panel[(b, seed, tag)])))
    for name in names:
        for (variant, seed), rs in records.items():
            instances.append(dict(instance=name, variant=variant, training_seed=seed, common_actor=name in active,
                no_actor=name in inactive, **panel([r for r in rs if r['instance_name'] == name])))
        instances.append(dict(instance=name, variant='native', training_seed=None, common_actor=False,
            no_actor=name in inactive, **panel([r for r in native if r['instance_name'] == name])))
        for seed in SEEDS:
            for label, a, b in [('structure_noBC', 'mlp', 'b0'), ('structure_BC', 'mlp_bc', 'b0_bc'),
                                 ('BC_within_MLP', 'mlp_bc', 'mlp'), ('combined', 'mlp_bc', 'b0')]:
                pa = panel([r for r in records[(a, seed)] if r['instance_name'] == name])
                pb = panel([r for r in records[(b, seed)] if r['instance_name'] == name])
                delta = difference(pa, pb)
                pairs.append(dict(effect=label, instance=name, training_seed=seed, candidate=a, reference=b,
                    common_actor=name in active, no_actor=name in inactive, **delta))
                decompositions.append(dict(effect=label, instance=name, training_seed=seed,
                    contribution_to_full_mean_delta=delta['pdi_delta'] / len(names),
                    common_actor=name in active, no_actor=name in inactive))
    for name, rows in [('summary.csv', summaries), ('instances.csv', instances), ('effects.csv', effects),
                       ('paired.csv', pairs), ('mean_decomposition.csv', decompositions)]:
        csv_save(folder, name, rows)
    sources = [PACKAGE / 'release.json', PACKAGE / 'selection.json', PACKAGE / 'docs/PROTOCOL.md', PACKAGE / 'analyze.py']
    for result in loaded:
        sources += [output(result['shard']) / name for name in ('manifest.json', 'evaluation_rows.json', 'status.json')]
    write(folder / 'analysis.json', dict(status='complete', valid_tasks=len(accepted), model_tasks=len(accepted) - len(native),
        native_tasks=len(native), instances=names, common_actor_instances=active, no_actor_instances=inactive,
        mixed_actor_instances=mixed, cohorts=cohort, training_steps=0, gpu=False, no_model_selection=True,
        data_scope='complete66 original test instances' if whole_matrix else 'one shard of complete66 test evaluation',
        historical_test_exposure=True, full66_complete=bool(whole_matrix),
        source_sha256={str(p): sha(p) for p in sources}))
    lines = ['# 冻结100episode方法：测试评估结果（本报告{}实例）'.format(len(names)), '',
        '{}/{}有效完成，{}模型＋{}同机新原生，{}实例、训练种子11/22、solver0/1。'
        '所有B0/MLP与无/有BC条件均为旧100episode权重；不包含正在训练的持续监督分支。'.format(
            len(accepted), expected_count, len(accepted) - len(native), len(native), len(names)), '',
        '本阶段清单覆盖原66测试实例，含mspp16；只有全局4分片1188项有效才称全66完成。'
        '历史上已查看过测试数据，不称完全未见保留测试。冻结模型与实例后只测量，不按本轮输赢选检查点、换实例、调λ或更新权重。', '',
        '| 方法 | PDI s11↓ | PDI s22↓ | 合并PDI↓ | 证明率↑ | PAR-2↓ | optimize秒↓ | Actor ms/调用↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    nr = by_panel[('native', 'pooled', 'all')]
    lines.append('| relpscost | — | — | {:.3f} | {:.1%} | {:.3f} | {:.3f} | — |'.format(nr['pdi'], nr['proof_rate'], nr['par2'], nr['wall']))
    for variant in VARIANTS:
        a, b, p = (by_panel[(variant, s, 'all')] for s in (11, 22, 'pooled'))
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.1%} | {:.3f} | {:.3f} | {} |'.format(
            variant, a['pdi'], b['pdi'], p['pdi'], p['proof_rate'], p['par2'], p['wall'],
            '{:.3f}'.format(p['actor_ms']) if p['actor_ms'] is not None else '—'))
    lines += ['', 'PDI按实例等权，solver与训练种子为实例内重复。证明要求SCIP时间≤120秒；'
        'PAR-2证明完成取SCIP时间，否则计240秒。原生没有训练种子，不把重复使用参照当额外样本。', '',
        '| 比较 | 全体PDI变化 s11 / s22 | 共同Actor PDI变化 s11 / s22 |', '|---|---|---|']
    for label in ('structure_noBC', 'structure_BC', 'BC_within_MLP', 'combined'):
        values = []
        for tag in ('all', 'common_actor'):
            es = [e for e in effects if e['effect'] == label and e['panel'] == tag and e['training_seed'] in SEEDS]
            values.append(' / '.join('{:+.2f}%'.format(e['pdi_percent']) if e['pdi_percent'] is not None else 'undefined(ref0)' for e in es) or '无')
        lines.append('| {} | {} | {} |'.format(label, *values))
    lines += ['', '共同Actor实例{}；无Actor实例{}；混合参与实例{}。无Actor的PDI计时变化不归因于神经动作，'
        '分解见mean_decomposition.csv。'.format(active, inactive, mixed), '',
        '各实例所有方法与原生均在同一CPU分片配对。分片可以异构，cohorts.csv记录实际主机、CPU和成本；'
        '聚合绝对时延混合了实际访问状态与CPU，不是统一硬件/同状态延迟基准。本轮66全部新测，不填入旧8实例结果。'
        'mspp16在分片3普通worker全部退出后串行运行，全部模型与原生采用相同单worker/24GiB/300秒watchdog例外；其余8GiB/180秒watchdog，SCIP搜索均120秒。', '',
        '旧权重训练时结构间CUDA初始流未完全对齐，该局限不会被新增测试消除。'
        '两个训练种子与已有66实例不能证明普遍泛化或统计显著，积分上下界也不是统计置信区间。', '',
        'E1是本轮测试测量；E2是Actor/实例SHA、单线程CPU、概率/动作fork一致性、Critic零调用和无fallback核验；'
        'E3为收益原因及更广泛效果，不能仅凭本表确认。正负结果均保留，不自动训练或进入其他阶段。', '',
        '累计墙时以各RUN/shard-N/job_state.json最终值为准，每分片上限14400秒；报告与CSV持久保存，不依赖节点/tmp。']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=SHARDS)
    args = parser.parse_args()
    if args.shard is not None:
        complete = render(DOCS / ('shard-' + str(args.shard)), [load_shard(args.shard)])
        print('Shard{} report complete={}'.format(args.shard, complete))
        return 0 if complete else 1
    DOCS.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = render(DOCS, [load_shard(i) for i in SHARDS], whole_matrix=True)
    print('Global frozen test66 report complete={}; {}'.format(complete, DOCS / 'REPORT.md'))
    # Partial progress is normal while other shards run; details in analysis.json.
    return 0


if __name__ == '__main__':
    sys.exit(main())
