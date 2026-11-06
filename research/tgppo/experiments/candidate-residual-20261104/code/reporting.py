"""Instance-paired development results; never substitutes probes for solve results."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from project.utils.protocol import write_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows):
    names = sorted({r['instance_name'] for r in rows})
    instance_means = [statistics.mean(r['pdi_horizon_midpoint'] for r in rows
                                    if r['instance_name'] == n) for n in names]
    branches = sum(r.get('branch_count', 0) for r in rows)
    return dict(pdi=statistics.mean(instance_means),
        pdi_lower=statistics.mean(statistics.mean(r['pdi_horizon_lower'] for r in rows
            if r['instance_name'] == n) for n in names),
        pdi_upper=statistics.mean(statistics.mean(r['pdi_horizon_upper'] for r in rows
            if r['instance_name'] == n) for n in names),
        proof_rate=statistics.mean(bool(r['proof_completed'] and r['scip_solve_time'] <= 120) for r in rows),
        no_actor_tasks=sum(r.get('branch_count', 0) == 0 for r in rows),
        actor_ms_per_branch=1000*sum(r.get('callback_seconds', {}).get('actor', 0.) for r in rows)/branches
            if branches else None,
        mean_solve_seconds=statistics.mean(r['scip_solve_time'] for r in rows),
        mean_worker_seconds=statistics.mean(r['worker_wall_seconds'] for r in rows)
            if all(isinstance(r.get('worker_wall_seconds'),(int,float)) for r in rows) else None,
        mean_nodes=statistics.mean(r['nnodes'] for r in rows), tasks=len(rows), instances=len(names))


def report(root, matrix, episode, job_status):
    destination = matrix/'results'
    destination.mkdir(parents=True, exist_ok=True)
    complete_file = matrix/'evaluation'/('episode-{:04d}'.format(episode))/'complete.json'
    if not episode or not complete_file.exists():
        write_json(destination/'summary.json', dict(complete=False, episode=episode, jobs=job_status,
            reason='Common checkpoint or complete development evaluation unavailable; no performance values invented'))
        (destination/'REPORT.md').write_text('# 方法实验尚未完整\n\n尚无可报告的完整共同预算求解对照。'
            '请查看 summary.json 的逐作业状态；保留所有失败和未完成项。\n')
        return
    output = complete_file.parent
    manifest = read(output/'frozen_models.json')
    names, seeds = manifest['development'], manifest['solver_seeds']
    expected = {(n, s) for n in names for s in seeds}
    models = [('native', None, read(output/'native_rows.json'))]
    for m in manifest['models']:
        rows = read(output/('{}-s{}'.format(m['variant'], m['seed']))/'selected_results.json')
        models.append((m['variant'], m['seed'], rows))
    all_rows, model_rows, per_instance = [], [], []
    for variant, seed, rows in models:
        if len(rows) != len(expected) or {(r['instance_name'], r['solver_seed']) for r in rows} != expected:
            raise ValueError('Missing/duplicate dev rows: '+str((variant, seed)))
        if any(not r.get('valid') for r in rows):
            raise ValueError('Do not drop invalid evaluations')
        for r in rows:
            all_rows.append(dict(variant=variant, training_seed=seed, instance=r['instance_name'],
                solver_seed=r['solver_seed'], pdi=r['pdi_horizon_midpoint'],
                pdi_lower=r['pdi_horizon_lower'], pdi_upper=r['pdi_horizon_upper'],
                proof=bool(r['proof_completed'] and r['scip_solve_time'] <= 120),
                branches=r.get('branch_count', 0), nodes=r['nnodes'],
                solve_seconds=r['scip_solve_time'], worker_seconds=r.get('worker_wall_seconds'),
                actor_seconds=r.get('callback_seconds', {}).get('actor', 0.),
                callback_seconds=r.get('callback_seconds', {}).get('callback', 0.)))
        model_rows.append(dict(variant=variant, training_seed=seed, **aggregate(rows)))
        if variant != 'native':
            meta=next(m for m in manifest['models'] if m['variant']==variant and m['seed']==seed)
        else:
            meta={}
        model_rows[-1].update({k:meta.get(k) for k in ('complete_ppo_blocks',
            'logical_minibatch_updates','pending_transitions','trained_episodes')})
        for name in names:
            selected = [r for r in rows if r['instance_name'] == name]
            per_instance.append(dict(variant=variant, training_seed=seed, instance=name, **aggregate(selected)))
    paired = []
    for seed in (11, 22):
        for name in names:
            values = {v: next(r for r in per_instance if r['variant'] == v and
                r['training_seed'] == seed and r['instance'] == name) for v in ('b0', 'score', 'candidate')}
            b, s, c = (values[v] for v in ('b0', 'score', 'candidate'))
            paired.append(dict(training_seed=seed, instance=name,
                b0_pdi=b['pdi'], score_pdi=s['pdi'], candidate_pdi=c['pdi'],
                candidate_minus_b0=c['pdi']-b['pdi'], candidate_minus_score=c['pdi']-s['pdi'],
                candidate_minus_b0_lower=c['pdi_lower']-b['pdi_upper'],
                candidate_minus_b0_upper=c['pdi_upper']-b['pdi_lower'],
                candidate_minus_score_lower=c['pdi_lower']-s['pdi_upper'],
                candidate_minus_score_upper=c['pdi_upper']-s['pdi_lower'],
                candidate_no_actor=c['no_actor_tasks'], score_no_actor=s['no_actor_tasks'],
                b0_no_actor=b['no_actor_tasks']))
    learning, updates = [], []
    for v in ('b0', 'score', 'candidate'):
        for seed in (11, 22):
            directory = matrix/v/'training'/('B0-s'+str(seed))
            for f in sorted((directory/'collection').glob('e*/selected_results.json')):
                for r in read(f):
                    index = int(r['job_id'].split('-')[0][1:])+1
                    if index <= episode:
                        learning.append(dict(variant=v, training_seed=seed, episode=index,
                            instance=r['instance_name'], pdi=r['pdi_horizon_midpoint'],
                            total_reward=r.get('total_reward'), decisions=r.get('branch_count', 0)))
            for r in read(directory/'updates.json'):
                if r['episodes_seen'] <= episode:
                    updates.append(dict(variant=v, training_seed=seed, episode=r['episodes_seen'],
                        updated=r['updated'], actor_loss=r.get('actor_loss'), critic_loss=r.get('critic_loss'),
                        entropy=r.get('entropy'), transitions=r.get('num_samples'),
                        logical_updates=r.get('num_updates', 0),
                        learning_wall_seconds=r.get('learning_wall_seconds')))
    for filename, rows in [('evaluation_rows.csv', all_rows), ('model_summary.csv', model_rows),
            ('instance_summary.csv', per_instance), ('paired_instances.csv', paired),
            ('learning.csv', learning), ('updates.csv', updates)]:
        write_csv(destination/filename, rows)
    summary = dict(complete=True, episode=episode, models=model_rows, paired=paired,
        jobs=job_status, evaluation_stack=manifest['stack'],
        scope='5 previously used development instances; exploratory, not independent generalization')
    write_json(destination/'summary.json', summary)
    lines = ['# TGPPO 候选信息残差评分：方法与开发实验', '',
        '共同训练预算：每模型{}完整episode；训练种子11/22，全部5开发实例×2求解器种子，120秒。'.format(episode),
        '固定共同里程碑，不按开发输赢选择权重；原生参照同时展示。', '',
        '方法：原logits + wᵀ(m_i+e_i)，w为零初始化256维向量；评分捷径消融仅用wᵀm_i。', '',
        '| 方法 | 训练种子 | PDI↓ | cutoff证明率 | 无Actor任务/10 | Actor调用ms/分支 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in model_rows:
        lines.append('| {} | {} | {:.4f} | {:.1%} | {} | {} |'.format(r['variant'],
            r['training_seed'] if r['training_seed'] is not None else '—', r['pdi'], r['proof_rate'],
            r['no_actor_tasks'], '{:.3f}'.format(r['actor_ms_per_branch']) if r['actor_ms_per_branch'] is not None else '—'))
    lines += ['', '候选残差相对参照的实例等权PDI差（负数更好）：', '',
        '| 训练种子 | CR−B0 | CR−评分捷径 |', '|---|---:|---:|']
    signs = []
    for seed in (11, 22):
        chosen = [r for r in paired if r['training_seed'] == seed]
        db = statistics.mean(r['candidate_minus_b0'] for r in chosen)
        ds = statistics.mean(r['candidate_minus_score'] for r in chosen)
        signs.append((db, ds))
        lines.append('| {} | {:.5f} | {:.5f} |'.format(seed, db, ds))
    if all(db < 0 and ds < 0 for db, ds in signs):
        finding = '两个训练种子的平均差均支持CR；仍需逐实例测量区间与更大独立评估，不能宣称显著性或普遍泛化。'
    elif all(db < 0 for db, ds in signs):
        finding = 'CR在两种子上优于原参照，但未一致优于评分捷径；候选信息的独立收益尚未证实。'
    else:
        finding = '未观察到CR相对原参照及评分捷径的一致增益；保留本轮方法负结果，不用概率响应代替求解收益。'
    lines += ['', finding, '', 'PDI上下界是超时过冲的测量边界，不是统计置信区间。'
        '不把solver重复当独立实例。Actor未参与的任务保留在主表，其差异不归因于神经决策。', '',
        '当前为120秒/Adam的探索协议，未完成原论文3600秒规模复现。系统核验不计入本轮方法成果。', '',
        '![方法](method.png)', '![逐实例配对](paired.png)',
        '![共同预算训练曲线](learning.png)', '![开发性能](performance.png)', '']
    (destination/'REPORT.md').write_text('\n'.join(lines))
    plotter = Path(__file__).with_name('plot_results.py')
    alternate = Path('/home/duweiyue25/conda/envs/rl4scip/bin/python')
    candidates = [sys.executable] + ([str(alternate)] if alternate.exists() else [])
    plotting = []
    for interpreter in candidates:
        result = subprocess.run([interpreter, str(plotter), str(destination)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        plotting.append(dict(interpreter=interpreter, returncode=result.returncode, log=result.stdout[-4000:]))
        if result.returncode == 0:
            break
    write_json(destination/'plotting.json', dict(passed=plotting[-1]['returncode'] == 0, attempts=plotting))
    write_json(destination/'artifact_hashes.json', {p.name: sha256(p) for p in sorted(destination.iterdir())
        if p.is_file() and p.name != 'artifact_hashes.json'})
    print('Results:', destination/'REPORT.md', flush=True)
