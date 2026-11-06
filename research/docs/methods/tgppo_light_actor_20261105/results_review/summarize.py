"""Read-only analysis of completed jobs and the already-running CPU evaluation.

This script never imports the training runtime, starts SCIP, or submits a job.
--watch only waits for the existing finalizer and refreshes these review files.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'tgppo/runs/light-actor-20261105'
PACKAGE = ROOT / 'tgppo/experiments/light-actor-20261105'
VARIANTS = ('b0', 'b0_bc', 'mlp', 'mlp_bc', 'set', 'set_bc', 'isab', 'isab_bc')
KINDS = ('b0', 'mlp', 'set', 'isab')
SEEDS = (11, 22)
PARAMS = dict(b0=4738895, mlp=24493, set=32749, isab=92205)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_csv(name, rows):
    if not rows:
        return
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def collect_training():
    training, bc, hashes, initials = [], [], {}, {}
    release = read(PACKAGE / 'docs/release.json')
    for prefix, mapping in ((PACKAGE, release['files']), (ROOT, release['external_inputs'])):
        for name, expected in mapping.items():
            assert sha(prefix / name) == expected, ('Frozen input changed', name)
    protected = read(ROOT / 'tgppo/experiments/candidate-residual-20261104/docs/protected_sources.json')
    for name, expected in protected.items():
        assert sha(ROOT.parent / name) == expected, ('Protected source changed', name)
    for variant in VARIANTS:
        for seed in SEEDS:
            out = RUN / variant / 'training' / ('B0-s' + str(seed))
            jp = RUN / 'jobs' / ('{}-s{}.json'.format(variant, seed))
            job, summary, updates, init = read(jp), read(out/'summary.json'), read(out/'updates.json'), read(out/'initialization.json')
            assert job['status'] == summary['status'] == 'training_complete'
            assert job['next_episode'] == summary['episodes_seen'] == summary['trained_episodes'] == 100
            assert summary['pending_transitions'] == 0
            assert sum(u['num_updates'] for u in updates if u['updated']) == summary['optimizer_steps']
            assert sum(bool(u['updated']) for u in updates) == summary['complete_ppo_updates']
            assert sha(out / 'latest.pt') == summary['final_checkpoint_sha256']
            rows = []
            for p in sorted((out/'collection').glob('*/stack-*/selected_results.json')):
                rows.extend(read(p))
            assert len(rows) == 100 and len({r['job_id'] for r in rows}) == 100
            assert all(r['valid'] and r['worker_cpu_complete'] and not r['cuda_initialized'] for r in rows)
            environment = job['attempts'][-1]['environment']
            skipped = [u for u in updates if not u['updated']]
            training.append(dict(variant=variant, training_seed=seed, episodes=100,
                ppo_updates=summary['complete_ppo_updates'], optimizer_steps=summary['optimizer_steps'],
                unique_collected_transitions=sum(r['num_transitions'] for r in rows),
                skipped_update_at_episodes=';'.join(str(u['episodes_seen']) for u in skipped),
                skipped_sample_counts=';'.join(str(u['num_samples']) for u in skipped),
                job_minutes=job['charged_seconds']/60,
                learner_minutes=sum(u.get('learning_wall_seconds',0) for u in updates)/60,
                max_clip_fraction=max(u.get('clip_fraction',0) for u in updates),
                max_initial_ratio_error=max(u.get('initial_ratio_max_error',0) for u in updates),
                cpu=environment['stack']['cpu_model'], gpu=environment['gpu'],
                episode100_sha256=sha(out/'episode-0100.pt'), actor_sha256=sha(out/'actor-0100.pt')))
            initials[(variant, seed)] = init
            for p in (jp, out/'summary.json', out/'updates.json', out/'initialization.json', out/'episode-0100.pt', out/'actor-0100.pt'):
                hashes[str(p.relative_to(ROOT))] = sha(p)
            if variant.endswith('_bc'):
                b = read(out/'pretrain/result.json')
                assert b['completed_steps'] == 1000
                assert sha(out/'pretrain/actor.pt') == b['actor_sha256']
                options = [(0, b['initial']['validation']['macro']['normalized_nll'])]
                options += [(x['step'], x['validation']['macro']['normalized_nll']) for x in b['history'] if 'validation' in x]
                assert min(options, key=lambda x: (x[1], x[0]))[0] == b['selected_step']
                m = b['final']['validation']['macro']
                bc.append(dict(variant=variant, training_seed=seed, completed_steps=b['completed_steps'],
                    selected_step=b['selected_step'], validation_normalized_nll=m['normalized_nll'],
                    validation_accuracy=m['accuracy'], train_accuracy=b['final']['train']['macro']['accuracy'],
                    final_step_validation_normalized_nll=options[-1][1], seconds=b['seconds']))
                hashes[str((out/'pretrain/result.json').relative_to(ROOT))] = sha(out/'pretrain/result.json')
    audit = dict(frozen_files_verified=len(release['files']), external_inputs_verified=len(release['external_inputs']),
        protected_sources_verified=len(protected), jobs_complete=16, episodes=1600, source_sha256=hashes,
        same_critic_initialization_by_seed={str(s): len({initials[(v,s)]['common_critic_fingerprint'] for v in VARIANTS}) == 1 for s in SEEDS},
        same_cpu_torch_rng_by_seed={str(s): len({initials[(v,s)]['public_torch_rng_sha256'] for v in VARIANTS}) == 1 for s in SEEDS},
        same_cuda_rng_all_architectures_by_seed={str(s): len({initials[(v,s)]['public_cuda_rng_sha256'] for v in VARIANTS}) == 1 for s in SEEDS},
        same_cuda_rng_within_architecture_by_seed={str(s): all(initials[(k,s)]['public_cuda_rng_sha256'] == initials[(k+'_bc',s)]['public_cuda_rng_sha256'] for k in KINDS) for s in SEEDS},
        b0_seed11_bc_has_identical_initial_actor=initials[('b0',11)]['actor_fingerprint'] == initials[('b0_bc',11)]['actor_fingerprint'])
    write_csv('training.csv', training); write_csv('bc.csv', bc); write_json(HERE/'verification.json', audit)
    return training, bc, audit


def evaluation_snapshot():
    cohorts = list((RUN/'evaluation/episode-0100').glob('*/native.json'))
    assert len(cohorts) <= 1, 'Multiple CPU cohorts must not be silently pooled'
    native, tasks, coverage = [], [], []
    if cohorts:
        cohort = cohorts[0].parent
        native = read(cohort/'native.json')
        for variant in VARIANTS:
            for seed in SEEDS:
                folder = cohort/('{}-s{}'.format(variant,seed))
                selected = folder/'selected_results.json'
                rows = read(selected) if selected.exists() else []
                coverage.append(dict(variant=variant,training_seed=seed,committed_tasks=len(rows),
                    raw_tasks=len(list(folder.glob('attempt-*/dev-*.json')))))
                if rows:
                    assert len(rows) == 10 and len({r['job_id'] for r in rows}) == 10
                    assert all(r['valid'] and r['worker_cpu_complete'] and not r['cuda_initialized'] for r in rows)
                for r in rows:
                    tasks.append(dict(variant=variant,training_seed=seed,instance=r['instance_name'],solver_seed=r['solver_seed'],
                        pdi=r['pdi_horizon_midpoint'],nodes=r['nnodes'],proof=bool(r['proof_completed'] and r['scip_solve_time']<=120),
                        wall=r['optimize_wall_seconds'],actor_calls=r['branch_count'],actor_seconds=r['callback_seconds'].get('actor',0)))
    complete = (RUN/'results/complete.json').exists()
    if complete:
        done = read(RUN/'results/complete.json')
        assert done['complete'] and done['episode'] == 100 and done['seeds'] == [11,22] and len(tasks) == 160
    budget=read(RUN/'evaluation/budget.json') if (RUN/'evaluation/budget.json').exists() else None
    attempts=(budget or {}).get('attempts',[])
    stopped=bool(attempts and 'elapsed_seconds' in attempts[-1] and not complete)
    execution_status='complete' if complete else ('stopped_incomplete' if stopped else 'in_progress')
    snapshot = dict(updated_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        complete=complete,committed_model_tasks=len(tasks),expected_model_tasks=160,
        native_tasks=len(native),coverage=coverage,execution_status=execution_status,budget=budget)
    write_json(HERE/'evaluation_status.json',snapshot)
    write_csv('development_tasks.csv',tasks)
    return snapshot,native,tasks


def mean(rows,key):
    return statistics.mean(r[key] for r in rows)


def render(training, bc, audit):
    status,native,tasks=evaluation_snapshot()
    retention=read(HERE/'retention.json') if (HERE/'retention.json').exists() else None
    lines=['# 轻量 Actor × BC × PPO：训练结果核验与开发评估', '',
        '更新：{}。原始训练/评估、权重、冻结代码和先前报告不改；本目录为新增只读分析。'.format(status['updated_beijing']), '',
        '## 当前可以报告什么', '',
        '- 实际完成的是 **8条件 × 2训练种子（11/22）=16次训练**，每次100 episode，共1600 episode；不是仅seed=11。全部有完整权重和终态记录。',
        '- 所有训练均为 Gold 6148 + V100-SXM2-32GB，同型号CPU/GPU、同软件；没有本轮训练型号混杂，但没有证明不同容器的实际负载完全相同。',
        '- **已确认 BC 的结构差异：MLP/SET/ISAB 都能学到专家信号，原B0仍近均匀。** 轻量结构+专家预训练是可行的学习路径，能否改善求解以同机PDI为准。',
        '- 当前统一CPU开发评估：**{}/160模型任务已提交完整模型批次，另有10项匹配原生参照**。状态：{}。没有运行66实例测试集。'.format(status['committed_model_tasks'],'完整' if status['complete'] else ('异常停止，尚未完整' if status['execution_status']=='stopped_incomplete' else '仍在执行，不能提供完整方法排名')), '',
        '原生参照实际从server6既有缓存复用，匹配Gold6148型号、软件指纹和完整求解身份，并非本轮同步重测。模型之间在同一CPU部署环境比较；与缓存原生的计时/PDI比较保留时段与负载差异边界。', '',
        '## BC 的独立增量', '',
        '同一合法专家数据：12个TGPPO-train实例拟合1468状态，另3个train内部留出实例288状态；dev/test不参与拟合。SCIP8→SCIP6、cutoff及两项计数分母差异保留为迁移边界。', '',
        '所有BC组完成1000步；在预先冻结的0/100/200/400/600/800/1000步中按内部留出归一化NLL选初始化。以下是**参与选模的内部验证指标**，不是独立泛化确认；均匀NLL参考1、动作一致率参考8.35%。', '',
        '| Actor | 种子 | 选中步 | 选中NLL/log L↓ | 专家一致率↑ | 1000步末NLL | BC秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b in bc:
        lines.append('| {} | {} | {} | {:.4f} | {:.2%} | {:.4f} | {:.1f} |'.format(b['variant'],b['training_seed'],b['selected_step'],b['validation_normalized_nll'],b['validation_accuracy'],b['final_step_validation_normalized_nll'],b['seconds']))
    lines += ['',
        '**解释：** MLP/SET两种子都选100步，ISAB选200/100步；继续到1000步并未提高该NLL，不能通过单纯加大BC步数解决。先前400步小试门槛失败的记录保留；本轮按提前冻结的早停选模后，MLP/SET都低于0.90，ISAB-s11为0.9040仍略高，不能改写为所有结构均通过。', '',
        '**B0-s11的BC选中0步，初始化Actor与无BC组逐字节相同。** 因此这对条件并没有施加有效BC初始化；其后任何求解差异都不能归因于BC。B0-s22选100步，但NLL约1，只是微小数值变化，未形成有用专家拟合。', '',
        '## PPO 后专家信号是否保留', '']
    if retention and retention.get('complete'):
        table={(r['variant'],r['seed']):r for r in retention['rows']}
        analysis_budget=read(HERE/'analysis_budget.json')
        lines += ['本地CPU仅前向读取16份episode100权重，在相同288个既有专家状态上复核；没有优化步、GPU初始化或SCIP求解。模型读取/前向约{:.1f}秒，含启动约{:.1f}秒；与原准备合计{:.1f}/600秒。此表衡量专家迁移信号，PPO自身并不优化该NLL。'.format(retention['seconds'],analysis_budget['attempts'][-1]['elapsed_seconds'],analysis_budget['total_including_prior_preparation_seconds']),'',
            '| Actor | 无BC：PPO后NLL，s11 / s22 | BC初始化NLL，s11 / s22 | BC→PPO后NLL，s11 / s22 |',
            '|---|---|---|---|']
        bybc={(b['variant'],b['training_seed']):b for b in bc}
        for k in KINDS:
            plain=[table[(k,s)]['validation']['macro']['normalized_nll'] for s in SEEDS]
            start=[bybc[(k+'_bc',s)]['validation_normalized_nll'] for s in SEEDS]
            after=[table[(k+'_bc',s)]['validation']['macro']['normalized_nll'] for s in SEEDS]
            pair=lambda xs:' / '.join('{:.4f}'.format(x) for x in xs)
            lines.append('| {} | {} | {} | {} |'.format(k,pair(plain),pair(start),pair(after)))
        lines += ['',
            'MLP+BC、ISAB+BC的NLL在PPO后进一步降低；SET+BC有回退但仍优于其无BC条件。**没有观察到三种模型共同的“专家信号被完全遗忘”**，不能据此立刻增加防遗忘损失。NLL更低不等价于更低PDI。16个Actor权重均有实际变化，不能把B0的clip fraction=0解释成优化器没有执行。', '']
    lines += ['## 真实求解结果与结构/BC归因', '']
    if not status['complete']:
        lines += [('原统一评估已异常停止。SET+BC-s11的两个roi5alpha10n8任务触发180秒watchdog，100项已提交完整批次之外另有8项有效原始结果，仍缺52项有效结果；原预算90分钟已用约47分钟，保留所有失败。续接选择见continuation/RECOVERY.md。' if status['execution_status']=='stopped_incomplete' else '统一评估仍由最后完成的原Custom作业执行；本分析没有重复发起求解。')+' 仅已提交的完整10任务模型批次计入development_tasks.csv，不把未完成组的均值用于排名。', '',
            '**当前还不能说哪种Actor提高了PDI、哪种胜过relpscost，也不能声称测试集评测完成。** 本目录汇总进程在原评估结束后更新这节和配对表，不会启动训练或额外评估。', '',
            '开发协议固定episode100，5实例×solver seeds0/1；完整结果将分开比较同结构BC差、同预训练结构差，并保留无Actor实例及其分解。']
    else:
        by={(v,s):[t for t in tasks if t['variant']==v and t['training_seed']==s] for v in VARIANTS for s in SEEDS}
        nat=mean(native,'pdi_horizon_midpoint');natproof=mean([dict(proof=bool(r['proof_completed'] and r['scip_solve_time']<=120)) for r in native],'proof')
        lines += ['全部160项模型+10项原生有效完成，固定episode100。**这是5实例开发集，不是测试集。** PDI越小越好，表中均值等权训练种子及实例，solver重复不当作额外独立实例。', '',
            '原生relpscost：PDI **{:.3f}**，证明完成率 **{:.1%}**。'.format(nat,natproof), '',
            '| 条件 | PDI s11 | PDI s22 | 两种子均值 | 相对B0变化↓ | 相对原生变化↓ | 证明率 | Actor ms/调用 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        b0=mean([t for t in tasks if t['variant']=='b0'],'pdi')
        aggregate=[]
        for v in VARIANTS:
            rs=[t for t in tasks if t['variant']==v];pdi=mean(rs,'pdi');calls=sum(t['actor_calls'] for t in rs)
            ms=1000*sum(t['actor_seconds'] for t in rs)/calls if calls else 0
            lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:+.2%} | {:+.2%} | {:.1%} | {:.3f} |'.format(v,mean(by[(v,11)],'pdi'),mean(by[(v,22)],'pdi'),pdi,pdi/b0-1,pdi/nat-1,mean(rs,'proof'),ms))
            aggregate.append(dict(variant=v,pdi_s11=mean(by[(v,11)],'pdi'),pdi_s22=mean(by[(v,22)],'pdi'),mean_pdi=pdi,
                relative_b0=pdi/b0-1,relative_native=pdi/nat-1,proof_rate=mean(rs,'proof'),actor_ms_per_call=ms))
        lookup={(r['variant'],r['training_seed'],r['instance'],r['solver_seed']):r for r in tasks}
        pairs=[]
        for t in tasks:
            v=t['variant'];k=v[:-3] if v.endswith('_bc') else v;refs=[]
            if k!='b0':refs.append(('structure_same_pretraining','b0_bc' if v.endswith('_bc') else 'b0'))
            if v.endswith('_bc'):refs.append(('bc_within_structure',k))
            for effect,ref in refs:
                base=lookup[(ref,t['training_seed'],t['instance'],t['solver_seed'])]
                pairs.append(dict(effect=effect,variant=v,reference=ref,training_seed=t['training_seed'],instance=t['instance'],solver_seed=t['solver_seed'],
                    pdi_delta=t['pdi']-base['pdi'],reference_pdi=base['pdi'],method_pdi=t['pdi'],
                    nodes_delta=t['nodes']-base['nodes'],wall_delta=t['wall']-base['wall'],
                    both_actor_active=t['actor_calls']>0 and base['actor_calls']>0,
                    neither_actor_active=t['actor_calls']==0 and base['actor_calls']==0))
        write_csv('paired.csv',pairs);write_csv('development_summary.csv',aggregate)
        instances=sorted({t['instance'] for t in tasks})
        inactive=[n for n in instances if all(t['actor_calls']==0 for t in tasks if t['instance']==n)]
        active=[n for n in instances if all(t['actor_calls']>0 for t in tasks if t['instance']==n)]
        instance_rows=[]
        for v in VARIANTS:
            for n in instances:
                rs=[t for t in tasks if t['variant']==v and t['instance']==n]
                instance_rows.append(dict(variant=v,instance=n,mean_pdi=mean(rs,'pdi'),proof_rate=mean(rs,'proof'),
                    mean_nodes=mean(rs,'nodes'),mean_wall=mean(rs,'wall'),actor_calls=sum(t['actor_calls'] for t in rs)))
        write_csv('development_by_instance.csv',instance_rows)
        lines += ['', '完全没有Actor调用的实例：{}；所有条件均调用Actor的共同实例：{}。完整主表保留所有实例，另作以下归因分解：'.format('、'.join(inactive) or '无','、'.join(active) or '无')]
        if active:
            active_native=mean([r for r in native if r['instance_name'] in active],'pdi_horizon_midpoint')
            active_b0=mean([r for r in tasks if r['variant']=='b0' and r['instance'] in active],'pdi')
            lines += ['', '| 条件 | 共同Actor实例平均PDI | 相对B0 | 相对原生 |','|---|---:|---:|---:|']
            for v in VARIANTS:
                value=mean([r for r in tasks if r['variant']==v and r['instance'] in active],'pdi')
                lines.append('| {} | {:.3f} | {:+.2%} | {:+.2%} |'.format(v,value,value/active_b0-1,value/active_native-1))
        candidate=min((a for a in aggregate if a['variant'].endswith('_bc') and a['variant']!='b0_bc'),key=lambda a:a['mean_pdi'])
        lines += ['', '仅按当前开发全表均值，轻量+BC中最低的是 **{}**：相对B0 {:+.2%}，相对原生 {:+.2%}。这只是开发选择信号，必须结合上面的Actor分解、证明率和两种子方向；不能据此称显著或测试集获胜。'.format(candidate['variant'],candidate['relative_b0'],candidate['relative_native'])]
        lines += ['', '同结构BC的独立增量（负数有利；括号为两边均调用Actor的配对）：', '',
            '| 结构 | BC相对无BC，s11 | BC相对无BC，s22 | 参与Actor的配对均值差s11 / s22 |',
            '|---|---:|---:|---|']
        for k in KINDS:
            pp=[p for p in pairs if p['effect']=='bc_within_structure' and p['variant']==k+'_bc']
            d=[];active=[]
            for s in SEEDS:
                ss=[p for p in pp if p['training_seed']==s]
                d.append(mean(ss,'pdi_delta')/mean(ss,'reference_pdi'))
                aa=[p for p in ss if p['both_actor_active']];active.append('{:+.3f}'.format(mean(aa,'pdi_delta')) if aa else 'N/A')
            lines.append('| {} | {:+.2%} | {:+.2%} | {} |'.format(k,*d,' / '.join(active)))
        lines += ['', '逐实例/solver配对见paired.csv，不能用无Actor实例的计时差归因于神经评分；也不能从两种子、5开发实例推为显著或广泛泛化。真实部署ms/调用包含策略看到的状态分布差异，受控结构延迟仍以此前同状态探针为准。']
    lines += ['', '## 训练预算、对照边界与源码机制', '',
        '- 16组入口总用时{:.2f}作业小时，单组{:.1f}–{:.1f}分钟；这是并行作业墙时之和，不能当作串行等待时间。每组包含准备、BC/桥接和PPO。'.format(sum(t['job_minutes'] for t in training)/60,min(t['job_minutes'] for t in training),max(t['job_minutes'] for t in training)),
        '- MLP/SET/ISAB参数为24,493/32,749/92,205，B0为4,738,895，分别减少99.48%/99.31%/98.05%。成本增量已由之前同状态CPU探针验证；实际求解收益另计。',
        '- 所有组100 episode、24或25轮有效PPO更新，无最终待更新transition。SET+BC和ISAB+BC每种子各有一批不足256个样本，按原Agent.learn累积到后续批次；不是训练失败。training.csv记录准确批次与样本数。',
        '- 固定episode不意味着相同决策样本、Adam步数或学习器时间。BC组样本更少既可能来自更快完成搜索，也可能来自策略分布变化，不能仅凭这个数字声称求解更好。',
        '- Critic初始化和CPU Torch RNG在每个种子的8组中相同；**CUDA RNG在B0与轻量结构之间不同**，但同结构有/无BC之间相同。源码原因：轻量初始化在fork_rng(devices=[])内调用torch.manual_seed，CPU流恢复，CUDA流未被该fork保护。当前权重/代码不改；今后修复此处并做目标GPU状态核查，不能声称本轮公共GPU随机流完全一致。该差异会影响Critic dropout，是结构对照的额外随机性边界，不是BC配对同类差异。',
        '- 原Actor/Critic/Agent/H3源码未改；冻结{}文件、{}外部输入、{}受保护源码SHA均核对通过。'.format(audit['frozen_files_verified'],audit['external_inputs_verified'],audit['protected_sources_verified']), '',
        '## 下一步：先形成求解结论，再决定新实验', '',
        ('1. **本阶段现有统一开发评估已经完整，提交本报告供验收。无需再申请GPU，不追加训练。** 全16组共享90分钟CPU评估上限，完成后不重复启动finalize。' if status['complete'] else '1. **本阶段先完成现有统一开发评估并提交方法主表。无需再申请GPU，不追加训练。** 全16组共享90分钟CPU评估上限；当前由远端最后一个作业在执行，不能因尚无总表重复启动finalize。'),
        '2. 按同结构BC差和同预训练结构差筛选；要求看两种子方向、逐实例收益、证明率、Actor参与分解及相对relpscost。若只有BC一致率提升，不把它写成PDI收益；若仅无Actor项改善，不归因于模型。',
        '3. 若至少一个轻量+BC结构有可信的开发求解收益，下一阶段优先冻结其权重，做小规模跨实例确认；可保留B0/轻量×无BC/有BC的2×2对照。8实例×2训练种子×2solver种子×4条件=128模型任务，加16原生，共144项；正式实例清单、CPU同机缓存和墙时预算须在验收后冻结。先用本地CPU估时，确需额外资源再交Custom，不申请GPU训练。',
        '4. 若内部NLL有效但开发PDI无收益，先复用已保存的BC权重，在少量已知正反例做BC-only与BC→PPO对照，区分专家初始化本身、PPO目标及迁移分布；这属于下一阶段提案，尚未运行。当前没有共同的完全遗忘证据，不直接增加CL/辅助损失或扩大训练。',
        '5. 后续确有必要训练时，先在新独立副本修正CUDA随机流保护；不篡改本轮冻结代码、旧结果或用新配置续接旧权重冒充同一矩阵。', '',
        '用户随后已授权收齐评估及条件式跨实例确认；最新筛选规则与后续状态以[continuation/PLAN.md](../continuation/PLAN.md)及同目录DEVELOPMENT_REVIEW.md为准。仅筛选通过才开展已授权的小规模CPU确认；不追加GPU训练。CR线、历史/Mamba、树信用分配和66测试矩阵均不自动启动。', '',
        '数据索引：training.csv、bc.csv、retention.json、verification.json、evaluation_status.json；统一开发结果完整后另有development_summary.csv与paired.csv。原始结果位于research/tgppo/runs/light-actor-20261105/results。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    brief=['# 周五方法汇报：轻量 Actor + 专家预训练 + PPO', '',
        '状态时间：'+status['updated_beijing'], '',
        '- 方法：保持25/8/53廉价特征、原Critic/H3/PPO，替换为共享MLP、Deep Sets、单ISAB；原B0与三结构分别有/无BC，两个训练种子，独立比较结构与预训练。',
        '- 已完成：16次100episode训练；三个轻量Actor参数减少98.05%–99.48%。',
        '- 学习结果：MLP/SET内部选模NLL约0.83–0.85、专家一致率约39%–40%；B0近均匀。PPO后MLP+BC和ISAB+BC保留专家信号。',
        '- 求解结果：'+('160项同机开发评估完整，见REPORT.md主表；5开发实例不能称测试集。' if status['complete'] else ('原评估因两个watchdog异常停止，续接方案见continuation/RECOVERY.md；不能提供完整方法排名。' if status['execution_status']=='stopped_incomplete' else '同机开发评估仍在运行，当前不能宣称PDI提升或胜过relpscost。')),
        '- 边界：BC内部留出参与初始化选择；SCIP8→6专家迁移；2训练种子；结构比较的CUDA初始随机流未完全对齐，后续修复；旧基线和本轮权重保留。',
        '- 下一步：先确认现有开发PDI/证明率与配对归因，再冻结候选做小规模跨实例；现在不追加GPU训练。', '',
        '完整证据：[REPORT.md](REPORT.md)。']
    (HERE/'MEETING_BRIEF_20261106.md').write_text('\n'.join(brief)+'\n')
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--timeout',type=float,default=5400)
    args=p.parse_args();training,bc,audit=collect_training();start=time.monotonic();last=None
    write_json(HERE/'review_process.json',dict(pid=os.getpid(),host=os.uname().nodename,stopped=False,
        watch=args.watch,started_epoch=time.time(),maximum_watch_seconds=args.timeout,
        scope='read-only reporting; no job submission, training, or solver execution'))
    while True:
        status=render(training,bc,audit)
        mark=(status['committed_model_tasks'],status['complete'])
        if mark != last:
            print(status['updated_beijing'],mark,flush=True);last=mark
        if not args.watch or status['complete'] or time.monotonic()-start>=args.timeout:
            break
        budget=status.get('budget') or {};attempts=budget.get('attempts',[])
        if attempts and 'elapsed_seconds' in attempts[-1] and not status['complete']:
            print('Existing evaluation stopped before completion; no automatic retry.',flush=True);break
        time.sleep(30)
    write_json(HERE/'review_process.json',dict(stopped=True,evaluation_complete=status['complete'],
        updated_beijing=status['updated_beijing'],committed_model_tasks=status['committed_model_tasks'],
        scope='read-only reporting; no job submission, training, or solver execution'))


if __name__=='__main__':
    main()
