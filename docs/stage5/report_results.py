"""Generate stage5 tables/figures/report from all frozen outcomes; no model selection."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/research');DOC=ROOT/'docs/stage5';RUN=ROOT/'tgppo/runs/stage5'
REPORT=ROOT/'docs/阶段5_在线PPO对照与验收报告_TGPPO_2026-10-21.md'


def read(path):return json.loads(Path(path).read_text())

def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def mean(rows,key):return float(np.mean([r[key] for r in rows]))

def proof(row):return bool(row['proof_completed'] and row['scip_solve_time']<=120.+1e-6)

def summarize(rows):
    valid=[r for r in rows if r.get('valid')]
    if not valid:return dict(runs=len(rows),valid=0,failed=len(rows))
    callbacks=sum(r.get('branch_count',0) for r in valid)
    return dict(runs=len(rows),valid=len(valid),failed=len(rows)-len(valid),
        pdi=mean(valid,'pdi_horizon_midpoint'),pdi_lower=mean(valid,'pdi_horizon_lower'),pdi_upper=mean(valid,'pdi_horizon_upper'),
        proof_count=sum(proof(r) for r in valid),proof_fraction=sum(proof(r) for r in valid)/len(valid),
        par2=float(np.mean([r['scip_solve_time'] if proof(r) else 240. for r in valid])),
        solve_seconds=mean(valid,'scip_solve_time'),worker_wall_seconds=mean(valid,'worker_wall_seconds'),
        nodes=mean(valid,'nnodes'),branch_decisions=callbacks,
        total_branch_callback_seconds=sum(r.get('callback_seconds',{}).get('callback',0.) for r in valid),
        total_actor_seconds=sum(r.get('callback_seconds',{}).get('actor',0.) for r in valid),
        total_critic_seconds=sum(r.get('callback_seconds',{}).get('critic',0.) for r in valid),
        callback_ms_per_decision=(1000*sum(r.get('callback_seconds',{}).get('callback',0.) for r in valid)/callbacks if callbacks else None),
        actor_ms_per_decision=(1000*sum(r.get('callback_seconds',{}).get('actor',0.) for r in valid)/callbacks if callbacks else None),
        measurement_callback_seconds=sum(r.get('measurement_callback_seconds',0.) for r in valid),
        fallback_callbacks=sum(r.get('didnotrun_count',0) for r in valid),
        pdi_inexact_runs=sum(not r['pdi_horizon_exact'] for r in valid),
        no_branch_runs=sum(r.get('branch_count',0)==0 for r in valid))


def main():
    c=read(DOC/'execution_config.json');audit=read(DOC/'verification.json')
    if not audit['passed']:raise ValueError('Report requires successful audit; retain failure evidence')
    jobs=read(RUN/'evaluation/jobs.json');results=read(RUN/'evaluation/jobs/results.json');meta={j['job_id']:j for j in jobs}
    rows=[]
    fields=['valid','status','proof_completed','cutoff_proof','scip_solve_time','worker_wall_seconds','worker_cpu_seconds',
        'pdi_horizon_midpoint','pdi_horizon_lower','pdi_horizon_upper','pdi_horizon_exact','nnodes','branch_count',
        'gap','gap_finite','primal_bound','primal_bound_finite','dual_bound','dual_bound_finite','has_feasible_solution',
        'objective','original_optimal','didnotrun_count','fallback_action_count','measurement_callback_seconds','scip_time_overshoot_seconds']
    for r in results:
        j=meta[r['job_id']];row=dict(job_id=r['job_id'],group=j['group'],training_seed=j['training_seed'],
            solver_seed=j['solver_seed'],instance=j['instance_name'],budget_seconds=j['nominal_budget_seconds'],
            sampling_seed=j['sampling_seed'],checkpoint=j['checkpoint'])
        row.update({k:r.get(k) for k in fields});row['proof_within_120s']=proof(r)
        row['par2']=r['scip_solve_time'] if proof(r) else 240.
        for k,v in r.get('callback_seconds',{}).items():row['callback_'+k+'_seconds']=v
        rows.append(row)
    csvwrite(DOC/'all_solver_runs.csv',rows)
    enriched=[]
    for r in results:
        j=meta[r['job_id']];enriched.append(dict(r,group=j['group'],training_seed=j['training_seed'],instance=j['instance_name'],budget_seconds=j['nominal_budget_seconds']))
    primary=[r for r in enriched if r['budget_seconds']==7200.]
    native=[b['result'] for b in read(c['native_baselines']).values() if b['result']['instance_name'] in c['development']]
    group_table=[]
    for group in c['groups']:
        group_table.append(dict(group=group,**summarize([r for r in primary if r['group']==group])))
    group_table.append(dict(group='relpscost',**summarize(native)))
    csvwrite(DOC/'main_comparison.csv',group_table)
    per_instance=[];per_seed=[];curves=[];training=[];checkpointrows=[]
    for group in c['groups']:
        for name in c['development']:
            per_instance.append(dict(group=group,instance=name,**summarize([r for r in primary if r['group']==group and r['instance']==name])))
        for seed in c['training_seeds']:
            per_seed.append(dict(group=group,training_seed=seed,**summarize([r for r in primary if r['group']==group and r['training_seed']==seed])))
            d=RUN/'training'/('{}-s{}'.format(group,seed));s=read(d/'summary.json');training.append(s)
            for k,v in read(d/'selected_checkpoints.json').items():checkpointrows.append(dict(group=group,training_seed=seed,**v))
        for budget in (1800.,3600.,7200.):
            curves.append(dict(group=group,budget_seconds=budget,solver_seed=0,
                **summarize([r for r in enriched if r['group']==group and r['budget_seconds']==budget and r['solver_seed']==0])))
    for name in c['development']:
        per_instance.append(dict(group='relpscost',instance=name,**summarize([r for r in native if r['instance_name']==name])))
    csvwrite(DOC/'per_instance.csv',per_instance);csvwrite(DOC/'per_training_seed.csv',per_seed)
    csvwrite(DOC/'budget_curve_seed0.csv',curves);csvwrite(DOC/'training_summary.csv',training);csvwrite(DOC/'selected_checkpoints.csv',checkpointrows)
    comparisons=[];rng=np.random.default_rng(20261021);boot=rng.integers(0,len(c['development']),size=(10000,len(c['development'])))
    indexed={(r['group'],r['instance']):r for r in per_instance}
    for baseline in ('R0','Rprod','relpscost'):
        j=np.array([indexed['Rjoint',n]['pdi'] for n in c['development']]);b=np.array([indexed[baseline,n]['pdi'] for n in c['development']])
        d=j-b;relative=100.*(j[boot].mean(1)/b[boot].mean(1)-1.)
        seed_wins=None if baseline=='relpscost' else sum(next(r['pdi'] for r in per_seed if r['group']=='Rjoint' and r['training_seed']==s)<next(r['pdi'] for r in per_seed if r['group']==baseline and r['training_seed']==s) for s in c['training_seeds'])
        comparisons.append(dict(comparison='Rjoint-'+baseline,absolute_pdi_difference=float(d.mean()),
            pdi_change_percent=100.*(j.mean()/b.mean()-1.),instance_wins=int((j<b).sum()),instances=len(j),
            training_seed_wins=seed_wins,
            bootstrap_instance_95ci_absolute=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),
            bootstrap_instance_95ci_percent=np.quantile(relative,[.025,.975]).tolist(),
            measurement_difference_lower=float(np.mean([indexed['Rjoint',n]['pdi_lower']-indexed[baseline,n]['pdi_upper'] for n in c['development']])),
            measurement_difference_upper=float(np.mean([indexed['Rjoint',n]['pdi_upper']-indexed[baseline,n]['pdi_lower'] for n in c['development']]))))
    write(DOC/'paired_comparisons.json',comparisons)
    csvwrite(DOC/'paired_comparisons.csv',[dict(r,bootstrap_instance_95ci_absolute=str(r['bootstrap_instance_95ci_absolute']),bootstrap_instance_95ci_percent=str(r['bootstrap_instance_95ci_percent'])) for r in comparisons])
    summary=dict(stage=5,full_matrix_complete=True,training_runs=9,final_development_solves=135,early_development_solves=90,
        statistical_units=5,stage6_started=False,groups=group_table,comparisons=comparisons,
        training_cpu_seconds=sum(r['total_cpu_seconds'] for r in training),
        evaluation_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in results),
        bridge_cpu_seconds=read(DOC/'bridge_verification.json')['combined_bridge_cpu_seconds'],
        training_episode_range=[min(r['checkpoint_episodes'] for r in training),max(r['checkpoint_episodes'] for r in training)],
        complete_ppo_iteration_range=[min(r['complete_ppo_iterations'] for r in training),max(r['complete_ppo_iterations'] for r in training)])
    write(DOC/'summary.json',summary)
    colors={'R0':'#64748b','Rprod':'#d97706','Rjoint':'#0284c7'}
    fig,ax=plt.subplots(figsize=(7,4))
    for g in c['groups']:
        rr=[r for r in curves if r['group']==g];ax.plot([r['budget_seconds']/3600 for r in rr],[r['pdi'] for r in rr],'-o',label=g,color=colors[g])
    ax.set(xlabel='Nominal training CPU budget (core hours)',ylabel='Mean 120s PDI (lower is better)',title='Budget curve: same solver seed 0 at all points')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(DOC/'budget_curve.png',dpi=180);fig.savefig(DOC/'budget_curve.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for g in c['groups']:
        vals=[r['pdi'] for r in per_seed if r['group']==g];x=c['groups'].index(g)
        ax.bar(x,np.mean(vals),color=colors[g],alpha=.5);ax.scatter(np.full(3,x)+np.linspace(-.13,.13,3),vals,color=colors[g],s=28,zorder=3)
    ax.axhline(group_table[-1]['pdi'],color='#333',linestyle='--',label='relpscost (15 solves)')
    ax.set(xticks=range(3),xticklabels=c['groups'],ylabel='Mean 120s PDI',title='Final checkpoint: dots are training seeds (15 solves each)');ax.legend();fig.tight_layout()
    fig.savefig(DOC/'final_pdi.png',dpi=180);fig.savefig(DOC/'final_pdi.svg');plt.close(fig)
    joint=comparisons[0];prod=comparisons[1]
    stable=all(r['bootstrap_instance_95ci_absolute'][1]<0 and r['measurement_difference_upper']<0 and r['training_seed_wins']==3 for r in comparisons[:2])
    conclusion=('本轮有限预算实验中，联合组相对两组对照的改善较一致，但仍只限于五个开发实例。' if stable else '在线流程已跑通；本轮还没有稳定证据证明联合信息改善实际求解。')
    text=['# 阶段 5：在线 PPO 对照与验收报告','',
        '启动与授权日期：2026-10-21。用户：“验收通过，进入下一个阶段”。本阶段已完成，**等待阶段 5 验收；未进入阶段 6，也未启动后续方法或扩预算训练。**','',
        '## 结果先说人话','',conclusion,'',
        '九次独立在线训练、135 次最终开发评估和 90 次早期预算点评估均已完成。联合组最终 PDI 相对原信息组变化 **{:+.2f}%**，相对同容量边际组变化 **{:+.2f}%**；负数表示更好。完整更新涵盖的任务数为 **{}—{}**，完整 PPO 轮数为 **{}—{}**。这仍是有限预算复现，不能称为复现论文 500 episode 的性能。'.format(joint['pdi_change_percent'],prod['pdi_change_percent'],*summary['training_episode_range'],*summary['complete_ppo_iteration_range']),'',
        '## 公平比较与共同适配','',
        'R0 是共同校准后的原信息对照，Rprod 和 Rjoint 分别使用边际乘积与真实联合直方图。三组按训练种子配对 Actor/公共 Critic 初始权重、任务顺序及动作随机流；统一使用训练侧 μ/σ、MSE、原学习率、H3、120 秒、3 epochs 和完整 Critic。三组均从新初始化开始，未加载各自离线拟合头，也未使用开发最优检查点。最终求解只加载 Actor。','',
        '新增研究因素是 Critic 的搜索信息；静态价值尺度及输出初始化属于各组共同适配。Rjoint 与 Rprod 同参数量、同提取过程，其对照更接近关联结构的作用；Rjoint 与 R0 的差别还含额外容量。离线容量对照不能代替尚未实施的在线容量消融。','',
        '## 最终开发结果','',
        '每个学习组 45 次＝5 实例×3 训练种子×3 求解器种子；原生 relpscost 复用同协议的 15 次。统计单位是实例，不能把 45 次当作 45 个独立问题。PDI 越低越好；证明率和 PAR-2 保守按证明在 120 秒内完成统计，原始停止状态和超时界均保留。','',
        '| 组 | 有效运行 | 平均 PDI | 120 秒内证明 | PAR-2 / 秒 | 每次平均墙钟 / 秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in group_table:text.append('| {} | {}/{} | {:.3f} | {}/{} | {:.3f} | {:.3f} |'.format(r['group'],r['valid'],r['runs'],r['pdi'],r['proof_count'],r['valid'],r['par2'],r['worker_wall_seconds']))
    text+=['','![最终PDI](./stage5/final_pdi.png)','','联合组的配对差异（95% 区间为仅五实例的探索性 bootstrap，不是全分布保证）：','',
        '| 对照 | PDI 变化 | 改善实例 | 改善训练种子 | 实例 bootstrap 的 PDI 差值区间 |',
        '|---|---:|---:|---:|---:|']
    for r in comparisons:
        lo,hi=r['bootstrap_instance_95ci_absolute'];sw='—' if r['training_seed_wins'] is None else str(r['training_seed_wins'])+'/3'
        text.append('| {} | {:+.2f}% | {}/5 | {} | [{:.3f}, {:.3f}] |'.format(r['comparison'],r['pdi_change_percent'],r['instance_wins'],sw,lo,hi))
    text+=['','同时保留 SCIP 超过 120 秒时 PDI 的测量上下界；它与实例抽样区间是两类不确定性。不能根据落在测量区间或种子波动内的微小均值差认定改进。[逐实例表](./stage5/per_instance.csv)、[逐训练种子表](./stage5/per_training_seed.csv)、[配对差异及测量界](./stage5/paired_comparisons.json)。','',
        '## 实际训练量与预算','',
        '每次上限为 50 个任务 / 2 CPU 核时，按完整 PPO 更新提交模型。下表“进入完整更新的任务”包括零决策任务；“尝试任务”也包括已采样但更新未完成的任务。达到预算的半轮参数不参与评估，其 CPU 消耗仍计入上限。','',
        '| 组/种子 | 尝试任务 | 进入完整更新的任务 | 完整 PPO 轮 | 逻辑优化步 | 训练决策 | 总 CPU 核时 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in training:text.append('| {}/{} | {} | {} | {} | {} | {} | {:.3f} |'.format(r['group'],r['training_seed'],r['episodes_attempted'],r['checkpoint_episodes'],r['complete_ppo_iterations'],r['complete_optimizer_steps'],r['trained_transitions'],r['total_cpu_seconds']/3600))
    text+=['','九次训练合计 {:.3f} CPU 核时，桥接 {:.3f} 核时，225 次评估的工作进程合计 {:.3f} 核时；报告审计与调度另属既定工程余量。没有使用 GPU 在线训练。'.format(summary['training_cpu_seconds']/3600,summary['bridge_cpu_seconds']/3600,summary['evaluation_worker_cpu_seconds']/3600),'',
        '0.5/1/2 核时是预算上限，每点使用上限内最后一个完整更新。实际有效检查点可能远早于上限，多个点也可能复用同一模型；因此这是共同预算上限比较，不能写成三组完成了完全相同的训练量。早期点只用求解器种子 0，下面曲线的最终点也只取种子 0，以保持可比；上面的主表仍使用全部三个种子。','',
        '![预算曲线](./stage5/budget_curve.png)','','[实际训练明细](./stage5/training_summary.csv)、[各预算所选检查点及实际消耗](./stage5/selected_checkpoints.csv)、[相同种子的预算曲线数据](./stage5/budget_curve_seed0.csv)。','',
        '## 检查、失败与结论边界','',
        '46 项自动检查、40 项真实桥接审计及 {} 项最终审计通过。九次模型的两套优化器、固定原单位解码、完整 Critic 和检查点均可恢复。部署 Critic 耗时合计为 0；扩展搜索信息未在评估提取。完整回调、Actor、测量事件开销，原始 gap/界/有限标志及全部状态见[逐次运行表](./stage5/all_solver_runs.csv)。失败计数为 {}，未静默剔除异常或把无穷 gap 改成零。'.format(len(audit['checks']),sum(not r.get('valid') for r in results)),'',
        '这轮开发集只有五个实例，且之前已经用于探索。最终 checkpoint 未按开发结果挑选，仍不能把开发诊断当成未见测试集验证。训练数据覆盖与完整更新量也有限；未超过原生求解器、组间差异不稳定或收益较小，都不能证明大规模训练后永远无效。反之，小样本正结果也不能代替论文规模复现。','',
        '[审计](./stage5/verification.json)、[汇总JSON](./stage5/summary.json)、[冻结配置](./stage5/execution_config.json)、[实现与复运行说明](../tgppo/experiments/stage5/STAGE5.md)、[代码补丁](./stage5/stage5.patch)。旧阶段 4、校准模型及原版仓库的保留检查已通过。','',
        '本阶段不扩大训练、不增加方法。下一步建议验收本阶段后进入阶段 6，整理周五汇报，将复现程度、共同校准和待验证的信息增量分开表述。**现停在阶段 5，等待用户验收。**','']
    REPORT.write_text('\n'.join(text))
    completion=dict(stage=5,status='delivered_pending_acceptance',report=str(REPORT),stage6_started=False,
        primary_finding=conclusion,all_required_experiments_complete=True,verification_passed=True)
    write(RUN/'stage5_completion.json',completion)
    p=ROOT/'docs/stage1/protocol_draft.json';d=read(p);d['stage5_progress']='delivered_pending_acceptance';d['stage5_delivery']=completion;write(p,d)
    p=ROOT/'docs/阶段交付与验收_TGPPO复现与增量改进.md';s=p.read_text();s=s.replace('| 5 在线对照 | 进行中：先完成三组真实 PPO 桥接，再开展冻结主矩阵 | 2026-10-21 用户已授权进入；完成后单独验收 |','| 5 在线对照 | 已交付[验收报告](./阶段5_在线PPO对照与验收报告_TGPPO_2026-10-21.md)：9 次训练、135 次最终评估、90 次早期评估 | 等待用户验收；阶段 6 未开始 |');p.write_text(s)
    record=ROOT/'docs/阶段5_在线PPO执行记录_TGPPO_2026-10-21.md'
    if record.exists():
        r=record.read_text().replace('## 正在执行的正式对照','## 已完成的正式对照')
        r=r.replace('**目前没有完整主矩阵结果，不能判断信息增量改善了在线求解。**',
            '**阶段 5 已完成，结果与结论见[验收报告](./阶段5_在线PPO对照与验收报告_TGPPO_2026-10-21.md)；等待用户验收，阶段 6 未开始。**')
        record.write_text(r)
    print(json.dumps(completion,ensure_ascii=False))

if __name__=='__main__':main()
