"""Read-only analysis of the authorized20-task ablation; never starts solves."""
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'tgppo/runs/light-actor-20261105/bc-only-ablation-20'
sys.path.insert(0,str(HERE.parent/'continuation'))
from decide import panel

def read(path):return json.loads(Path(path).read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csv_save(name,rows):
    if rows:
        with (HERE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    state=read(OUT/'status.json');manifest=read(OUT/'manifest.json');records={};native=[];tasks=[];missing=[]
    assert sha(HERE/'run.py')==manifest['runner_sha256'] and sha(HERE/'PLAN.md')==manifest['plan_sha256']
    for task in manifest['tasks']:
        path=OUT/'raw'/(task['job']['job_id']+'.json')
        if not path.exists():missing.append(dict(job_id=task['job']['job_id'],status='missing'));continue
        row=read(path)
        if not row.get('valid'):missing.append(dict(job_id=row['job_id'],status=row.get('status')));continue
        assert row['worker_cpu_complete'] and not row['cuda_initialized'] and row['sampling_seed']==task['job']['sampling_seed']
        if task['condition']=='native':native.append(row)
        else:
            assert row['callback_seconds']['critic']==0 and row['fallback_action_count']==0
            records.setdefault((task['condition'],task['training_seed']),[]).append(row)
        tasks.append(dict(condition=task['condition'],training_seed=task['training_seed'],instance=row['instance_name'],
            solver_seed=row['solver_seed'],pdi=row['pdi_horizon_midpoint'],lower=row['pdi_horizon_lower'],upper=row['pdi_horizon_upper'],
            proof=bool(row['proof_completed'] and row['scip_solve_time']<=120),nodes=row['nnodes'],wall=row['optimize_wall_seconds'],
            actor_calls=row.get('branch_count',0),actor_seconds=row.get('callback_seconds',{}).get('actor',0)))
    csv_save('tasks.csv',tasks);csv_save('missing.csv',missing)
    if missing or state['status']!='complete':
        (HERE/'REPORT.md').write_text('# BC-only小消融未完整\n\n有效{}/20项，状态{}；保留原始记录，不以子集代替完整对照，不自动重试。\n'.format(len(tasks),state['status']));return
    assert len(tasks)==20 and len(native)==4 and len(records)==4
    for model in manifest['models'].values():
        assert sha(model['source'])==model['source_sha256'] and sha(model['frozen_actor'])==model['actor_sha256']
    summary=[];pairs=[];instance_rows=[]
    names=[item['instance'] for item in manifest['selected']]
    for condition in ('bc_only','bc_ppo'):
        for seed in (11,22):
            assert len(records[(condition,seed)])==4
            summary.append(dict(condition=condition,training_seed=seed,**panel(records[(condition,seed)])))
    for name in names:
        for seed in (11,22):
            before=panel([r for r in records[('bc_only',seed)] if r['instance_name']==name])
            after=panel([r for r in records[('bc_ppo',seed)] if r['instance_name']==name])
            direction='ppo_better' if after['upper']<before['lower'] else ('ppo_worse' if after['lower']>before['upper'] else 'measurement_intervals_overlap')
            pairs.append(dict(instance=name,training_seed=seed,bc_only_pdi=before['pdi'],bc_ppo_pdi=after['pdi'],
                pdi_percent=100*(after['pdi']/before['pdi']-1),pdi_delta=after['pdi']-before['pdi'],
                measurement_delta_lower=after['lower']-before['upper'],measurement_delta_upper=after['upper']-before['lower'],
                proof_delta=after['proof_rate']-before['proof_rate'],wall_delta=after['wall']-before['wall'],direction=direction))
    means={condition:panel([r for seed in (11,22) for r in records[(condition,seed)]]) for condition in ('bc_only','bc_ppo')}
    means['native']=panel(native)
    all_better=all(pair['direction']=='ppo_better' and pair['proof_delta']>=0 for pair in pairs)
    all_worse=all(pair['direction']=='ppo_worse' and pair['proof_delta']<=0 for pair in pairs)
    verdict='consistent_improvement_on_both_selected_cases' if all_better else ('consistent_regression_on_both_selected_cases' if all_worse else 'mixed_or_inconclusive_on_selected_cases')
    result=dict(status='complete',verdict=verdict,means=means,pairs=pairs,instances=names,
        improvement_pairs=sum(p['direction']=='ppo_better' for p in pairs),regression_pairs=sum(p['direction']=='ppo_worse' for p in pairs),
        charged_wall_seconds=state['charged_wall_seconds'],budget_seconds=1200,training_steps=0,gpu=False,
        limitation='Two previously inspected cases, selected from earlier results; not a generalization estimate or a new selection set',
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (OUT/'manifest.json',OUT/'status.json',OUT/'evaluation_rows.json',HERE/'run.py',HERE/'PLAN.md',HERE/'analyze.py')})
    csv_save('summary.csv',summary);csv_save('paired.csv',pairs);save('analysis.json',result)
    lines=['# BC-only 与 BC＋PPO：20项本地消融结果','',
        '20/20有效完成：16模型＋4同机新原生；CPU{}，执行{:.2f}/1200秒，worker墙时合计{:.2f}秒。无训练/GPU/自动重试，源权重不变。'.format(manifest['stack']['cpu_model'],state['charged_wall_seconds'],state['summed_worker_wall_seconds']),'',
        'BC-only是正式PPO实际采用的BC第100步权重（原BC预算1000步）；BC＋PPO从同一权重出发完成100episode。BC格式包装逐张量等值，指纹与训练初始化相同，未另选BC权重。','',
        '## E1：真实求解结果','',
        '| 条件 | PDI↓ | 证明率↑ | optimize秒↓ | Actor ms/调用↓ |','|---|---:|---:|---:|---:|']
    for condition in ('native','bc_only','bc_ppo'):
        m=means[condition];lines.append('| {} | {:.3f} | {:.1%} | {:.3f} | {} |'.format(condition,m['pdi'],m['proof_rate'],m['wall'],'{:.3f}'.format(m['actor_ms']) if m['actor_ms'] is not None else '—'))
    lines += ['', 'PDI为固定120秒百分比gap积分；学习组均值含两训练种子，原生没有训练种子。实例为比较单位，solver种子为内部重复。','',
        '| 实例 | 训练种子 | BC-only PDI | BC＋PPO PDI | PPO后变化 | 证明率变化 | 测量方向 |','|---|---:|---:|---:|---:|---:|---|']
    for pair in pairs:lines.append('| {} | {} | {:.3f} | {:.3f} | {:+.2f}% | {:+.1f} pp | {} |'.format(pair['instance'],pair['training_seed'],pair['bc_only_pdi'],pair['bc_ppo_pdi'],pair['pdi_percent'],100*pair['proof_delta'],pair['direction']))
    for name in names:
        values={condition:panel([r for seed in (11,22) for r in records[(condition,seed)] if r['instance_name']==name]) for condition in ('bc_only','bc_ppo')}
        for condition,value in values.items():instance_rows.append(dict(instance=name,condition=condition,**value))
    csv_save('instances.csv',instance_rows)
    lines += ['', '## 结论及边界','',
        '预定口径结论：**{}**。两实例合并PDI变化{:+.2f}%，不以这个合并均值代替逐实例/逐种子结果。'.format(verdict,100*(means['bc_ppo']['pdi']/means['bc_only']['pdi']-1)),'',
        '4个实例×训练种子配对中，PPO改善{}个、退步{}个。当前不能把“BC对PPO有帮助”升级为“PPO对BC有稳定附加收益”；也不足以断言PPO在所有实例上无效。'.format(result['improvement_pairs'],result['regression_pairs']),'',
        '这两个对象来自此前已查看的跨实例结果：neos-1215259为共同正例，neos-512201为BC相对无BC反例。因此这是PPO附加价值的局部消融，不是新的独立测试、泛化估计或更长训练的收敛证据。测量上下界不是统计置信区间。','',
        'E2：网络/特征/部署worker/SCIP设置不改，BC初始化指纹和源SHA前后验证；Critic部署调用为0。PPO期间未加BC/CL辅助损失。','',
        'E3：更长PPO能否改善仍需预定开发里程碑验证；本轮没有启动长训练或全66评测。当前先利用已有BC/episode20/episode100在5开发实例补学习曲线更省成本，不直接扩大500episode。局部负结果不能单独证明H3、信用分配或遗忘是原因。','',
        '本阶段完成后停止，等待验收。训练与评测流程/时间建议另见TRAINING_AND_EVALUATION.md。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
