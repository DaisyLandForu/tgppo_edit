"""Common-host development tables; no statistical or held-out test claims."""
import csv
from pathlib import Path
import numpy as np
from project.utils.protocol import write_json


def dump_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(directory,models,native_rows,episode,seeds,info):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=[];summary=[]
    for model in models:
        rows=model['rows']
        for r in rows:
            raw.append(dict(variant=model['variant'],training_seed=model['seed'],instance=r['instance_name'],
                solver_seed=r['solver_seed'],pdi=r['pdi_horizon_midpoint'],nodes=r['nnodes'],
                proof=bool(r['proof_completed'] and r['scip_solve_time']<=120.),scip_seconds=r['scip_solve_time'],
                optimize_wall_seconds=r['optimize_wall_seconds'],actor_calls=r.get('branch_count',0),
                actor_seconds=r.get('callback_seconds',{}).get('actor',0.)))
        calls=sum(r.get('branch_count',0) for r in rows)
        bc=model.get('bc')
        summary.append(dict(variant=model['variant'],training_seed=model['seed'],episode=episode,
            mean_pdi=float(np.mean([r['pdi_horizon_midpoint'] for r in rows])),
            proof_rate=float(np.mean([r['proof_completed'] and r['scip_solve_time']<=120. for r in rows])),
            mean_nodes=float(np.mean([r['nnodes'] for r in rows])),
            mean_optimize_wall=float(np.mean([r['optimize_wall_seconds'] for r in rows])),
            actor_ms_per_call=(1000*sum(r.get('callback_seconds',{}).get('actor',0.) for r in rows)/calls) if calls else None,
            actor_calls=calls,complete_ppo_updates=model['updates'],
            bc_completed_steps=bc['completed_steps'] if bc else 0,
            bc_selected_step=bc['selected_step'] if bc else None,
            bc_seconds=bc['seconds'] if bc else 0.,
            bc_validation_normalized_nll=bc['final']['validation']['macro']['normalized_nll'] if bc else None,
            bc_validation_accuracy=bc['final']['validation']['macro']['accuracy'] if bc else None,
            pending_episodes=model['progress']['next_episode']-model['progress']['trained_episodes']))
    lookup={(r['variant'],r['training_seed'],r['instance'],r['solver_seed']):r for r in raw}
    paired=[]
    for row in raw:
        variant=row['variant'];bc=variant.endswith('_bc');kind=variant[:-3] if bc else variant
        comparisons=[]
        if kind!='b0':comparisons.append(('structure_at_same_pretraining','b0_bc' if bc else 'b0'))
        if bc:comparisons.append(('bc_within_structure',kind))
        for effect,reference in comparisons:
            base=lookup[(reference,row['training_seed'],row['instance'],row['solver_seed'])]
            paired.append(dict(effect=effect,variant=variant,reference=reference,training_seed=row['training_seed'],
                instance=row['instance'],solver_seed=row['solver_seed'],pdi_delta=row['pdi']-base['pdi'],
                nodes_delta=row['nodes']-base['nodes'],wall_delta=row['optimize_wall_seconds']-base['optimize_wall_seconds'],
                both_actor_active=row['actor_calls']>0 and base['actor_calls']>0,
                method_actor_calls=row['actor_calls'],reference_actor_calls=base['actor_calls']))
    native=dict(mean_pdi=float(np.mean([r['pdi_horizon_midpoint'] for r in native_rows])),
        proof_rate=float(np.mean([r['proof_completed'] and r['scip_solve_time']<=120. for r in native_rows])))
    dump_csv(directory/'tasks.csv',raw);dump_csv(directory/'summary.csv',summary);dump_csv(directory/'paired.csv',paired)
    write_json(directory/'summary.json',dict(episode=episode,seeds=list(seeds),native=native,methods=summary,stack=info['stack']))
    lines=['# 三种轻量 Actor：统一开发集评估','',
        '固定 episode{}；训练种子{}。所有部署评估在同一台CPU运行；不是独立测试集结果。'.format(episode,','.join(map(str,seeds))),
        '训练CPU/GPU允许异构，其环境与归一化来源见各jobs/contexts记录；共同评估不能消除训练硬件混杂。','',
        '原生 relpscost：平均PDI {:.3f}，证明率 {:.1%}。'.format(native['mean_pdi'],native['proof_rate']),'',
        '| 组别 | 训练种子 | PDI↓ | 证明率↑ | Actor ms/调用↓ | PPO更新轮 | BC选中步/完成步 | BC耗时秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| {} | {} | {:.3f} | {:.1%} | {} | {} | {} / {} | {:.1f} |'.format(r['variant'],r['training_seed'],
            r['mean_pdi'],r['proof_rate'],'{:.3f}'.format(r['actor_ms_per_call']) if r['actor_ms_per_call'] is not None else 'N/A',
            r['complete_ppo_updates'],r['bc_selected_step'] if r['bc_selected_step'] is not None else '-',r['bc_completed_steps'],r['bc_seconds']))
    lines+=['','paired.csv 分开提供同预训练的结构差，以及同结构的BC差。完整主表保留未调用Actor实例；both_actor_active列用于排除无Actor计时差的神经决策归因。',
        '实例是比较单位，solver种子是实例内重复；1/2个训练种子均不足以宣称广泛稳定收益。BC额外数据与更新成本须单列。',
        '本阶段不启动66实例测试、CL、历史或树信用分配。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
