"""Analyze the conditional frozen2x2 cross-instance run without starting jobs."""
import json
from pathlib import Path
import statistics as st
from decide import read,sha,panel,csv_write

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'tgppo/runs/light-actor-20261105/cross-instance-8'


def main():
    status=read(RUN/'status.json');manifest=read(RUN/'manifest.json');decision=read(HERE/'decision.json')
    assert manifest['decision_sha256']==sha(HERE/'decision.json')
    rows=[];missing=[];records={};native=[]
    for task in manifest['tasks']:
        p=RUN/'raw'/(task['job']['job_id']+'.json')
        if not p.exists():missing.append(dict(job_id=task['job']['job_id'],status='missing'));continue
        r=read(p)
        if not r.get('valid'):missing.append(dict(job_id=r['job_id'],status=r.get('status')));continue
        assert r['worker_cpu_complete'] and not r['cuda_initialized'] and r['sampling_seed']==task['job']['sampling_seed']
        if task['variant']=='native':native.append(r)
        else:
            assert r['callback_seconds']['critic']==0 and r['fallback_action_count']==0
            records.setdefault((task['variant'],task['training_seed']),[]).append(r)
        rows.append(dict(variant=task['variant'],training_seed=task['training_seed'],instance=r['instance_name'],
            solver_seed=r['solver_seed'],pdi=r['pdi_horizon_midpoint'],nodes=r['nnodes'],
            proof=bool(r['proof_completed'] and r['scip_solve_time']<=120),wall=r['optimize_wall_seconds'],
            actor_calls=r.get('branch_count',0),actor_seconds=r.get('callback_seconds',{}).get('actor',0)))
    csv_write('cross_tasks.csv',rows);csv_write('cross_missing.csv',missing)
    if missing or status['status']!='complete':
        (HERE/'CROSS_INSTANCE_REPORT.md').write_text('# 跨实例确认未完整\n\n有效{}/144项，状态{}；缺失/失败保留在cross_missing.csv，不用子集冒充完整结果，不启动追加训练或求解。\n'.format(len(rows),status['status']))
        return
    assert len(rows)==144 and len(native)==16
    for m in manifest['models'].values():
        assert sha(m['actor'])==sha(m['frozen_actor'])==m['actor_sha256'] and sha(m['checkpoint'])==m['checkpoint_sha256']
    names=[x['instance'] for x in manifest['selected']]
    active=[n for n in names if all(r['branch_count']>0 for rr in records.values() for r in rr if r['instance_name']==n)]
    inactive=[n for n in names if all(r['branch_count']==0 for rr in records.values() for r in rr if r['instance_name']==n)]
    panels=[];instances=[];lookup={}
    for key,rs in records.items():
        assert len(rs)==16
        for tag,subset in [('all',names),('common_actor',active)]:
            if subset:
                values=panel([r for r in rs if r['instance_name'] in subset]);lookup[key+(tag,)]=values
                panels.append(dict(variant=key[0],training_seed=key[1],panel=tag,**values))
        for name in names:instances.append(dict(variant=key[0],training_seed=key[1],instance=name,
            **panel([r for r in rs if r['instance_name']==name])))
    csv_write('cross_panels.csv',panels);csv_write('cross_instances.csv',instances)
    variants=manifest['variants'];kind=decision['chosen']['kind'];candidate=decision['chosen']['variant'];nat=panel(native)
    lines=['# 冻结轻量Actor的8实例跨实例探索复核', '',
        '144/144项完成：128模型+16项本轮同机新原生。两个训练种子、两个solver种子、固定episode100；无训练/GPU/权重更新。', '',
        'CPU：{}。经过墙时{:.1f}秒，worker墙时合计{:.1f}秒；没有混合旧Gold6148绝对计时。'.format(manifest['stack']['cpu_model'],status['charged_wall_seconds'],status.get('summed_worker_wall_seconds',0)), '',
        '实例在本轮方法之前已按原生成本分层冻结，过去研究查看过；因此不是全新独立保留集。8实例分层配额也不等于66实例总体分布。', '',
        '原生PDI {:.3f}，证明率 {:.1%}，平均optimize墙时 {:.3f}秒。'.format(nat['pdi'],nat['proof_rate'],nat['wall']), '',
        '| 条件 | PDI s11 | PDI s22 | 平均PDI | 对原生变化 | 证明率 | optimize秒 | Actor ms/调用 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    deployment=[]
    for v in variants:
        a,b=lookup[(v,11,'all')],lookup[(v,22,'all')];m=panel(records[(v,11)]+records[(v,22)])
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:+.2%} | {:.1%} | {:.3f} | {:.3f} |'.format(v,a['pdi'],b['pdi'],m['pdi'],m['pdi']/nat['pdi']-1,m['proof_rate'],m['wall'],m['actor_ms']))
        for seed in (11,22):
            rs=records[(v,seed)];calls=sum(r['branch_count'] for r in rs)
            timing={k:sum(r['callback_seconds'].get(k,0) for r in rs) for k in ('actor','features','callback','critic')}
            deployment.append(dict(variant=v,training_seed=seed,actor_calls=calls,
                actor_ms_per_call=1000*timing['actor']/calls if calls else None,
                feature_ms_per_call=1000*timing['features']/calls if calls else None,
                callback_ms_per_call=1000*timing['callback']/calls if calls else None,
                critic_seconds=timing['critic'],mean_optimize_wall=panel(rs)['wall']))
    csv_write('cross_deployment.csv',deployment)
    paired=[];decomposition=[]
    lines += ['', '## 增量及跨种子方向', '', '| 比较 | 全体PDI变化 s11 / s22 | 共同Actor实例变化 s11 / s22 |','|---|---|---|']
    comparisons=list(dict.fromkeys([(kind,'b0'),(kind+'_bc','b0_bc'),(kind+'_bc',kind),(candidate,'b0')]))
    for v,ref in comparisons:
        allvalues=[];activevalues=[]
        for seed in (11,22):
            for tag,bucket in [('all',allvalues),('common_actor',activevalues)]:
                if (v,seed,tag) not in lookup:continue
                a,b=lookup[(v,seed,tag)],lookup[(ref,seed,tag)];delta=100*(a['pdi']/b['pdi']-1);bucket.append(delta)
                paired.append(dict(variant=v,reference=ref,training_seed=seed,panel=tag,pdi_percent=delta,
                    pdi_delta=a['pdi']-b['pdi'],proof_delta=a['proof_rate']-b['proof_rate'],
                    measurement_delta_lower=a['lower']-b['upper'],measurement_delta_upper=a['upper']-b['lower']))
            total=lookup[(v,seed,'all')]['pdi']-lookup[(ref,seed,'all')]['pdi']
            contribution=lambda subset:sum(
                panel([r for r in records[(v,seed)] if r['instance_name']==n])['pdi']-
                panel([r for r in records[(ref,seed)] if r['instance_name']==n])['pdi']
                for n in subset)/len(names)
            act,non=contribution(active),contribution(inactive)
            decomposition.append(dict(variant=v,reference=ref,training_seed=seed,full_pdi_delta=total,
                common_actor_contribution=act,no_actor_contribution=non,other_instance_contribution=total-act-non))
        fmt=lambda xx:' / '.join('{:+.2f}%'.format(x) for x in xx) if xx else 'N/A'
        lines.append('| {} vs {} | {} | {} |'.format(v,ref,fmt(allvalues),fmt(activevalues)))
    csv_write('cross_factor_effects.csv',paired)
    csv_write('cross_no_actor_decomposition.csv',decomposition)
    lines += ['', '无Actor实例：{}；共同Actor实例：{}。完整主表始终保留所有实例，不用无Actor计时差解释神经策略收益。'.format('、'.join(inactive) or '无','、'.join(active) or '无'), '',
        '## 逐实例正反例', '', '| 实例 | 原生PDI | B0 | {} | {}+BC |'.format(kind,kind),'|---|---:|---:|---:|---:|']
    for name in names:
        values=[panel([r for r in native if r['instance_name']==name])['pdi']]
        for v in ('b0',kind,kind+'_bc'):values.append(panel([r for seed in (11,22) for r in records[(v,seed)] if r['instance_name']==name])['pdi'])
        lines.append('| {} | {} |'.format(name,' | '.join('{:.3f}'.format(x) for x in values)))
    lines += ['', '只报告冻结候选的完整结果，不根据新结果换模型、实例或继续调参；两个训练种子、8个探索实例不能宣称显著或普遍泛化。训练CUDA流边界仍存在。', '',
        '本阶段完成后停止，等待用户验收，不自动训练或扩大66实例。']
    (HERE/'CROSS_INSTANCE_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
