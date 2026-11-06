"""Evaluate the written advancement rule from committed development records only."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT/'tgppo/runs/light-actor-20261105'
RESULTS = RUN/'results'
VARIANTS = ('b0','b0_bc','mlp','mlp_bc','set','set_bc','isab','isab_bc')
PARAMS = dict(mlp=24493,set=32749,isab=92205)
SEEDS = (11,22)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def save(name,value):
    p=HERE/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def csv_write(name,rows):
    if rows:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def panel(rows):
    calls=sum(r.get('branch_count',0) for r in rows)
    return dict(pdi=st.mean(r['pdi_horizon_midpoint'] for r in rows),
        lower=st.mean(r['pdi_horizon_lower'] for r in rows),upper=st.mean(r['pdi_horizon_upper'] for r in rows),
        proof_rate=st.mean(bool(r['proof_completed'] and r['scip_solve_time']<=120) for r in rows),
        nodes=st.mean(r['nnodes'] for r in rows),wall=st.mean(r['optimize_wall_seconds'] for r in rows),
        actor_calls=calls,actor_ms=(1000*sum(r.get('callback_seconds',{}).get('actor',0) for r in rows)/calls) if calls else None,
        feature_seconds=sum(r.get('callback_seconds',{}).get('features',0) for r in rows))


def decide():
    done_path=RESULTS/'complete.json'
    if not done_path.exists():
        save('decision.json',dict(status='waiting_for_existing_evaluation',cross_instance_allowed=False,
            rule_sha256=sha(HERE/'PLAN.md'),extra_solver_tasks=0));return None
    done=read(done_path)
    fresh_native=done.get('native_reference','').startswith('fresh')
    assert done['complete'] and done['episode']==100 and done['seeds']==[11,22] and done['model_tasks']==160
    cohort=Path(done['cohort']);native=read(cohort/'native.json')
    models=read(cohort/'models-seeds-11-22.json')
    expected_hashes=read(ROOT/'docs/stage1/protocol_draft.json')['data']['hashes']
    for row in native:
        assert row['identity']['instance_sha256']==expected_hashes[row['instance_name']]
    assert len(models)==16
    records={};sources={str(done_path.relative_to(ROOT)):sha(done_path),
        str((cohort/'models-seeds-11-22.json').relative_to(ROOT)):sha(cohort/'models-seeds-11-22.json'),
        str((cohort/'native.json').relative_to(ROOT)):sha(cohort/'native.json')}
    for model in models:
        v,s=model['variant'],model['seed'];rows=model['rows']
        assert v in VARIANTS and s in SEEDS and len(rows)==10
        assert sha(model['checkpoint'])==model['checkpoint_sha256']
        actor=RUN/v/'training'/('B0-s'+str(s))/'actor-0100.pt'
        assert sha(actor)==model['actor_sha256']
        assert all(r['valid'] and r['worker_cpu_complete'] and not r['cuda_initialized'] and
            r['callback_seconds']['critic']==0 and r['fallback_action_count']==0 for r in rows)
        for r in rows:
            canonical=r.get('canonical_sampling_job_id',r['job_id'])
            assert r['sampling_seed']==int(hashlib.sha256((str(s)+':'+canonical).encode()).hexdigest()[:8],16)
        records[(v,s)]=rows
    names=sorted({r['instance_name'] for r in native})
    expected={(n,s) for n in names for s in (0,1)}
    assert len(names)==5 and len(native)==10
    assert all({(r['instance_name'],r['solver_seed']) for r in rr}==expected for rr in records.values())
    active=[n for n in names if all(r['branch_count']>0 for rr in records.values() for r in rr if r['instance_name']==n)]
    inactive=[n for n in names if all(r['branch_count']==0 for rr in records.values() for r in rr if r['instance_name']==n)]
    panels=[];per_instance=[];by_panel={}
    for (v,s),rows in records.items():
        for tag,subset in [('all',names),('common_actor',active)]:
            if not subset:continue
            values=panel([r for r in rows if r['instance_name'] in subset]);by_panel[(v,s,tag)]=values
            panels.append(dict(variant=v,training_seed=s,panel=tag,**values))
        for n in names:
            per_instance.append(dict(variant=v,training_seed=s,instance=n,actor_participates=n in active,
                **panel([r for r in rows if r['instance_name']==n])))
    for n in names:
        per_instance.append(dict(variant='native',training_seed=None,instance=n,actor_participates=False,
            **panel([r for r in native if r['instance_name']==n])))
    differences=[];decisions=[]
    for v in VARIANTS[2:]:
        kind=v[:-3] if v.endswith('_bc') else v
        refs=('b0','b0_bc',kind) if v.endswith('_bc') else ('b0',)
        reasons=[]
        if len(active)<2:reasons.append('fewer_than_two_common_actor_instances')
        for s in SEEDS:
            for ref in refs:
                for tag in ('all','common_actor'):
                    if (v,s,tag) not in by_panel:continue
                    a,b=by_panel[(v,s,tag)],by_panel[(ref,s,tag)]
                    pdi_pass=a['upper']<b['lower'];proof_pass=a['proof_rate']>=b['proof_rate']
                    differences.append(dict(variant=v,reference=ref,training_seed=s,panel=tag,
                        pdi_delta=a['pdi']-b['pdi'],pdi_percent=100*(a['pdi']/b['pdi']-1),
                        measurement_delta_lower=a['lower']-b['upper'],measurement_delta_upper=a['upper']-b['lower'],
                        proof_delta=a['proof_rate']-b['proof_rate'],pdi_pass=pdi_pass,proof_pass=proof_pass))
                    if not pdi_pass:reasons.append('{}:{}:{}:PDI'.format(s,ref,tag))
                    if not proof_pass:reasons.append('{}:{}:{}:proof'.format(s,ref,tag))
            a,b=by_panel[(v,s,'all')],by_panel[('b0',s,'all')]
            if a['actor_ms'] is None or b['actor_ms'] is None or a['actor_ms']>.5*b['actor_ms']:
                reasons.append('{}:deployment_cost'.format(s))
        wins=[]
        for n in active:
            if all(panel([r for r in records[(v,s)] if r['instance_name']==n])['upper'] <
                   panel([r for r in records[(ref,s)] if r['instance_name']==n])['lower']
                   for s in SEEDS for ref in refs):wins.append(n)
        if len(wins)<2:reasons.append('fewer_than_two_joint_instance_wins')
        decisions.append(dict(variant=v,kind=kind,passed=not reasons,references=list(refs),
            common_instance_wins=wins,failed_conditions=reasons,
            mean_pdi=st.mean(by_panel[(v,s,'all')]['pdi'] for s in SEEDS),parameters=PARAMS[kind]))
    eligible=[d for d in decisions if d['passed']]
    chosen=min(eligible,key=lambda d:(d['mean_pdi'],d['parameters'],d['variant'])) if eligible else None
    native_panel=panel(native)
    deployment=[]
    for (v,s),rs in records.items():
        calls=sum(r['branch_count'] for r in rs);wall=sum(r['optimize_wall_seconds'] for r in rs)
        timing={k:sum(r['callback_seconds'].get(k,0) for r in rs) for k in ('actor','features','callback','critic')}
        deployment.append(dict(variant=v,training_seed=s,actor_calls=calls,
            actor_ms_per_call=1000*timing['actor']/calls if calls else None,
            feature_ms_per_actor_call=1000*timing['features']/calls if calls else None,
            actor_fraction_of_optimize_wall=timing['actor']/wall,
            callback_fraction_of_optimize_wall=timing['callback']/wall,
            critic_seconds=timing['critic'],mean_optimize_wall=wall/len(rs),
            mean_worker_wall=st.mean(r['worker_wall_seconds'] for r in rs)))
    csv_write('deployment.csv',deployment)
    control=[]
    bclook={(r['instance_name'],r['solver_seed']):r for r in records[('b0_bc',11)]}
    for r in records[('b0',11)]:
        q=bclook[(r['instance_name'],r['solver_seed'])]
        trace=lambda row:[(e.get('primal'),e.get('dual'),e.get('reason')) for e in row['bounds_trace']]
        control.append(dict(instance=r['instance_name'],solver_seed=r['solver_seed'],
            b0_nodes=r['nnodes'],b0_bc_nodes=q['nnodes'],b0_calls=r['branch_count'],b0_bc_calls=q['branch_count'],
            both_proof=bool(r['proof_completed'] and q['proof_completed']),
            equal_observed_bound_sequence=trace(r)==trace(q),
            b0_pdi=r['pdi_horizon_midpoint'],b0_bc_pdi=q['pdi_horizon_midpoint'],
            b0_wall=r['optimize_wall_seconds'],b0_bc_wall=q['optimize_wall_seconds']))
    csv_write('timing_control.csv',control)
    result=dict(status='passed' if chosen else 'no_candidate_passed',
        evaluated_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        rule_sha256=sha(HERE/'PLAN.md'),rule_timing='written after some evaluations started; before inspecting light-model solver metrics',
        decision_script_sha256=sha(__file__),
        cross_instance_allowed=bool(chosen),chosen=chosen,candidates=decisions,common_actor_instances=active,
        results_directory=str(RESULTS.resolve()),
        no_actor_instances=inactive,native=native_panel,source_sha256=sources,
        native_reference=(done['native_reference'] if fresh_native else 'cached exact solver identity and Gold6148/software fingerprint; not a contemporaneous rerun'),
        training_randomness_limit='B0/light CUDA initial RNG differs; Critic weights and within-architecture BC RNG match')
    csv_write('development_panels.csv',panels);csv_write('development_instances.csv',per_instance)
    csv_write('factor_effects.csv',differences);save('decision.json',result)
    lines=['# 完整开发评估：结构、BC与原生参照', '',
        '160/160模型任务和10项{}原生参照齐全。固定100 episode、两个训练种子；仅5开发实例。'.format('本轮同机新测' if fresh_native else '缓存'), '',
        '结果源：{}。CPU：{}；旧Gold6148的108项有效记录另行保留，不混合绝对时间。'.format(str(RESULTS),done['stack']['cpu_model']), '',
        '## 方法主表', '',
        'PDI越低越好。原生relpscost PDI {:.3f}，证明率 {:.1%}；{}。'.format(native_panel['pdi'],native_panel['proof_rate'],'与全部模型在本轮同一主机新测' if fresh_native else '参照为匹配CPU型号/软件/求解身份的既有缓存，非同步重测'), '',
        '| 方法 | PDI s11 | PDI s22 | 两种子PDI均值 | 对原生变化 | 证明率 | Actor ms/调用 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        a,b=(by_panel[(v,s,'all')] for s in SEEDS);rs=records[(v,11)]+records[(v,22)];allvalues=panel(rs)
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:+.2%} | {:.1%} | {:.3f} |'.format(v,a['pdi'],b['pdi'],allvalues['pdi'],allvalues['pdi']/native_panel['pdi']-1,allvalues['proof_rate'],allvalues['actor_ms']))
    lines += ['', '## 两种增量分开分析', '',
        '以下负数表示改善。结构：相同预训练下相对B0；BC：同结构有BC相对无BC。共同Actor实例为{}，无Actor实例为{}，不将后者计时差归因于神经评分。'.format('、'.join(active),'、'.join(inactive)), '',
        '| 比较 | 全体PDI变化 s11 / s22 | 共同Actor PDI变化 s11 / s22 |', '|---|---|---|']
    factor_rows=[];decomposition=[]
    for v in VARIANTS:
        kind=v[:-3] if v.endswith('_bc') else v
        refs=[]
        if kind!='b0':refs.append(('结构','b0_bc' if v.endswith('_bc') else 'b0'))
        if v.endswith('_bc'):refs.append(('BC',kind))
        for effect,ref in refs:
            all_delta=[];active_delta=[]
            for s in SEEDS:
                for tag,bucket in [('all',all_delta),('common_actor',active_delta)]:
                    a,b=by_panel[(v,s,tag)],by_panel[(ref,s,tag)]
                    bucket.append(100*(a['pdi']/b['pdi']-1))
                    factor_rows.append(dict(effect=effect,variant=v,reference=ref,training_seed=s,panel=tag,
                        pdi_percent=100*(a['pdi']/b['pdi']-1),proof_delta=a['proof_rate']-b['proof_rate']))
            fmt=lambda xs:' / '.join('{:+.2f}%'.format(x) for x in xs)
            lines.append('| {}：{} vs {} | {} | {} |'.format(effect,v,ref,fmt(all_delta),fmt(active_delta)))
            for s in SEEDS:
                total=by_panel[(v,s,'all')]['pdi']-by_panel[(ref,s,'all')]['pdi']
                active_contribution=(len(active)/len(names))*(by_panel[(v,s,'common_actor')]['pdi']-by_panel[(ref,s,'common_actor')]['pdi']) if active else 0
                inactive_contribution=sum(
                    panel([r for r in records[(v,s)] if r['instance_name']==n])['pdi']-
                    panel([r for r in records[(ref,s)] if r['instance_name']==n])['pdi']
                    for n in inactive)/len(names)
                decomposition.append(dict(effect=effect,variant=v,reference=ref,training_seed=s,
                    full_pdi_delta=total,common_actor_contribution=active_contribution,
                    no_actor_contribution=inactive_contribution,
                    other_instance_contribution=total-active_contribution-inactive_contribution))
    csv_write('independent_effects.csv',factor_rows)
    csv_write('no_actor_decomposition.csv',decomposition)
    lines += ['', '无Actor实例仍可能因根节点处理进度/墙时变化产生PDI差；不能将其归于Actor。'
        'no_actor_decomposition.csv将每项结构/BC均值差按共同Actor、无Actor及其余实例分解，三项相加为完整主表差值。']
    lines += ['', '## 逐开发实例：BC候选与原生/原Actor', '',
        '每格为PDI / 证明率。学习组平均两个训练种子；原生仅其两个solver重复，不当作额外训练种子。', '',
        '| 实例 | 原生 | B0 | MLP+BC | SET+BC | ISAB+BC |','|---|---:|---:|---:|---:|---:|']
    for n in names:
        values=[]
        for v in ('native','b0','mlp_bc','set_bc','isab_bc'):
            rs=[r for r in native if r['instance_name']==n] if v=='native' else [r for s in SEEDS for r in records[(v,s)] if r['instance_name']==n]
            m=panel(rs);values.append('{:.2f} / {:.0%}'.format(m['pdi'],m['proof_rate']))
        lines.append('| {} | {} |'.format(n,' | '.join(values)))
    lines += ['', '## 是否进入跨实例确认', '',
        '筛选结论：**{}**。'.format('通过，选择'+chosen['variant'] if chosen else '没有候选通过，不新增跨实例求解'), '',
        '| 候选 | 通过 | 跨两种子的共同正例 | 未通过条件 |','|---|---|---|---|']
    for d in decisions:
        lines.append('| {} | {} | {} | {} |'.format(d['variant'],d['passed'],'、'.join(d['common_instance_wins']) or '无','; '.join(d['failed_conditions']) or '无'))
    lines += ['', '条件在PLAN.md写明，完整数值在factor_effects.csv。推进门槛不是显著性检验；不因结果放宽条件。', '',
        '若没有候选通过，本轮停止，不自动扩大训练/CL或启动66实例。BC专家一致率或NLL有效不能替代求解收益。', '',
        '原B0-s11的BC初始化选0步，不能将其后两条件的求解差归因于BC。B0与轻量组CUDA初始流未完全对齐，架构归因需保留局限；同结构BC组CUDA初始流匹配。', '',
        '补充计时核对：timing_control.csv保留当前cohort的B0-s11与B0+BC-s11。旧Gold6148记录中，dcmulti/vpm2已证明任务的节点数、Actor调用数、观测到的界值序列一致，PDI仍有差异（例如dcmulti solver0约8.366与13.170）。不能将其差异归因于有效BC初始化，也不能仅凭这些摘要证明整棵搜索树完全相同；CPU负载/调度等具体原因未定位。这个旧例子不混入当前cohort均值。', '',
        '部署成本见deployment.csv：Actor、features及整个callback有嵌套关系，不能相加；Critic部署时间为0。ms/调用属于实际访问状态分布下的观察值，受控同状态CPU延迟参照此前离线探针。', '',
        '若候选通过，按PLAN.md冻结胜出结构的2×2权重，在独立CPU环境144项上限内确认，不修改本轮训练或旧结果。']
    (HERE/'DEVELOPMENT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=RESULTS)
    args=parser.parse_args();RESULTS=args.results.resolve();decide()
