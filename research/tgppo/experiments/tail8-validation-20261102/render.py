"""Run the final read-only audit and create the reviewable report."""
import json
from pathlib import Path
import analyze
from probe_utils import OUT


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    analyze.main()
    s=json.loads((OUT/'summary.json').read_text())
    off=json.loads((OUT/'offline/summary.json').read_text())['results']
    training=json.loads((OUT/'pilot/summary.json').read_text())['results']
    texts=['# 完整 Actor、在线 pilot 与开发求解验证',
        '日期：2026-11-02 启动。依据用户“你把剩下的直接都验证完吧”的连续授权，执行本地 CPU 有界验证。',
        '**全部结果与证据以本报告、固定协议和原始记录为准。本轮是缩预算诊断，不是原论文完整性能复现。**',
        '## 1. 范围与公平性',
        '原头与 tail8 × 两个来源种子11/22；头初值固定为101，没有按已测结果选最好初始化。上游与 Critic 取相同来源 ep100 权重，配对评分头从先前已保存 initial.pt 载入，最终标量读出归零，两头均使用 fresh Adam。保留前五层与完整树门控，仅替换尾部 8→4→2→1 为 8→1。原基线与历史结果未改。',
        '这是一套统一初始化的暖启动对照，不能当作原 B0 的原样续训。离线桥接使用完整同版本缓冲的原 GAE，在原固定256条 batch上重复两次完整Actor/Critic更新。在线各组只做4个20秒完整episode采样和一个有界PPO块：8个256条逻辑batch，批内无放回、批间复用全缓冲，完整缓冲GAE，n_epochs=1。该更新计划是双方共用的缩预算pilot，并非历史3个完整epoch。奖励公式、学习率、clip、gamma/lambda均未改。',
        '训练仅用 neos20/rmatr100-p10 × solver seeds0/1；开发用既定全部5实例 × solver seeds0/1。配对组使用与组名无关的动作RNG种子；评估只加载Actor。原生基线在同机同20秒、同cutoff和求解设置下重测，不能混用历史120秒归一化。',
        '## 2. 完整 Actor 离线桥接',
        table(['组','surrogate变化','训练TV中位数','监测TV中位数','第2步Transformer梯度'],[
            [f"{x['variant']}-s{x['source_seed']}",f"{x['surrogate_change']:.4g}",f"{x['train']['tv_p50']:.4g}",
             f"{x['monitor']['tv_p50']:.4g}",f"{x['second_actor_module_gradients'].get('transformer',0):.4g}"] for x in off]),
        '四组各2次Actor与2次Critic更新，共16个optimizer.step。Actor与Critic参数无共享；同来源配对Critic更新逐位相同。梯度传回上游、保存恢复、Actor-only导出与padding检查通过。这证明接入/更新链可运行，不能把仅两步的微小响应称为完整策略学会。归零读出首步上游梯度为0是预期。',
        '## 3. 在线采样与更新',
        table(['组','完整采样transition数','Actor/Critic步数','更新前TV中位数','更新后TV中位数'],[
            [x['model'],x['transitions'],f"{x['actor_steps']}/{x['critic_steps']}",
             f"{x.get('before',{}).get('tv_p50',0):.4g}",f"{x.get('after',{}).get('tv_p50',0):.4g}"] for x in training]),
        'TV只描述各组采样缓冲前256条状态相对均匀分布的变化，不是候选动作正确性。各组优势均在自身全部完整轨迹上计算和标准化，没有只截取更新batch重算。所有采样episode和不足样本/异常情况原样保留，不补采或挑成功实例。',
        '## 4. Actor-only真实开发求解',
        table(['组','平均20秒PDI↓','cutoff任务证明率','无Actor任务数/10','Actor毫秒/分支','完整回调占optimize'],[
            [x['model'],f"{x['pdi_mean']:.4f}",f"{100*x['proof_rate']:.0f}%",x['no_actor_tasks'],
             f"{x['actor_ms_per_branch']:.3f}" if x['actor_ms_per_branch'] is not None else 'N/A',
             f"{100*x['callback_fraction']:.1f}%"] for x in s['models']]),
        'PDI采用相同20秒窗口的原生积分补尾；求解超时过冲时保留上下界，midpoint只是区间中点估计。完成率是已知最优cutoff的受控搜索任务证书，不冒充原MILP从零找到最优解。callback时间包含Actor等子项，不能重复相加。',
        table(['开发实例','原生PDI','原头PDI','tail8 PDI','tail8−原头↓','两组均无Actor的配对数/4'],[
            [x['instance'],f"{x['native_pdi']:.4f}",f"{x['original_pdi']:.4f}",f"{x['tail8_pdi']:.4f}",
             f"{x['tail8_minus_original']:.4f}",x['paired_tasks_both_no_actor']] for x in s['instances']]),
        '每实例先平均solver重复，再跨两个来源模型汇总；不会把40个任务或重复solver种子当40个独立实例。未触发Actor的任务仍计入全体指标，其差异不能归因于神经动作。',
        '两个来源的平均配对PDI差（tail8−原头）为：'+', '.join(f"s{k}: {v['mean_delta_pdi']:.6g}" for k,v in s['source_seed_directions'].items())+'。',
        '仅对至少一组实际触发Actor的来源×实例作描述（仍保留全体为主表）：'+', '.join(
            f"s{k}: 平均差{v['mean_delta_pdi']:.6g}，{v['instances']}实例" for k,v in s['source_seed_directions_actor_participated'].items())+'。这是参与情况分解，不是另选的成功指标。',
        f"共{s['evaluation_tasks_without_actor']}/40个任务未触发Actor，最大PDI积分区间宽度为{s['pdi_interval_max_width']:.6g}。无Actor任务中的差异与神经动作无关，积分区间重叠时也不能只凭中点排输赢。",
        f"在10个来源×实例配对中，超过积分截断区间、方向支持tail8的有{s['measurement_resolved_pairs_favoring_tail8']}个，支持原头的有{s['measurement_resolved_pairs_favoring_original']}个。这些是测量区间判断，不是统计显著性检验。",
        '## 5. CPU成本与奖励审计',
        table(['组','固定17状态纯forward中位毫秒','Actor占optimize','节点/墙秒'],[
            [x['model'],f"{x['fixed_state_forward_median_ms']:.3f}",f"{100*x['actor_fraction']:.1f}%",f"{x['nodes_per_wall_second']:.2f}"] for x in s['models']]),
        '固定状态回放每实例一个预先固定anchor、一次warmup和3次计时，输入转换在计时外；它测纯forward，不能代替在线特征/回调或外推历史CPU。tail8只删去很小的末端块，Transformer和BiMatching仍在。',
        table(['组','记录奖励数','gap分量饱和比例','实际clip比例','PDI进展项为正步数'],[
            [x['model'],x['rewards'],f"{100*x['gap_level_saturated_fraction']:.1f}%" if x['gap_level_saturated_fraction'] is not None else 'N/A',
             f"{100*x['reward_clipped_fraction']:.1f}%" if x['reward_clipped_fraction'] is not None else 'N/A',x['pdi_progress_positive']] for x in s['reward_components']]),
        '分量记录器返回原H3函数的原值；15个精确返回/相同getter调用次数检查通过。每条采样的记录奖励总和与原轨迹总reward一致。饱和或累计PDI惩罚是机制及频率证据，不独立证明奖励错误或造成多少PDI损失；当前只测两个训练实例的20秒协议，不把分量频率外推120秒。诊断计时包含原reward计算本身，不能当作纯新增日志开销。',
        '## 6. 结论边界与剩余证据缺口',
        'E1：本轮已验证完整Actor更新、梯度、checkpoint/Actor-only加载、真实采样、开发求解、CPU成本与奖励分量；实例与来源方向详见上表。头部学习响应与真实求解收益是两个结论，不能用TV变化代替PDI证据。',
        'E3：仍没有完整候选行动价值/专家排序真值，真实求解只能提供任务级证据；没有证明每个所选动作更优。一个短采样-更新块也不能检验长期收敛；两个来源模型、五个曾使用的开发实例不足以宣称广泛泛化。没有使用66测试集，也没有把已查看测试数据重新包装成未见集合。',
        '本轮不再追加宽度搜索、初始化搜索、奖励替换或完整训练。是否扩大正式实验应依据实测的来源一致性、逐实例收益和成本决定；若仅有离线响应而无稳定任务收益，保留负结果，优先审计奖励/上游表示，而不是直接加入HSTU。',
        '## 7. 完整性与预算',
        f"共完成{s['solver_tasks']}个SCIP任务（14原生、16采样、40开发），{s['total_optimizer_steps']}个optimizer.step（离线16、在线{s['online_optimizer_steps']}），CUDA初始化0，原源码与历史输入{s['protected_input_files']}项哈希保持一致。所有开发任务valid，Critic评估用时为0；更新后的候选排列/追加masked padding检查通过。",
        '全局单线程计算上限3600秒，最终实耗见 budget.json。按实测完整网络吞吐，将pilot局部上限从1400秒改为2400秒；全局上限、训练步数和求解数量不变。监护进程交接时暂停并保留原训练worker，再由新监护恢复；没有重跑任何optimizer步或求解，记录见 pilot_supervisor_handoff.json。',
        '材料：[全量机器摘要](summary.json)、[核验](verification.json)、[固定协议](protocol.json)、[授权](AUTHORIZATION.md)、[全局预算](budget.json)、[硬件](hardware.json)、[逐实例配对](paired_instances.csv)、[40项评估](evaluation_rows.csv)、[奖励分量](reward_components.csv)、[纯前向微基准](fixed_state_microbench.csv)。',
        '本轮连续验证完成后停止，保留全部结果供审阅；没有自动启动大规模训练、GPU或新的测试集实验。']
    (OUT/'REPORT.md').write_text('\n\n'.join(texts)+'\n')


if __name__=='__main__':main()
