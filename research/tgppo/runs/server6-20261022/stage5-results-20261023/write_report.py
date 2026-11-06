#!/usr/bin/env python3
"""Write the stage-5 delivery from audited aggregates, without experiment changes."""
import csv
import datetime as dt
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
data = json.loads((OUT/'summary.json').read_text())
audit = json.loads((OUT/'audit.json').read_text())
dest = ROOT/'docs/阶段5_服务器六作业结果与验收报告_TGPPO_2026-10-23.md'
rel = '../tgppo/runs/server6-20261022/'
art = rel+'stage5-results-20261023/'
combined = {r['method']:r for r in data['combined']}
native = combined['native']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


main_table = table(['方法','有效实例 / 求解次数','平均 PDI ↓','相对 B0 的 PDI 变化','PAR-2 / 秒 ↓','120 秒内证明完成率 ↑'], [
    ['SCIP relpscost（原生）' if r['method']=='native' else r['method'],
     '66 / '+str(r['runs']), '{:.6f}'.format(r['mean_pdi']), '{:+.4f}%'.format(r['pdi_change_percent_vs_B0']),
     '{:.3f}'.format(r['mean_par2']), '{:.2%}（{}/{}）'.format(r['proof_rate'],r['proof_count'],r['runs'])]
    for r in data['combined']])
seed_table = table(['方法','训练种子','所选 episode','开发集 PDI','测试集 PDI','测试 PAR-2 / 秒','测试证明完成'], [
    [r['method'],r['training_seed'],m['selected_episode'],'{:.6f}'.format(m['development_mean_pdi']),
     '{:.6f}'.format(r['mean_pdi']),'{:.3f}'.format(r['mean_par2']),'{}/198'.format(r['proof_count'])]
    for r in data['per_training_seed'] if r['method']!='native'
    for m in data['selected_models'] if m['model']==r['model']])
uncertainty = table(['对比 B0','平均 PDI 差值','测量差值保守范围','条件实例 bootstrap 95% 区间','逐实例 PDI 低 / 平 / 高'], [
    [r['method'],'{:+.6f}'.format(r['mean_delta']),
     '[{:+.4f}, {:+.4f}]'.format(r['measurement_mean_delta_lower'],r['measurement_mean_delta_upper']),
     '[{:+.4f}, {:+.4f}]'.format(r['conditional_instance_bootstrap_95_low'],r['conditional_instance_bootstrap_95_high']),
     '{}/{}/{}'.format(r['instances_point_lower'],r['instances_point_equal'],r['instances_point_higher'])]
    for r in data['paired_instance_comparisons']])
train_table = table(['模型','完整 episode','有效 PPO 更新','优化步','transition','训练墙时 / 小时','采样 CPU / 核时','峰值 CUDA allocated / GiB'], [
    [r['model'],r['episodes'],r['effective_ppo_updates'],r['optimizer_steps'],r['transitions'],
     '{:.3f}'.format(r['training_wall_hours']),'{:.3f}'.format(r['worker_cpu_core_hours']),
     '{:.3f}'.format(r['peak_cuda_allocated_gib'])] for r in data['training']])
deploy_table = table(['方法','Actor 平均毫秒 / 实际分支回调','Actor / worker 总墙时','完整分支回调 / optimize 墙时','Critic / 扩展搜索特征'], [
    [r['method'],'{:.3f}'.format(r['actor_milliseconds_per_branch']),
     '{:.2%}'.format(r['actor_fraction_of_worker_wall']),'{:.2%}'.format(r['callback_fraction_of_optimize_wall']),
     '均为 0'] for r in data['deployment'] if r['method']!='native'])
models_table = table(['模型','选中 episode','actor_best.pt SHA-256'], [
    [r['model'],r['selected_episode'],'`'+r['actor_best_sha256']+'`'] for r in data['selected_models']])
cost = data['costs']
tasks = list(csv.DictReader((OUT/'per_task_with_sources.csv').open()))
mspp = [r for r in tasks if r['memory_exception']=='True']
mspp_wall = [float(r['worker_wall_seconds']) for r in mspp]
mspp_rss = [int(r['peak_rss_kib'])/1024**2 for r in mspp]
dev = {m:sum(r['development_mean_pdi'] for r in data['selected_models'] if r['model'].startswith(m+'-s'))/2
       for m in ['B0','R0','Rjoint']}
written = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间')
report = f'''# 阶段 5：服务器六作业主矩阵结果与验收材料

整理时间：{written}。对应新协议 `server6-20261022`。这是阶段 5 的实际实验交付，等待用户验收；阶段 6 未启动。旧机器中断矩阵仍保留为历史证据，不与本轮训练混计或冒充续接。

## 1. 主要结论

**六份模型均完成 500 个完整训练 episode；同一开发集选模冻结后，66 个测试实例的七路对照已全部补齐。本轮没有验证 R0 或 Rjoint 相对 B0 的稳定求解收益。** R0 的测试平均 PDI 比 B0 高 0.0291%，Rjoint 高 0.1714%；两个训练种子上的变化方向相反。SCIP 原生 relpscost 在当前共同求解配置下，平均 PDI、PAR-2 和证明完成率均优于三组学习策略。这里的结果是实际测得的短时协议结果，不能写成“改进有效”或“论文性能已复现”。

{main_table}

平均值按**实例等权**汇总，每实例先平均 solver seeds 0/1/2；学习方法再对两个训练种子等权平均。每组学习方法有 396 条求解记录，原生基线 198 条，统计实例数始终都是 66。原生相对 B0 的平均 PDI 降低 **12.6132%**，分母是 B0。正的“相对 B0 变化”表示 PDI 更高、点估计更差。

![主结果]({art}main_results.png)

PDI 是 120 秒内 SCIP 百分比 gap 对时间的积分，单位为“百分比·秒”，越低越好；最大为 12000。它同时反映界值质量和改善速度，不能只解释为运行时间。PAR-2 对 120 秒内完成证明的任务取实际 SCIP 时间，其余记 240 秒。证明完成包括已核实最优 cutoff 下的证明；`infeasible` 可能是 cutoff 证明，不能解释为原问题本身不可行，也不能把“证明完成率”改称“找到最优可行解比例”。

## 2. 两个训练种子与开发选模

{seed_table}

六次训练全部保留 `actor_best.pt` 与 episode 500 的 `actor_final.pt`。本次测试使用各自开发集选出的 `actor_best.pt`，没有依据测试集更换里程碑或挑出一个训练种子当赢家。全局冻结时间为 **2026-10-23 11:16:43 北京时间**，早于本轮所有测试结果文件。

开发选模规则：5 个开发实例 `vpm2, dcmulti, eil33-2, neos-476283, roi5alpha10n8`，每实例相同 solver seeds 0/1/2，比较 100/200/300/400/500 episode，最小实例等权平均 PDI，精确并列取更早里程碑。共 6×5×5×3=450 条有效开发求解，复算每个候选得分与冻结选择一致。

开发选中模型均值：B0={dev['B0']:.6f}，R0={dev['R0']:.6f}，Rjoint={dev['Rjoint']:.6f}。Rjoint 在开发集上的微小优势没有转化为这 66 个测试实例的平均收益。开发分数包含选模效应，不是未见集泛化收益的证据。

![共同预算开发曲线]({art}development_curves.png)

上图是共同训练 episode 预算上的**开发集曲线**；星号是测试前冻结的检查点。没有额外在测试集上扫描五个里程碑。`actor_final.pt` 被保留，但未作为新的候选再测试。

## 3. 不确定性与结论边界

{uncertainty}

差值定义为方法减 B0，负值才是 PDI 改善。测量范围来自每条原生 PDI 对停止时间超出 120 秒部分的保守界限，**不是统计置信区间**。另做 20,000 次按实例配对 bootstrap，固定 NumPy RandomState 种子 20261023；它只反映在这些已训练、已选定模型下的实例变动，不覆盖重新训练种子的总体不确定性，也不使用 396 条重复作为独立样本。

Rjoint 的 PDI 中点 bootstrap 区间偏向 B0，但计入 PDI 测量界限后，平均差值仍可跨过 0；加上只有两个训练种子且方向相反，不能把点估计写成稳健的显著退化或提升。R0 同样未显示稳定改善。原生基线的平均优势在本轮的测量范围及条件实例重采样下方向一致，但它也没有在所有单个实例上更好。

本轮 B0 是修复后的原架构在 **120 秒 / Adam / 四任务批更新 / 500 episode** 下的参照，**不是论文 3600 秒 / AdamW 的全规模性能复现**。R0 接入共同剩余时间及固定训练侧价值校准；Rjoint 在此基础上增加 Critic 搜索前沿联合直方图及相应网络。没有 Rprod 或本轮在线容量对照，不能把差异单独归因于联合关联信息，也不能将之前的工程修复称为研究增益。

## 4. 完整性、硬件和 mspp16 资源修复

当前最终测试是 **1365 条原有效结果 + 21 条 mspp16 补跑 = 1386 条唯一有效记录**，即 7 路×66 实例×3 求解器种子。原来的 21 条失败另存，不计入有效记录。全部任务终止、worker CPU 计时完整、指标有限；Actor 评估未初始化 CUDA、未执行 Critic、未提取 Critic 扩展搜索信息，没有动作 fallback。

原七路评测都尝试完 198 个任务，但每路 mspp16 的三个任务因单 worker **8 GiB 虚拟地址空间上限**报 SCIP 分配失败并以 -11 退出；所以旧平台状态仍然是失败，不能改写为旧作业成功。隔离 E5 诊断仅用于定位内存原因，没有混入 Gold 正式结果。

用户批准的例外只用于 **mspp16**：全部六份模型及原生基线统一为 **24 GiB 单进程地址空间上限、单 worker、32 GiB 容器**；SCIP 搜索仍是 120 秒，watchdog 300 秒覆盖读入和清理。其余 65 个实例保持原 8 GiB 结果。没有重训、扩大 episode 或改求解算法。

补跑 21 条全部有效，状态均为 `timelimit`、PDI 精确等于 12000，证明未完成，节点数和 Actor 分支回调均为 0。大实例在本时限内未进入 Actor 选分支阶段，所以此实例不能用于判断哪个 Actor 更强。worker 墙时 {min(mspp_wall):.1f}—{max(mspp_wall):.1f} 秒，峰值 RSS {min(mspp_rss):.2f}—{max(mspp_rss):.2f} GiB；24 GiB 是虚拟地址上限，不能与 RSS 混为同一个量。最后一份补跑在 14:42:35 北京时间完成。

实际正式评测 CPU 均为 **Xeon Gold 6148 @ 2.40GHz**；预检记录配额约 8.01 CPU、内存约 32 GiB。软件为 Python 3.8、Torch 1.13.1+cu117、SCIP 6.0.1 / SoPlex 4.0.1、stage4 定制 PySCIPOpt。SCIP 与绑定二进制 SHA、数据和 cutoff、冻结 Actor、solver/action RNG 种子均已核查。随机分到 Silver/E7 的启动失败与重复锁竞争日志保留，没有纳入正式结果。相同 CPU 型号不能保证排除共享节点负载及频率差异，此轮没有独占硬件或 CPU/GPU 推理的配对性能实验。

共同求解设置包含：known-optimal cutoff、关闭 heuristics、`conflict_usesb=false`、`probing_bounds=false`、`checksol=false`、`reevalage=0`，均沿用冻结协议。原生对照使用相同外围设置，仅分支策略为原生 relpscost；这里不是不加这些设置的 SCIP 默认求解结果。

## 5. 训练完成与部署成本

{train_table}

六路已审计 Actor/Critic 和两套 Adam 状态、CUDA RNG、稳定检查点、最终轨迹缓存；参数有实际更新。种子 22 在 episode 244/396 对应批次因不足 256 条样本暂不更新，轨迹继续保留并在后续批次消费；所以有效 PPO 更新为 123，而种子 11 为 125，最终无未消费轨迹。这是统一样本阈值规则的结果，不是中途少训练两个 episode。

{deploy_table}

Actor 延迟为总 CPU 推理秒数除以真实 Actor 分支次数，非每实例求解时间。`features` 是原 Actor 所需特征，扩展前沿输入仅供训练 Critic，最终部署关闭。完整回调包含 Actor、原特征等子计时，嵌套计时不可相加。学习方法回调开销可观，但不能仅凭约 21 毫秒/次断定换 V100 后会更快或能追回与 relpscost 的差距；本轮没有该配对试验。

**选模和测试实际都使用 CPU Actor，GPU 利用率为 0 是预期的。** 用户在 Crater 申请 V100 是由于 CPU-only 不能指定 Gold 节点的调度限制，并非测试计算需要。六次正式训练的 PPO 更新实际使用 CUDA，CPU worker 采样不初始化 CUDA。

已测成本分开列账：

- 正式训练：六作业训练进程墙时合计 **{cost['training_gpu_allocation_process_wall_hours']:.3f} GPU 占用小时**；采样 worker CPU 合计 **{cost['training_collection_cpu_core_hours']:.3f} 核时**；CUDA event 更新区间合计 **{cost['training_cuda_event_hours']:.3f} 小时**。CUDA event 是更新区间耗时，不等于 GPU 利用率、能耗或计费时间。
- 开发选模：450 条有效求解，worker CPU **{cost['development_valid_worker_cpu_core_hours']:.3f} 核时**。
- 最终测试：1386 条有效求解，worker CPU **{cost['test_valid_worker_cpu_core_hours']:.3f} 核时**，其中 mspp16 正式补跑 **{cost['test_recovery_valid_worker_cpu_core_hours']:.3f} 核时**，已包含在前一数值中。
- 原 21 条 mspp16 失败：worker 墙时相加 **{cost['old_mspp16_failed_worker_wall_seconds_sum']:.1f} 秒**；缺少完整 CPU 计数，不能编造失败 CPU 核时或从总成本中抹掉它们。
- 另有两个独立的辅助合成 GPU 训练会话，实际运行约 100.93 和 80.06 秒（合计约 0.0503 GPU 小时），均 SIGTERM 结束；它们不是 TGPPO 样本，不进入主结果和上述正式训练预算。

这些量不可以互相相加当作统一成本。环境构建、准备基线、桥接、吞吐试跑、失败预检、排队、容器启动与退出后的占用另有日志，当前缺少完整 Crater 资源账单；因此上述成本不是项目全部花费，也不是平台计费值。`custom_log_inventory.csv` 提供全部启动日志及日志时间窗口近似，不把它冒充调度器资源账。

## 6. 配置、模型与原始证据

{models_table}

全部 `actor_final.pt` 的路径与 SHA 另见 [selected_models.csv]({art}selected_models.csv)。可恢复训练状态为各模型目录的 `latest.pt`；本次无需恢复或重训。

核心身份：

- 解析后协议 SHA-256：`{audit['parent_protocol_sha256']}`。
- 六模型冻结清单 SHA-256：`{audit['frozen_selection_sha256']}`。
- mspp16 修复清单 SHA-256：`{audit['recovery_policy_sha256']}`。
- libscip SHA-256：`{audit['stack']['libscip_sha256']}`。
- stage4 绑定 SHA-256：`{audit['stack']['binding_sha256']}`。

文件索引：

- [审计检查与完成收据]({art}audit.json)、[聚合结果与成本]({art}summary.json)、[全部读取来源的 SHA 台账]({art}source_ledger.json)。
- [主结果 CSV]({art}main_results.csv)、[两个训练种子明细]({art}per_training_seed.csv)、[66 实例逐方法表]({art}per_instance.csv)、[逐实例配对差值]({art}paired_instances.csv)。
- [1386 条任务及原始路径/SHA]({art}per_task_with_sources.csv)、[保留的 21 条失败路径/SHA]({art}preserved_failures.csv)、[全部 Custom 日志索引]({art}custom_log_inventory.csv)。
- [训练成本]({art}training_cost.csv)、[部署成本]({art}deployment_cost.csv)、[共同预算开发曲线数据]({art}development_curves.csv)、[条件重采样与测量范围]({art}paired_uncertainty.csv)。
- [原训练审计]({rel}post-training-20261023/training_audit.json)、[选模复核]({rel}post-training-20261023/selection-review-20261023T032024Z/REPORT.md)、[冻结选模]({rel}post-training-20261023/frozen_selection.json)。原审计中的 `selection_complete=false` 是当时训练审计快照，未被改写；选模完成以随后冻结清单和本次审计为证。
- [只读复核生成的完整 66 实例快照]({rel}test/mspp16-recovery-20261023/reports/20261023T065206075850Z/comparison.json)、[原失败结果目录]({rel}test/official66-best/)、[补跑记录及原始清单]({rel}test/mspp16-recovery-20261023/)、[内存诊断]({rel}diagnostics/mspp16-memory-20261023T054530Z/REPORT.md)。
- [冻结协议](../migration/2026-10-22/server-run/protocol.json)、[补跑例外说明](../migration/2026-10-22/server-run/mspp16-recovery/README.md)、[分析重算脚本]({art}analyze.py)、[绘图脚本]({art}plot_results.py)。

## 7. 遗留范围与验收状态

1. 用户授权的新六训练模型 / 66 测试实例主矩阵已完成；原失败和内存资源例外已披露。另行要求的独立“兼容评测集”尚未提供可核实实例、划分及 cutoff，未运行，也没有把本包 66 实例重复冒充第二个数据集。
2. 只有两个训练种子；没有 Rprod 或在线同容量对照，无法独立归因联合关联信息。PDI 差异小、开发集仅五实例，测量界限与种子波动均限制结论。
3. 本轮没有达到论文 3600 秒 / AdamW 的规模，也没有 CPU/GPU Actor 推理配对对照；后续若改协议、增预算或补消融，需另行明确范围，当前没有自动启动。
4. 此次只新增结果分析、表格、图与本文。原训练/选模/实验入口及运行时源码保持原样，原始日志、模型与台账未覆盖。图采用独立临时 Matplotlib 环境，正式共享 Conda 环境未安装新包。

**阶段 5 交付待用户验收；阶段 6 未启动。** 本轮可如实汇报：“完成了多种子训练、开发选模和 66 实例对照；当前预算下未观察到 Critic 搜索信息带来稳定的求解收益，原生 relpscost 在整体指标上更好。”
'''
dest.write_text(report)
(OUT/'README.md').write_text(f'''# 阶段 5 实际结果（2026-10-23）

[中文完整报告](../../../../docs/{dest.name})。

七路补齐：66 实例、3 个求解器种子、1386 条有效记录。R0/Rjoint 相对 B0 未显示稳定收益；原生 relpscost 整体更好。详见 `main_results.csv` 与 `per_training_seed.csv`。

- `audit.json`：覆盖、冻结、资源例外与软件身份。
- `per_task_with_sources.csv`：所有最终任务及其原始文件和 SHA。
- `preserved_failures.csv`：原 21 条 mspp16 失败，没有删去。
- `per_instance.csv`、`paired_instances.csv`：按实例比较。
- `main_results.png/svg/pdf`、`development_curves.png/svg/pdf`：由真实聚合数据生成；后者仅开发集曲线。
- `training_cost.csv`、`deployment_cost.csv`、`summary.json`：成本、计时及各训练种子结果。
- `source_ledger.json`：只读输入来源；`deliverable_sha256.json`：本次交付文件哈希。

复算（不启动求解、只写本目录分析结果）：

```bash
cd /home/duweiyue25/tgppo-transfer/research
source migration/2026-10-22/server-run/env.sh
/home/duweiyue25/conda/envs/tgppo/bin/python tgppo/runs/server6-20261022/stage5-results-20261023/analyze.py
```

绘图需要 Matplotlib；本次使用独立临时环境 `/tmp/tgppo-stage5-plot-20261023`，不依赖其持续存在，已保存 PNG/SVG/PDF。运行 `plot_results.py`、`write_report.py` 可由分析产物重建图和报告。重新生成后需更新交付文件哈希。阶段 5 待验收，阶段 6 未启动。
''')
print('REPORT_WRITTEN='+str(dest))
