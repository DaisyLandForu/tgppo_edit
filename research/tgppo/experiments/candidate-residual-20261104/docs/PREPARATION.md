# 方法实现与GPU提交交付

2026-11-04。**独立实现与本地准备完成，B0-s11已在GPU上运行；其余5项受矩阵环境检查阻挡，现已提供CPU型号放宽补丁，暂无完整开发性能结果。** 用户可按[CUSTOM_JOBS.md](CUSTOM_JOBS.md)直接补跑5项，保留正在运行的B0-s11。最新修订见[CPU-flex](../cpu-flex/AMENDMENT.md)。

实现：原TGPPO、评分捷径S、候选残差CR三组。S/CR新增向量各256参数、零初始化；CR比S仅多原候选编码的残差输入。原Actor、Critic、BiMatching、TreeGate、H3、PPO和memory文件均保留。独立runtime按配置在训练、恢复和Actor-only部署时选择Actor；独立server仅补充方法配置、初始化记录及当前实验的GPU检查入口。

本地11项检查通过，覆盖两个种子的公共权重/RNG与初始策略逐位一致、非零残差公式、非均匀策略下排列与padding、单候选、真实CPU optimizer更新、恢复模型/动量/RNG、Actor-only导出、旧配置兼容、任务顺序前缀、共同预算回退、GPU缺失拒绝与报告配对统计。检查用合成状态/临时报告数据，不作为方法性能实验。

受保护原源码及输入41项SHA核对不变；本地未新增SCIP求解，未运行GPU。最终记录：[local_checks.json](local_checks.json)、[unit-tests.log](unit-tests.log)、[bundle_manifest.json](bundle_manifest.json)。作图的8个PNG/PDF输出检查通过，只保留不含性能数字的[方法图](method.png)与[PDF](method.pdf)；[作图检查](plotting_check.json)明确标记其余图使用临时测试数据。

自动执行：复用或补测一次共享原生归一化；每种架构一次合成GPU更新/恢复检查，不额外采样SCIP；六运行各100完整episode并保存20/100；最后终止的作业选择共同里程碑，统一CPU评估60项开发任务、原生参照与作图。原生记录必须同CPU/软件及完整任务身份匹配，否则补测。不给未启动作业、缺失checkpoint、失败任务或未完成评估补造数字。

每GPU入口累计140分钟×6，统一CPU评估60分钟，本地准备计算上限10分钟，合计上限15小时10分钟，小于方案16小时。平台每项请求3小时30分钟，为最后一项自动汇总预留时间；内部训练上限仍140分钟。训练在北京时间11月5日22:00前结束，23:00冻结。各组100未齐时只能共同回退20；没有完整共同20矩阵则交付缺失状态。

最终真实结果将写入 `research/tgppo/runs/candidate-residual-20261104/results/REPORT.md` 及逐任务、逐实例、配对、训练CSV和PNG/PDF。当前没有创建冒充真实方法结果的性能表。

目标V100、生产batch256真实CUDA更新及求解收益仍须在Custom上实测。当前本地检查只证明实现和CPU链路可用，不保证残差方案得到正收益。
