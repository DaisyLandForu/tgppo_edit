# TGPPO 候选信息残差评分

**当前状态（2026-11-05，北京时间）：用户已暂时终止 CR 研究线，不再扩大训练、调参或追加该线评测。** 六份权重及全部历史报告保留；下方为历史结果与入口，不是待执行任务。其他改进方向见[后续文献调研与建议](../../../docs/methods/tgppo_post_cr_20261105/REPORT.md)。

主方法：`原logits + wᵀ(BiMatching输出 + 原候选编码)`；消融只使用BiMatching输出，w零初始化、两组各加256参数。原Actor、Critic、奖励和PPO源码保留，独立runtime负责按checkpoint配置选择Actor。

**最新：8个训练外实例的112项跨实例探索评估已完成，CPU执行39分42秒。** 完整主表CR相对B0的PDI平均−0.82%、相对S−0.07%，证明率不变；新增局部收益与明确反例同时保留。见[跨实例结果](../../../docs/methods/tgppo_candidate_residual_20261104/cross_instance/REPORT.md)和[周五汇报更新](../../../docs/methods/tgppo_candidate_residual_20261104/cross_instance/MEETING_BRIEF_20261106.md)。六权重冻结，未训练或申请Custom，已停止待验收。

此前冻结权重的本地CPU复测：28项新种子任务累计257.80秒，dcmulti局部收益得到支持，vpm2平均退步仍在。见[CPU复测报告](../../../docs/methods/tgppo_candidate_residual_20261104/cpu_retest/REPORT.md)。

**主矩阵已完成并验收：六组均100 episode，60项模型开发评估与10项原生参照齐全，无需再提交Custom。** 候选残差得到局部收益，整体策略提升尚未证实；全体PDI优势的95.6%来自未调用Actor的实例，不能据此宣称方法整体有效。

- [结果复核与阶段交付](../../../docs/methods/tgppo_candidate_residual_20261104/RESULTS_REVIEW.md)
- [周五汇报稿：方法、主表、正反例与结论](../../../docs/methods/tgppo_candidate_residual_20261104/MEETING_BRIEF_20261106.md)
- [CPU型号限制修订与硬件来源](cpu-flex/AMENDMENT.md)

- [GPU Custom六条命令、自动评估和预算](docs/CUSTOM_JOBS.md)
- [已批准的方法与实验方案](../../../docs/methods/tgppo_candidate_residual_20261104/PLAN.md)
- [本地准备交付](docs/PREPARATION.md)

代码与配置封存在`docs/bundle_manifest.json`，CPU检查在`docs/local_checks.json`；入口拒绝与检查版本不一致的代码。`configs/{b0,score,candidate}.json`只在方法开关、组别和输出目录上不同。

CPU方法检查不是求解性能结果。已完成的正式PDI、逐实例对照和图表位于`runs/candidate-residual-20261104/results/`；原自动产物保留，解释与收益判断以以上复核报告为准。
