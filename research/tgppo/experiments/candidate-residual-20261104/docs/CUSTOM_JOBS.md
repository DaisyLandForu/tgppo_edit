# TGPPO 候选残差：GPU Custom 作业

**当前状态（2026-11-05，北京时间）：用户已暂时终止 CR 研究线；不要继续提交下列训练、恢复或评估命令。** 入口仅作历史存档，现有权重与记录保留。下一步建议见[独立调研报告](../../../../docs/methods/tgppo_post_cr_20261105/REPORT.md)，新路线尚未启动实验。

**最新：8实例跨实例探索112/112项已完成，执行39分42秒，无需Custom或GPU。** 六Actor保持冻结，训练/原始基线未改。完整主表CR相对B0的PDI平均−0.82%、相对S−0.07%，证明率不变；Actor参与子集有弱正信号，同时存在明显反例。见[跨实例报告](../../../../docs/methods/tgppo_candidate_residual_20261104/cross_instance/REPORT.md)及[更新后的周五汇报稿](../../../../docs/methods/tgppo_candidate_residual_20261104/cross_instance/MEETING_BRIEF_20261106.md)。本阶段已停止，等待验收；下方GPU入口保留作历史存档。

**此前CPU复测已完成：** 现有权重冻结，新增solver seeds2/3的28项任务在本地累计257.80秒完成，无需Custom或GPU。见[CPU复测报告](../../../../docs/methods/tgppo_candidate_residual_20261104/cpu_retest/REPORT.md)。dcmulti新增四配对均改善，vpm2平均PDI仍退步；用户随后授权了以上跨实例阶段。

**主矩阵状态：六个训练作业、统一开发评估及汇总均已完成并验收，无需再申请或提交Custom。** 原阶段材料见[结果复核](../../../../docs/methods/tgppo_candidate_residual_20261104/RESULTS_REVIEW.md)与[周五汇报稿](../../../../docs/methods/tgppo_candidate_residual_20261104/MEETING_BRIEF_20261106.md)，新种子结果以上述CPU复测报告为准。全体PDI的表面优势主要来自无Actor实例，不能归因于残差评分。

以下保留已执行入口和原预算，供复核使用。训练CPU型号限制已放开，原命令自动转入CPU-flex兼容入口。

本地方法检查由 Codex 执行。下面是**真实 PPO 训练**，三组×两种子，最多600个完整120秒episode。

原作业资源：**1张V100、建议8个专用CPU核（最低4核）、32GiB内存、3小时30分钟作业时限**。训练可使用不同CPU型号及V100规格，保留相同软件环境和共享工作盘；Python3.8/Torch1.13.1+cu117及定制绑定要求保留。六条历史命令如下：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant b0 --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant b0 --seed 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant score --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant score --seed 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant candidate --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh run --variant candidate --seed 22
```

本轮六项已完成。此前5个在矩阵环境预检失败且没有checkpoint的作业通过兼容入口继续，失败记录和原已用时间保留，没有清零预算。

每个作业自动完成环境检查、归一化复用/补测、一次不新增SCIP求解的合成GPU更新/恢复检查、训练和20/100里程碑保存。**归一化按CPU/软件指纹分别缓存，同类节点共享，不同CPU不会错误混用。** 不同CPU的准备可以并行，等待和补测仍计入原预算。

六个作业都进入结束状态后，最后结束者自动执行**一次统一CPU开发评估与汇总、作图**。不需要另申 summarize 作业。它会在同一台评估机器上评测全部模型，必要时重测10项同机原生参照；不混用不同CPU的评估数字。

## 预算和停止

- 每个GPU入口累计上限**2小时20分钟**，包含启动、归一化等待/补测、桥接和训练；六项最多14作业小时。3小时30分钟是平台外层时限，为最后一项留出额外CPU汇总时间，不等于六项均可训练3.5小时。
- 统一评估/作图累计上限1小时，本地准备计算上限10分钟；三者合计15小时10分钟，小于原16小时总作业墙时预算。这里按各并行作业耗时**相加**，不是只计算并行经过时间。
- 北京时间**11月5日22:00前结束训练，23:00冻结材料**。到资源/日期上限保留checkpoint，不自动延长训练、补种子或做500 episode。
- 优先统一使用episode100；如部分组到限，只有全部六组均存在episode20时才统一降到20。任一组连20都未完成，则明确报告矩阵不完整，不混用不同训练量。
- 非有限、接口或环境不匹配会停止并保留失败记录。先查看日志再决定是否恢复；同一命令附加`--resume`只使用原剩余预算，不清零预算、不丢旧attempt。平台直接杀死且未完成记账的attempt按保守上限扣费，不能自动续跑。

## 查看进度与结果

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh status
```

主结果：`research/tgppo/runs/candidate-residual-20261104/results/REPORT.md`。同目录自动生成逐任务/逐实例CSV、配对差、训练曲线、方法图、PNG/PDF和权重哈希。汇报以PDI、证明率和实际调用成本为主，不把响应变大当作求解收益。

若所有训练完成但最后的汇总被平台打断，可在原环境或本地CPU环境执行下面一条；它核对已有状态，仅补齐剩余统一评估，不训练，也不重跑已匹配的有效评估：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/candidate-residual-20261104/custom_job.sh finalize
```

只重生图表/文字、不开新求解则用`report`。本轮仅5个已使用开发实例，不使用66测试实例。

训练Python源码、配置、checkpoint协议没有改变；原提交脚本及校验清单已归档，入口补丁在`cpu-flex/`。CPU修订时运行中的B0-s11保留并已正常完成。不同训练CPU、归一化来源和恢复记录已写入`cpu-flex/training_environments.json`及`results/TRAINING_ENVIRONMENTS.md`。最终60项开发评估统一在同一台机器执行；异构训练结果不宣称严格控制训练CPU的消融。详情见[修订记录](../cpu-flex/AMENDMENT.md)。

本轮两个B0均重新训练：旧B0没有本轮统一20-episode回退checkpoint，因此没有把旧ep100自动认定为完全匹配。**原生归一化**在严格核对完整求解身份与CPU/软件指纹后复用或补测。
