# 证据来源与阅读顺序

本目录所有本地链接均指向原始阶段产物，材料整理未覆盖这些文件。优先使用日期最新的全66与持续监督结论；旧报告中的“尚未运行”是当时状态，不用于描述现在。

## S1

全面诊断：[第一阶段组会摘要](../research/docs/diagnostics/tgppo_20261030/MEETING_BRIEF_20261030.md)、[论文局限证据矩阵](../research/docs/diagnostics/tgppo_20261030/01_paper_limitations_evidence_matrix.md)、[训练／奖励／评分诊断](../research/docs/diagnostics/tgppo_20261030/02_training_reward_actor_diagnostics.md)、[CPU成本分析](../research/docs/diagnostics/tgppo_20261030/03_runtime_and_solver_diagnostics.md)。用于近均匀策略、历史成本、样本覆盖和事实／机制／假设区分。

## S2

局部定位：[多阶段收缩定位](../research/docs/diagnostics/tgppo_20261102_signal_localization/REPORT.md)、[原PPO单步重放](../research/docs/diagnostics/tgppo_20261102_ppo_replay/REPORT.md)、[原头重置](../research/docs/diagnostics/tgppo_20261102_head_reset/REPORT.md)、[8维读出](../research/docs/diagnostics/tgppo_20261102_tail8_probe/REPORT.md)、[完整Actor与在线pilot](../research/docs/diagnostics/tgppo_20261102_end2end/REPORT.md)。局部响应与真实求解收益分开。

## S3

CR路线：[原方法计划](../research/docs/methods/tgppo_candidate_residual_20261104/PLAN.md)、[初次结果](../research/docs/methods/tgppo_candidate_residual_20261104/RESULTS_REVIEW.md)、[冻结CPU复测](../research/docs/methods/tgppo_candidate_residual_20261104/cpu_retest/REPORT.md)、[8实例探索](../research/docs/methods/tgppo_candidate_residual_20261104/cross_instance/REPORT.md)。本线已终止，不从旧建议恢复训练。

## S4

实际方法：[结构与流程说明](../research/docs/methods/tgppo_light_actor_20261105/continuation/METHOD.md)、[轻量Actor源码](../research/tgppo/experiments/light-actor-20261105/code/project/policy/light_actor.py)、[原Actor源码](../research/tgppo/experiments/light-actor-20261105/code/project/policy/actor.py)、[BC实际损失与选模](../research/tgppo/experiments/light-actor-20261105/code/pretrain.py)、[运行配置](../research/tgppo/experiments/light-actor-20261105/configs/mlp_bc.json)。用于区分结构整体替换、BC初始化与持续监督。

## S5

三结构开发与小样本探索：[完整170项开发结果](../research/docs/methods/tgppo_light_actor_20261105/continuation/DEVELOPMENT_REVIEW.md)、[开发＋144项跨实例解释](../research/docs/methods/tgppo_light_actor_20261105/continuation/RESULTS_REVIEW.md)、[筛选规则](../research/docs/methods/tgppo_light_actor_20261105/continuation/PLAN.md)。这是当时选择MLP＋BC的依据，后来全66对BC结论有所更新。

## S6

专家与训练：[数据兼容性审计](../research/tgppo/runs/light-actor-20261105/offline/data_audit.json)、[冻结原始20／5划分](../research/docs/stage1/protocol_draft.json)、[16组训练审阅](../research/docs/methods/tgppo_light_actor_20261105/results_review/REPORT.md)、[BC表](../research/docs/methods/tgppo_light_actor_20261105/results_review/bc.csv)、[训练表](../research/docs/methods/tgppo_light_actor_20261105/results_review/training.csv)。训练审阅中的早期评估未完成状态已由S5和S8更新。

## S7

PPO增量：[20项BC-only局部消融](../research/docs/methods/tgppo_light_actor_20261105/bc_only_ablation_20261105/REPORT.md)、[70项三时间点开发曲线](../research/docs/methods/tgppo_light_actor_20261105/dev_curve_20261105/RESULTS_REVIEW.md)。没有稳定附加收益不等于PPO理论无效，也不是收敛证明。

## S8

**最新完整全66结果**：[复核报告](../research/docs/methods/tgppo_light_actor_test66_20261105/RESULTS_REVIEW.md)、[自动原表](../research/docs/methods/tgppo_light_actor_test66_20261105/summary.csv)、[独立复核数据](../research/docs/methods/tgppo_light_actor_test66_20261105/REVIEW_DATA.json)、[最终核验](../research/docs/methods/tgppo_light_actor_test66_20261105/FINAL_AUDIT.json)。本次主结果优先采用这些数值。

## S9

全66评测协议：[冻结协议](../research/tgppo/experiments/frozen-test66-20261105/docs/PROTOCOL.md)、[分片记录](../research/docs/methods/tgppo_light_actor_test66_20261105/cohorts.csv)。测量界属于时限过冲的积分范围，不是统计置信区间；未按全66结果重选历史权重。

## S10

**最新持续监督闭环**：[110项结果解释](../research/docs/methods/tgppo_supervised_ppo_20261105/results/RESULTS_REVIEW.md)、[独立重算表](../research/docs/methods/tgppo_supervised_ppo_20261105/results/verified_summary.csv)、[机器复核数据](../research/docs/methods/tgppo_supervised_ppo_20261105/results/REVIEW_DATA.json)、[冻结方案](../research/tgppo/experiments/supervised-ppo-20261105/docs/PROTOCOL.md)。当前分支未通过预定筛选，没有进入全66。

## S11

论文对齐：[B0与论文协议核查](../research/docs/methods/tgppo_light_actor_test66_20261105/BASELINE_PAPER_ALIGNMENT_20261106.md)。当前B0结构保留，训练／时限／优化器等未完全对齐论文；历史500episode也不等于3600秒完整复现。

## 论文原始来源

- Ben Mhamed等：[Learning Branching Policies for MILPs with Proximal Policy Optimization](https://arxiv.org/html/2511.12986v1)，§4.3、§5、§6。用于核对原方法、训练协议与作者明示局限；不将本地评分头问题说成论文自述。
- Zaheer等：[Deep Sets，NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)。支持SET的集合汇聚结构依据；不提供本项目MILP性能保证。
- Lee等：[Set Transformer，ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)。支持ISAB诱导点交互依据；不推导其在当前CPU分支场景一定优于MLP。

上述三个主来源已于2026-11-06联网核对。其他调研见[既有轻量模型调研](../research/docs/methods/tgppo_light_actor_survey_20261105/REPORT.md)；本次不新增论文路线或训练任务。
