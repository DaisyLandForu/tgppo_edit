# 持续专家监督：完整开发结果

110/110有效：100模型＋10同机新relpscost。固定轻量MLP与既有BC初始化；两个训练种子11/22，5开发实例×solver0/1，未访问66测试。

方法增量：Actor损失增加0.01×实例均衡训练专家CE/log(K)，每PPO逻辑minibatch16专家状态；原奖励/Critic/PPO优化步骤及部署结构保持。BC-only不重训。

| 方法 | 合并PDI↓ | 证明率↑ | optimize秒↓ | Actor ms/调用↓ |
|---|---:|---:|---:|---:|
| relpscost | 3061.590 | 70.0% | 59.474 | — |
| BC-only | 3038.224 | 60.0% | 61.825 | 1.725 |
| control-PPO100 | 3035.890 | 60.0% | 61.754 | 1.746 |
| control-dev-selected | 3026.187 | 60.0% | 61.947 | 1.718 |
| expert-PPO100 | 3044.066 | 60.0% | 62.429 | 1.733 |
| expert-dev-selected | 3031.100 | 60.0% | 61.587 | 1.711 |

选择：control s11→20；control s22→20；expert s11→100；expert s22→0

预写筛选结果：**本次持续监督增量尚未过筛选**。失败条件['11:control:all:PDI', '22:control:all:PDI', '22:control:common_actor:PDI', '22:bc:all:PDI', '22:bc:common_actor:PDI', 'fewer_than_two_cross_seed_joint_instance_wins']；跨两种子同时优于对照及BC的共同Actor实例[]。

共同Actor实例['vpm2', 'dcmulti', 'eil33-2', 'neos-476283']；无Actor实例['roi5alpha10n8']。全体与共同Actor面板、逐种子、等episode比较见CSV。无Actor项计时变化不归因于神经决策，积分测量上下界不是统计置信区间。

训练配对CPU/GPU型号一致性{'11': False, '22': True}；不同硬件时保留训练条件混杂边界，不能把同机评估解释为训练也完全同机。

E1：本轮测量只回答持续监督的独立增量。E2：实际配对初始化与CPU/CUDA流、源SHA和监督次数被检查；Actor-only部署不加载专家数据或Critic。E3：收益原因、泛化和更长训练效果仍需验证。

当前旧轻量结构＋BC的已有结果保持；本次监督增量未过筛选不等于原整体方法失败。筛选通过也不自动扩训或测试，需交付验收；没有自动300/500、CL、改奖励或树信用分配。

累计评估墙时及预算见/home/duweiyue25/tgppo-transfer/research/tgppo/runs/supervised-ppo-20261105/development-110/job_state.json；节点/tmp消失后仍可从共享evaluation_rows.json重新汇总。
