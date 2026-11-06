# 第一阶段：证据盘点与协议审计

整理日期：2026-10-30（Asia/Shanghai）。本目录为新的诊断交付，待验收；不推进第二阶段。
本轮以[2026-10-29 交接文档](../../../../TGPPO_瓶颈诊断_Codex工作交接_2026-10-29.md)为研究指导；先读[组会摘要](MEETING_BRIEF_20261030.md)。

证据等级：**E1** 已测事实；**E2** 代码或论文机制；**E3** 待验证假设；**E4** 当前证据否定或支持不足。源码机制不能直接升级为求解性能归因。

## 1. 本轮实际完成与版本身份

**E1：当前目录是没有 `.git` 的传入快照。** 工作区根目录、`research/`、上游 `tgppo/repo/`、服务器实验目录均无 Git 元数据，因此 `git rev-parse HEAD` 无法取得 HEAD。交接中的 `e00cc4e…` 不是本轮已核实 HEAD，不能据此宣布版本相同。训练配置记录的上游来源是 `4f90ccddd7999779c93ff2425ca8dd056113a523`，它也不等于本地修改后的运行版本。

**E1：改用内容身份核查，历史来源台账 1,665 项全部匹配，无缺失或不符。** 其中包括实际服务器代码、输入数据、checkpoint、选模冻结清单、逐任务原始结果及历史报告来源。本轮重新计算 39 个运行 Python 文件的组合哈希，按原 `server.plan()` 展开协议并使用原序列化规则，复得历史协议 SHA。证据：[逐文件核验](source_hash_audit.csv)、[协议重算](protocol_reward_audit.json)。这是内容连续性证据；不能恢复 Git 提交历史，也不能独立认证历史硬件测量。

| 身份 | 本轮重算 SHA-256 |
|---|---|
| 原始 `protocol.json` 文件字节 | `6e1b13e97c103496cbed1d502f4d8ecc7c6bdfe7ef7d5741aea2aef02aedf13a` |
| 展开后的运行协议（含划分、数据与源码身份） | `32bed3e2cce3dd36e30e5bde1d3c95321b322081ce123296f00b92e3cdf18b80` |
| 39 个运行 Python 文件组合 | `f46650285933dd69789b2164b5cce5b3122bf9f5b9deb8b00c323a73f8210ccf` |
| 六模型选模冻结清单 | `a8489c97fa042f980fc6f2b770ad1e1597524b4b38d8b91f721f25df1c32bd1a` |
| 历史 libscip 内容身份 | `07f3c32fa1fab0a43559bc9968633edef6a346f6a9ee5147cff8808a31a02613` |
| 历史定制绑定内容身份 | `78659cc66d900fa622e0a26f7958f7902bb5f65cea769079c0f6d3a9634c52f3` |
| 本轮取得的论文 v1 HTML | `b7f36cca6002fc98a29ac85ec95fd7d84cd8a7ff6b3e503b8f4a761fa41a621f` |

原始协议文件 SHA 与展开后协议 SHA 本来就不同，不应误报为版本冲突。重算方法见 [audit_protocol_reward.py](scripts/audit_protocol_reward.py)。

**E1：Actor、BiMatching/TreeGate、H3 三份源码与本地保存的上游对应文件逐字节相同。** 本次评分退化迹象发生在继承的网络与已有训练权重上；当前没有证据把它归因于本轮新增诊断或 Critic 扩展。相同源码不保证其他训练协议相同。

## 2. 实验矩阵与证据账本

主运行目录：[server6-20261022](../../../tgppo/runs/server6-20261022/)。历史根 README 与 `research/AGENTS.md` 的旧阶段状态已过时；结果状态以[2026-10-23 权威报告](../../阶段5_服务器六作业结果与验收报告_TGPPO_2026-10-23.md)、冻结清单和原始产物为准。该历史报告仍标“待验收”，本轮不代替用户补写验收记录。

| 内容 | 实际可用证据 / 核查结果（E1） | 限定或缺口 |
|---|---|---|
| 训练 | B0/R0/Rjoint × s11/s22；各 500 episode；六份 summary 均完成 | 新实验六路，不混入旧 stage5 中断矩阵或后来的 transformer ablation |
| 权重 | 每路 ep100/200/300/400/500、latest、actor_best、actor_final，共 48 个 `.pt` | 未发现可归属本六路的初始权重；`initialization.json` 只有摘要哈希，init 输出为 N/A |
| PPO 日志 | 750 条记录，过滤 `updated=true` 为 744 次更新，22,539 个 optimizer steps | 未存每步 ratio 分位数、KL、历史梯度；不能由 clip=0 倒推 |
| 轨迹 | 3,000 份训练 pickle 均存在；本轮读取全部 1,000 份 B0 轨迹，615,922 个 transition；其余两组使用既有逐 episode 元数据 | B0 全量统计与六路固定观测检查分开计数；没有把未读取的其余 pickle 宣称已全量审计 |
| 开发选模 | 5 实例 × 3 solver seeds × 5 checkpoint × 6 模型 = 450 条历史开发结果；每模型独立冻结选择 | 没有测试全部里程碑；开发曲线仅用于探索 |
| 测试 | 7 路 × 66 实例 × 3 seeds = 1,386 唯一有效任务，重新核对组合键与实例等权聚合 | 66 才是实例统计单位；学习方法每组 396 条，native 198 条 |
| 原始成本与曲线 | 每任务汇总计时、最终节点、bounds_trace 可读；本轮提取实际 gap–time 示例 | 缺节点与时刻同步序列、每次 Actor 耗时和测试候选数量；不能生成真实 per-callback p95 或 gap–nodes 曲线 |
| H3 | 存逐步总 reward、value、log_prob、done、完整 episode 边界 | 缺每步 H3 分量及同步原始 bound/PDI/时钟快照；总 reward 不能唯一反解 |
| 校准背景 | [统一 Critic 校准报告](../../阶段4补充_统一Critic校准与验收报告_TGPPO_2026-10-21.md) | 本轮引用历史结果，不重做拟合；离线 MSE 不作为求解收益 |

历史选模冻结时间为 2026-10-23 11:16:43 北京时间，早于最终测试结果文件，历史审计与本次哈希核查一致。完整训练来源与冻结模型见 [training_audit.json](../../../tgppo/runs/server6-20261022/post-training-20261023/training_audit.json)、[frozen_selection.json](../../../tgppo/runs/server6-20261022/post-training-20261023/frozen_selection.json)、[selected_models.csv](../../../tgppo/runs/server6-20261022/stage5-results-20261023/selected_models.csv)。

## 3. 论文—本地协议对齐

原文：[TGPPO v1](https://arxiv.org/html/2511.12986v1)，核对 §4.1–4.3、§5.1–5.4 与附录 A；本地公式差异详见 [02](02_training_reward_actor_diagnostics.md)。以下属于 E2 的论文/代码描述，历史数值配置同时有 E1 配置文件支持。

| 维度 | 论文 | 实际 server6 | 判断 |
|---|---|---|---|
| 求解器 | SCIP 6.0.1、定制 PySCIPOpt | 同版本，SoPlex 4.0.1、stage4 定制接口 | 版本层面一致；论文二进制/全部参数无逐项独立证明 |
| 外围设置 | 已知最优 cutoff、关闭 primal heuristics | 同原则；另冻结 conflict_usesb、probing_bounds、checksol=false，reevalage=0 | 有受控设置可比性；不是 SCIP 默认生产配置 |
| cuts / presolve | §6 将更完整交互列为未来方向 | settings.py 明确保留 presolve 和 root/local separation，maxrounds=-1 | 不能从论文未来“re-enable cuts”反推本地关闭 cuts |
| 时限 | 3600 秒 | 120 秒 | 设计性缩预算；改变难度、可见轨迹和 PDI，影响结论 |
| 数据 | 25 个训练实例，solver seeds 0–4 | 固定 20 train / 5 dev，seeds 0–2 | 实例覆盖、增强重复不同，不能说完全对齐 |
| 选模 | nested CV、nodes/PDI 复合准则 | 5 个开发实例的算术平均 PDI，5 个里程碑独立选模 | 有选模不确定性；不是复现论文 CV |
| 更新与优化器 | 500 episodes，每 episode 后 3 epochs，AdamW | 500 episodes，4 episode 批收集，3 epochs，Adam | 影响优化；没有本轮证据证明影响为零 |
| 网络与常用超参 | 256 / 5 层 / 8 头，γ=.97、λ=.92 等 | 相应数值一致；Actor rollout/重算 eval，Critic 更新 dropout=.05 | 数值一致不代表训练过程等价 |
| 参数共享 | §4.1 称 shared front-end | Actor/Critic 独立实例，参数交集被代码检查为空 | 论文文字—代码核查项，禁止据此直接重构 |
| Actor 观测 | 25 维候选，61 维 node+MIP | 25、8+53；R0/Rjoint 增量仅给 Critic | Actor-only 评估未使用额外信息 |
| H3 | 附录 A 的 difficulty 和 PDI 项公式 | 保留上游源码的分段线性 difficulty、baseline-level PDI 项 | 有明确公式对齐缺口；尚未量化性能影响 |
| 汇总 | easy 节点 / hard PDI、SGM | 全 66 实例等权算术平均 PDI，另报 PAR-2/证明 | 指标、划分均不同，禁止直接比较百分数 |
| 硬件 | 40-core CPU 节点 | 历史 Gold 6148；V100 更新、CPU 采样与部署 | 资源方案不同；没有跨硬件因果对照 |

本次没有把上述差异中的任何一项标为“无影响已验证”。基线 B0 的准确名称仍为“当前 120 秒共同协议下的修正原架构参照”。

## 4. 异常、样本覆盖与统计边界

**E1：21 条 mspp16 原失败保留，21 条内存例外补跑有效。** 同一实例七路统一 24 GiB 地址空间、单 worker、120 秒；其余实例保留原 8 GiB 记录。补跑均 PDI=12000、节点=0、Actor=0，不能作为 Actor 比较证据。原失败 CPU 计数不完整，不能补造核时。

**E1：全 3,000 训练任务中 554 次没有 transition。** 对两个 B0，`comp21-2idx/misc03/supportcase7` 各 25 次均无训练决策；20 个名称等权抽样不等于 20 个实例等权贡献 PPO 样本。B0-s11 前三大样本源占 50.33%，B0-s22 占 50.65%。证据：[训练覆盖](training_instance_coverage.csv)、[元数据审计](protocol_reward_audit.json)。这确认样本不均衡，尚不能认定重采样必然改善。

本轮已查看的 66 实例全部标记为**后续探索数据**，不得继续调参后仍称独立未见测试。新正式确认需要外层保留集或事先冻结的新划分。solver seeds 为实例内重复；两个训练种子不足以估计重新训练总体稳定性。既有实例 bootstrap 固定了六份已选权重，不覆盖训练种子不确定性；PDI overshoot 保守界限是测量区间，不能当统计置信区间。

## 5. 输出与权限边界

本轮只新增本目录文件：只读统计、12 份 checkpoint 的冻结前向、无 optimizer 的梯度检查、文献快照、图表与报告。无新 SCIP solve、无 GPU 初始化、无权重更新、无基线代码/原日志覆盖；离线 Actor 检查约 41 秒 CPU 墙时，主要数据审计约 230 秒。当前时间不同机器上的离线耗时不用于比较历史部署性能。

复现入口与验收检查见 [README](README.md)。第二阶段仅有[候选方案](04_hypotheses_and_low_cost_validation.md)，未执行。等待用户对本阶段报告验收。
