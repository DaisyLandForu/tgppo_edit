# TGPPO 项目接续说明

核查日期：2026-10-22（北京时间）。此文件是可迁移的项目记忆；实验事实以模型、原始日志和报告为准，不把聊天承诺当成实验完成证据。

## 用户目标与工作约定

复现 TGPPO MILP 学习分支，先跑通可靠闭环，再验证 Critic-only 搜索信息，之后迁往实验室服务器扩大正式实验。原计划周五 2026-10-23 汇报。

用户要求：每阶段完成后交付总结和可核查材料，等待明确验收再进入下一阶段。阶段 1—4（含统一 Critic 校准）已验收，阶段 5 已授权但尚未完成或验收；阶段 6 未授权启动。充分调研之后才增加新方法。没有必须得到正结果的要求。

不要把迁移解读为授权扩大训练预算、改 GPU 在线学习器、改奖励或更换数据。正式规模实验需要独立配置与公平预算；先完成迁移验证，再向用户交付具体方案。

## 当前可信状态

- 阶段 1：论文、官方代码、相关工作调研及协议冻结。
- 阶段 2：GAE、有限时任务终止、奖励/评估、Actor-only 路径等正确性修复；已验收。
- 阶段 3：75 次原生基线、15 次开发评估；交付 B0 仅完整训练 4 个 episode，没有复现论文性能。
- 阶段 4：50 条固定策略轨迹、46,001 条决策，搜索前沿实现及离线拟合；其后统一价值尺度校准，各组能拟合训练回报，但泛化弱，联合关联收益未稳定证实。
- 阶段 5：三组真实在线更新桥接完成，46 项自动检查和 40 项桥接审计通过。正式矩阵仅 R0/s11、Rprod/s11 启动；2026-10-22 核查时调度器、监控器、训练进程均不存在，停止原因尚未查明。旧 scheduler/live_status 的 running 标志已过时。
- 已实际读取两份 latest.pt：两组各有 4 轮完整 PPO 更新、16 个任务进入完整更新，policy_version=4，memory 为空。不要把这两份模型称为九次训练的最终共同预算模型。
- 其余七次训练未启动，阶段 5 最终开发评估没有落盘。没有 summary.json、selected_checkpoints.json 或最终验收报告。

核查证据另存 `observed_status.json`；旧原始日志保留，不伪造正常退出记录，也不把用户未要求的“暂停”写成用户指令。

## 方法和冻结对照

R0：原状态 + 共用剩余时间 + 共同价值校准；Rprod：边际外积；Rjoint：完整联合直方图。Rcap 为离线容量对照，HL-Gauss 仅离线实验。在线主线 MSE。

Actor 不接收新增特征、与 Critic 无共享参数，部署仅加载 Actor。前沿是动作前全部 leaves/children/siblings，8×8 界值/深度直方图 + 5 个标量。Rprod/Rjoint 同容量同提取过程，R0 无额外前沿扫描。

固定训练侧校准：μ=-13.166822661165169，σ=8.840326545776836。rollout value、GAE 与回报目标均是原回报单位，只有 Critic 损失标准化。在线训练完整 Critic，不使用离线编码缓存。离线校准头不作为在线组别专属预训练。

三组×训练种子 11/22/33，求解器种子 0/1/2；原授权每次最多 50 episode / 2 CPU 核时。任务时限 120 秒、watchdog 180 秒，20/5 训练开发划分；66 测试实例保留。主指标 120 秒原生 PDI（保留超时测量区间），并报告证明完成、PAR-2、成本和失败。样本统计单位是实例，不能把求解器重复当作独立实例。

Actor lr=.00024，Critic lr=.00012，clip=.16，entropy=.003，gamma=.97，lambda=.92，batch=256，epochs=3；微批 16，维持外层逻辑 minibatch 与优化步，不截断候选。Actor rollout/重算 eval，Critic 更新 dropout=.05。

## 代码和入口

原版源码：`tgppo/repo`，commit `4f90ccddd7999779c93ff2425ca8dd056113a523`。

各修复/研究版本位于 `tgppo/experiments/stage2,stage3,stage4,stage4-calibration,stage5`。当前研究主线 `stage5/code`。它们是带未提交改动的 Git worktree；仅 clone 官方仓库会丢掉所有本地修改，不能代替完整目录/补丁。

关键文件：

- `stage5/code/project/online.py`：带新增输入的轨迹 schema3、原单位 value/GAE、归一化价值损失。
- `stage5/code/project/policy/calibrated_critic.py`、`search_critic.py`：校准与信息表示。
- `stage5/code/stage5.py`：训练及评估入口。
- `docs/stage5/orchestrate.py`、`finish_stage5.py`：批任务与报告；当前进程已停止，不能直接再次运行到旧目录。
- `docs/stage5/execution_config.json`、`docs/stage1/protocol_draft.json`：冻结实验配置及阶段授权。
- `docs/stage5/stage5.patch`、`stage5_from_calibration.patch`：完整及增量补丁。

主线导入路径必须让 `tgppo/experiments/stage4/metrics_binding/src` 排在前面，再使用 `tgppo/experiments/stage5/code`。只 source `tgppo/activate.sh` 会指向旧原版 repo/code，不足以启动研究主线。

此 custom binding 包含 getCandsState/getMIPState/getNodeState，以及 getIntegralSnapshot、getOpenNodeSnapshot。普通 PyPI PySCIPOpt 1.4.9 不能代替。需要把 .pyx/.pxd/.pxi 等源码和 SCIP/SoPlex 构建资料带走，新服务器按 ABI 重编译；不能只搬 .so。

## 恢复与迁移边界

`project.runtime.load_agent` 可恢复模型、两套优化器、RNG、轨迹及 policy_version。但当前 `stage5.py train` 没有完整矩阵的 --resume 入口，且会拒绝已有输出目录。恢复矩阵调度和 CPU 预算记账仍需补齐。不要靠删 scheduler.json 或重置 next_episode 冒充完整接续；不要丢弃之前已消耗的 CPU。

旧代码中的 `/root/research`、`/root/anaconda3`、CPU affinity=[0,1,2,3]、资源上限和 checkpoint 路径依赖本机。迁移到非 root 或 Slurm 环境时，在独立副本调整入口，不对历史结果 JSON、已冻结源码和哈希台账全局替换路径。

使用 Slurm/实验室作业系统或独立持久会话托管长任务；Codex 恢复对话不等于恢复 Python 进程，也不保证对话结束后队列永久存活。

旧机器的原生基线作为历史证据保留。正式换机实验需要同服务器重测归一化基线；若从 120 秒改为 3600 秒，更不能复用旧基线。

## 必读文档

1. `docs/阶段交付与验收_TGPPO复现与增量改进.md`
2. `docs/阶段1_实验协议_TGPPO.md`、`docs/stage1/protocol_draft.json`
3. `docs/阶段3_基线复现与验收报告_TGPPO_2026-10-21.md`
4. `docs/阶段4补充_统一Critic校准与验收报告_TGPPO_2026-10-21.md`
5. `docs/阶段5_在线PPO对照接续方案_校准后_TGPPO_2026-10-21.md`
6. 本目录 `README_迁移清单与顺序.md`、`observed_status.json`。

## 新服务器的首条任务建议

“请先完整阅读 AGENTS.md 和 migration/2026-10-22/CODEX_HANDOFF.md，核对我迁来的代码、定制绑定、数据哈希与有效 checkpoint。当前阶段 5 未完成；先检查环境、路径与作业恢复方式，提出正式实验的具体配置和预算，暂不启动大规模训练。遵守逐阶段交付验收。”


## 2026-10-22 后续：用户明确的六 GPU 作业方案

用户本轮明确选择 B0/R0/Rjoint×训练种子11/22，六个 GPU 作业各500 episode，每100保存，按开发集选最好模型。此为新的具体准备范围；旧50episode/2CPU核时入口不改写，不能把新作业混成旧矩阵的续接。

已建立独立副本 `tgppo/experiments/server-20261022`，启动包 `migration/2026-10-22/server-run/`。支持 CPU SCIP 采样、CUDA PPO 更新、完整状态/CUDA RNG恢复、六作业Slurm数组、开发集选模、Actor-only导出及未见集合评估。以包内README/协议/检查记录为准。

本机CPU路径和中断恢复/选模检查通过；**实验室V100及Slurm尚未验证，完整六作业未启动**。正式训练设目标GPU桥接门槛；迁移后应先重测同类CPU/软件的基线，再通过GPU真实更新与恢复。不要把代码已准备写成V100已验证或500已完成。

开发集仅用于在100/200/300/400/500之间逐训练种子选权重，导出actor_best.pt并保留actor_final.pt。两个训练种子均应评价，测试集不得用于选模。当前仍是120秒/Adam/四任务更新的短时协议，B0不能冒充完整论文3600秒/AdamW复现。

## 2026-10-22 直传压缩包更新

用户决定先用压缩包迁移，GitHub 上传留待写入授权之后。最新直传包与逐文件 SHA 清单位于本目录，具体使用方法见 `README_迁移清单与顺序.md` 和 `HANDOFF_SERVER_2026-10-22.md`。用户特别确认 MPS 实例属于训练/评估数据，因此**最新包包含 25 个训练/开发和 66 个测试 `.mps.gz` 实例**；此前关于“数据需要另行手传”的说法对最新包已失效。包内也包含 SCIP/SoPlex 源码，但没有编译产物、旧模型、轨迹、Conda 环境或 Codex 登录状态。

服务器 Codex 的可复制任务位于 `PROMPT_服务器Codex.md`。本地打包和校验不能替代目标机器的 Python/SCIP/定制绑定、V100/Slurm 检查；上述六作业范围已由用户明确授权，前置验证通过后可直接按冻结协议启动。
