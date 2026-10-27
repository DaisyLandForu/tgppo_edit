# 学校服务器交接：TGPPO 六作业训练

更新日期：2026-10-22，北京时间。此文件是交接记忆；实验结果以原始日志、检查点与评估结果为准。源机已经准备服务器代码并通过本地 CPU 检查，**没有在实验室 V100/Slurm 验证，也没有启动六个正式训练作业**。

## 研究目标和真实进度

目标是在统一的 120 秒 MILP 分支协议下比较：B0（修正后的原 TGPPO 架构）、R0（共同价值校准）、Rjoint（Critic 增加动作前搜索前沿联合信息）。Actor 输入和参数不增加；部署只需 Actor。Rprod 是边际外积对照，本轮因时间/算力暂缓，因此本轮不能单独证明“联合关联”带来的收益。

用户已明确授权 B0/R0/Rjoint × 训练种子 11/22 共六个 V100 作业，每个 500 个完整 episode，100/200/300/400/500 保存检查点。每作业模板 8 CPU/32 GiB/1 GPU、24 小时上限。四个完整任务一批，CPU SCIP 采样、V100 FP32 PPO 更新。目标服务器吞吐尚未测得，不能保证一夜跑完；应使今晚启动的作业具备完整恢复能力。

阶段 1—4（含统一 Critic 校准）已验收。阶段 3 B0 只训练 4 个完整 episode，并未复现论文性能。旧阶段 5 CPU 九作业矩阵中断：R0/s11 与 Rprod/s11 各 16 episode、4 次完整 PPO 更新，另外七个未启动，无最终选模/验收。旧 running 标志过时，原始记录保留。新六作业是独立实验，不叫旧矩阵续接。阶段 6 未启动；完成阶段 5 后先汇报等待用户验收。

论文完整配置与本轮不同：论文使用 3600 秒任务、AdamW、每 episode 更新；本轮 B0 是 120 秒/Adam/四任务批更新共同协议的参照，不能称论文全规模复现。

## 代码、协议和未包含的数据

- 本轮研究代码：tgppo/experiments/server-20261022。六作业配置和启动脚本：migration/2026-10-22/server-run；先读该目录 README.md。
- 上游原版：tgppo/repo。历史 stage2 至 stage5 版本均保留。stage4 定制绑定源码位于 tgppo/experiments/stage4/metrics_binding；其 src 需在 PYTHONPATH 中先于本轮 code。必须有 getCandsState、getMIPState、getNodeState、getIntegralSnapshot、getOpenNodeSnapshot 五个接口；普通 PyPI PySCIPOpt 1.4.9 不满足需求。
- 归一化所需的训练侧统计 tgppo/runs/stage4-calibration/final_train_only_statistics.json 已纳入交接，μ=-13.166822661165169，σ=8.840326545776836。value 和 GAE 保持原单位，仅 Critic 损失内部标准化。
- cutoffs.json、download_manifest.json 和全部 25 个训练/开发、66 个测试 .mps.gz 实例均包含在直传压缩包中；解压后核对文件 SHA。若后续只从公开 GitHub 仓库克隆，需要按其独立清单确认数据是否齐全。
- 旧模型 .pt、轨迹 .pkl 和编译二进制未打包。直传包包含 SCIP 6.0.1、SoPlex 4.0.1 源码（排除旧 build/.git）；固定上游 commit 在 tgppo/setup/source_versions.json，可在服务器从包内源码重建到 tgppo/solver。不要把源机 .so 直接视为适合新 ABI。
- 环境描述在 migration/2026-10-22/environment。直传包文件与排除清单见 ARCHIVE_CONTENT_MANIFEST.json。新机器需自建 Python 3.8 + Torch/CUDA + 定制 PySCIPOpt + 求解器；源机未构建可直接使用的容器镜像。

## 今晚在服务器的执行顺序

1. 解压到服务器 research 根目录。阅读 AGENTS.md、本文件、CODEX_HANDOFF.md、server-run/README.md 和冻结协议；核对目录与文件清单。不要把代码同步当成环境验证。
2. 在构建/计算节点用包内源码重建 SCIP 6.0.1、SoPlex 4.0.1、Python 环境和 stage4 定制绑定。env.sh 默认 SCIP 安装根为 tgppo/solver；将 TGPPO_PYTHON 指向新环境的 Python。核查 CUDA 真正看到 V100、CPU/内存配额、五个绑定接口及本机小型检查。
3. 核对包内 MPS 实例、train/dev/test 划分、实例 SHA 和 cutoffs。服务器原生 relpscost 归一化基线必须重测；旧机器结果只能作为历史证据。
4. 按 server-run/README.md 填 Slurm 分区/账户后，先完成 25×3 原生基线和完整 256/5/8 模型的三组目标 V100 桥接。桥接核对真实完整任务、CPU rollout/GPU 重算、Actor/Critic 更新、优化器及 CUDA RNG 恢复、Actor-only 加载。用代表性训练实例估算 100 和 500 episode 的时间/显存；若需调共同实验协议，应明确记录差异。
5. 前置条件通过后，直接提交已授权的六作业训练数组。各作业独立目录，latest.pt 供显式恢复，100/200/300/400/500 为稳定里程碑；只有 summary.json 才代表 500 任务完成。队列限时则续跑，不覆盖旧尝试，核查 sacct 记账及真实状态。
6. 六作业结束后，每个训练种子分别在固定的 5 个开发实例×3 个求解器种子上比较五个里程碑，取实例等权 120 秒 PDI 最低者，导出 actor_best.pt 并保留 actor_final.pt。两个训练种子都报告，不再从两个种子中只挑一个。
7. 选模冻结后，Actor-only 接入 SCIP 分支回调，在 66 个保留测试实例和后续兼容评测集上，与原生 relpscost 用相同时限、硬件、SCIP 参数和配对求解器种子比较。新评测集需 .mps.gz、可信 cutoff 和明确清单；记录逐实例 PDI、证明完成、节点、时间、回调开销与失败。阶段 5 完成后报告并等待用户验收，不自动进入阶段 6。

## 证据与边界

源机本地 50 项检查通过，记录在 server-run/verification.json 和 tests-final.log；这些是工程检查，不是 V100 正式训练结果。旧阶段 5 断点与进程状态在 observed_status.json。早期 12 作业成本设想保留在 server_training_cost_proposal.json，**不是当前六作业冻结配置**；当前配置以 server-run/protocol.json 为准。路径、作业队列、GPU 性能与今晚能否完成均须在目标机核查。完整运行命令见 server-run/README.md。
