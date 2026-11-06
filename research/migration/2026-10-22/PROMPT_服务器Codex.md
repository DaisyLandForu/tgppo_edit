# 给学校服务器 Codex 的接续任务

在服务器把直传压缩包解压到工作目录后，从 research 根目录启动 Codex，并复制下面整段：

> 我已把 TGPPO 项目的直传压缩包解压到当前 research 目录。请先完整阅读 AGENTS.md、migration/2026-10-22/HANDOFF_SERVER_2026-10-22.md、migration/2026-10-22/CODEX_HANDOFF.md、migration/2026-10-22/server-run/README.md、migration/2026-10-22/server-run/protocol.json 和 docs/阶段交付与验收_TGPPO复现与增量改进.md；将其中事实与服务器文件、实例 SHA、环境、Slurm 队列逐项核对。
>
> 我的目标是今晚尽早启动 B0、R0、Rjoint 各训练种子 11/22 的六个独立 V100 作业。每作业 500 个完整 episode，100/200/300/400/500 保存稳定检查点，训练必须可恢复。这个六作业协议已经明确授权；普通构建、路径和 Slurm 参数可自行处理，不要停在计划。阶段 5 尚未完成；阶段 6 不得自动启动。
>
> 请先核对 ARCHIVE_CONTENT_MANIFEST.json 与 25 个训练/开发、66 个测试 .mps.gz 实例及 cutoffs.json；包内已有 SCIP 6.0.1 和 SoPlex 4.0.1 源码，但无已编译求解器。重建 Python 3.8/Torch CUDA/SCIP/SoPlex 与 stage4 定制 PySCIPOpt 五个接口，核对 tgppo/runs/stage4-calibration/final_train_only_statistics.json，确认 V100、CPU/内存配额和 Slurm 分区/账户。不要把源机 .so、旧模型或 running 文件当作服务器验证。新输出放在 tgppo/runs/server6-20261022。
>
> 然后按 server-run/README.md 跑小型检查、服务器原生 relpscost 基线和三组真实 GPU 桥接；核对 CPU SCIP 采样、CUDA PPO、Actor/Critic 更新、优化器与 CUDA RNG 恢复、Actor-only 加载。用代表性训练任务实测吞吐与峰值显存，估算 100/500 episode 的时间。前置检查通过后，立即提交 train.sbatch 六作业数组，并报告作业 ID、每组真实进度和是否能在今晚时限内完成。若队列或时限中断，按 latest.pt 显式恢复；不能覆盖旧尝试或把 heartbeat 当训练完成。若必须改核心实验协议、训练算法或样本预算，先说明影响并征求我决定。
>
> 训练完成后用同一开发集规则为每个训练种子单独选 actor_best.pt，同时保留 actor_final.pt；两个训练种子都报告，不挑一个赢家。选模冻结后，Actor-only 接入 SCIP，在保留测试集和核实过 cutoff 的兼容评测集上与原生 relpscost 公平对照。提交阶段 5 的实际结果、成本、异常、配置和日志路径，等我明确验收后再进入阶段 6。B0 是本轮 120 秒/Adam/四任务批更新参照，不是论文 3600 秒/AdamW 全规模复现；本轮没有 Rprod，不能单独宣称联合关联收益。
>
> 如果某项构建受阻，继续完成不依赖它的核查，并给出确切缺口、已尝试步骤与最短解决路径。保留历史原始日志和哈希台账，不重写旧实验事实。
