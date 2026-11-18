# 11月9日训练任务失败与平台回收核查

本次只读原记录，修复放在独立 `paper-recovery-20261109` 入口。
没有提交／取消 Custom，没有真实训练、SCIP 求解或 GPU 启动。
快照时间与原记录在 `SNAPSHOT.json`；运行中的条件可能随后继续推进。

## 逐项结论

| 条件 | 已提交 episode | 已确认事实与原因 | 处理 |
|---|---:|---|---|
| 新 B0-s11 | 11 | 今早启动日志报 `Prior execution retained; use explicit --resume`，没有新训练进度 | 新入口自动显式恢复；保留新 trainmode 协议 |
| 新 B0-s22 | 12 | 同上；不是此次 V100 不支持或 loss 异常 | 同上 |
| 旧 MLP-s11 | 113 | 第114次完整轨迹及 pending 已保存，更新时收到 SIGTERM；与截图“已释放”一致 | 恢复 pending，无需再求解该轨迹；CPU 副本避免 GPU 空闲回收 |
| 旧 MLP-s22 | 70 | 第71次 `Cached task identity changed`；仍用了旧 paper-aligned 入口 | 新入口隔离 CPU cohort，不删除旧采样身份 |
| 旧 MLP＋BC-s11 | 137 | 核查时处于下一次 normalization，原续跑仍有提交进度 | 保留正在运行的作业，不并发迁移 |
| 旧 MLP＋BC-s22 | 78 | 快照时正在更新第79次，随后可能继续推进 | 保留，不并发迁移 |
| 旧 B0-s11 | 137 | 当前 `stopped_before_next_complete_episode`，累计脚本预算仅余1594.454秒，小于5100秒完整episode预留 | 用显式 `--no-total-wall-limit` 可继续；不是平台设了48小时 |

08:39附近另有 `BlockingIOError: Resource temporarily unavailable` 的启动日志，
证明某次启动碰到同条件执行锁。与截图旧B0恢复失败、原B0续跑随后仍推进的时间相符；
该旧 launcher 日志未打印 variant，不能仅靠文件名百分之百绑定到截图任务。
新入口在持有本作业锁时才写失败状态，避免不同容器相同 PID 导致重复启动覆盖原状态；不能删除锁绕过。

`CHECKPOINT_READ_AUDIT.json` 对五个已停止条件的完整 checkpoint 作了 CPU 只读加载：
模型、两套 AdamW、轨迹、RNG 和游标字段均在；MLP 两套优化器分别18／118个已初始化状态，
B0分别116／118个。没有重置动量、权重或进度。

## 两小时回收为何会重复

`IDLE_TIMELINE.json` 中 MLP-s11 第114次：

- 新 CPU 原生归一化 worker：3600.515秒。
- 接下来的完整采样 worker：3601.035秒。
- pending 保存后约1.144秒收到 SIGTERM，更新未提交。

两段 CPU 任务累计7201.55秒，足以覆盖用户提供的两小时 GPU 闲置窗口。
日志直接证明 SIGTERM 和采样／更新位置，平台的“低利用率回收”原因来自用户截图／说明；
没有发现这次错误由 CUDA OOM 或 NaN 导致。
换节点若 CPU 改变，还需要匹配的新原生分母，重新提交可能再次出现同一间隔。
即便只剩采样，连续无 Actor 的 episode、根节点处理和更新前 CPU 计算也可能延长闲置。

## 修复与边界

统一入口：
`research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh`。

GPU 模式恢复原目录，自动显式 resume，并只允许 CPU 型号变化的采样迁移；
任务／checkpoint／动作种子／软件／绑定仍严格检查。这修复启动失败，不保证免于再次回收。

CPU 模式在独立 `runs/paper-recovery-20261109/cpu-{aligned,trainmode}` 中续训。
首次复制原权重、AdamW、RNG、进度与有效 pending，只修改 learner_device 和配置哈希；
pending 的清理目录改到新副本，源轨迹保留。源实验有活动锁时拒绝导入。
之后从自己的 latest 恢复，不重新导入父实验；归一化仅按完整 CPU／软件身份只读复用。

必须申请 **0 GPU 的 CPU Custom**。这消除本作业的 GPU 闲置回收条件，
不承诺平台不会因其他原因中断，也不承诺 CPU 更快。
共享 Critic 仍是重网络，长轨迹 CPU 更新可能明显慢于 GPU；不能据少量合成状态外推500训练耗时。
CPU 与 CUDA 的浮点和 dropout 随机执行不同，保存迁移来源，不声称逐位等价；
新 B0 仍 train，旧 MLP 行为不变，GAE／每完整episode后3轮AdamW／全25×5／500／3600均保留。
若必须保持 CUDA 执行流，进一步方案是 CPU 采样完成后再申请 GPU 更新，
严格等待更新后再采下一条该策略轨迹；不能混入旧策略轨迹冒充同一 on-policy 更新。
当前无平台自动提交接口，因此尚未实现自动跨队列接续，不能承诺无人值守GPU阶段调度。

## 时间限制的纠正

平台没有总运行时限；原脚本自己有累计172800秒硬上限。
旧 B0 的26分钟由这个脚本上限算出，不是平台余额。
新入口用显式 `--no-total-wall-limit` 关闭累计墙时硬停止，完整保留旧花费，继续记录实际 elapsed。
目标仍是500完整episode，不修改原 frozen 配置文件；资源覆盖另记 recovery-attempt 和 budget。

单实例 SCIP3600秒保持不变。若改成更短，需要新的时限／归一化协议；
旧权重可以作为新实验初始化，但不能把混合时限训练描述为原论文同协议续接。
仅缩短 watchdog 会使未完成任务失败，并不保证有合法轨迹可更新。
已核查到的11月8日 `cpu_timing.py` 55秒／外层60秒仅保护一次离线诊断，不是防回收训练入口。

## 验证与交接

本地检查覆盖两条原协议的缓存身份、CPU导入、AdamW／RNG／游标保持、活动父作业拒绝、
待更新轨迹中断重放只提交一次、重复容器启动不覆盖状态、显式取消总墙时停止及完整原模型CPU桥接。
最终状态看 `CPU_CHECKS.json` 与 `cpu_checks.log`；一次90秒外层检查到限记录另保留，
这个诊断期限没有进入训练配置。

最终两条协议共16项全部通过，套件经过墙时122.650秒（含导入及文件读写）；
此前诊断到限和只读检查计时单独保留，未因此追加真实求解。
原 paper-aligned 的49个、新 trainmode 的53个冻结文件校验无变更。

推荐先提交新 B0 两种子和旧 MLP 两种子的四个 CPU 恢复作业；
两个正在推进的 MLP＋BC 暂不动。完整命令在新包 `commands.txt`。
旧 B0 若保留续训也有入口，但不能替代新 trainmode B0 对照。
本阶段完成准备后停止，由用户手动提交作业。
