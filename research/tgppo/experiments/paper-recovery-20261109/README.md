# 11月9日作业恢复

入口 `custom_recover.sh` 支持两条已有实验：`--experiment aligned` 是旧 paper-aligned，
`--experiment trainmode` 是 Actor 更新 train／标准 GAE 的新实验。
入口始终显式恢复，不重新初始化；原训练包与已有权重保持不改。

## 推荐：四个已停止条件改为 CPU 独立续训

每条命令一个 **4 CPU／64GiB／0 GPU** Custom。必须申请 CPU 作业；在 GPU 作业里隐藏 GPU
仍可能被平台回收。CPU 更新不保证比 GPU 快，但避免 CPU 求解期间占着 GPU 被两小时回收。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh run --experiment trainmode --variant b0 --seed 11 --device cpu --no-total-wall-limit
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh run --experiment trainmode --variant b0 --seed 22 --device cpu --no-total-wall-limit
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh run --experiment aligned --variant mlp --seed 11 --device cpu --no-total-wall-limit
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh run --experiment aligned --variant mlp --seed 22 --device cpu --no-total-wall-limit
```

本次核查两个 MLP＋BC 仍在推进，保留它们，不同时创建其 CPU 分支。
以后确认原条件停止后，可把 aligned 命令的 `--variant` 改为 `mlp_bc`，种子相应选择11／22。
旧 B0-seed11 若继续，使用 `--experiment aligned --variant b0 --seed 11`；它不替代新 B0。

CPU 首次启动在独立目录导入原模型、两套 AdamW、游标、RNG 与有效 pending；
只改变配置中的 learner_device 并重新计算配置哈希。原始检查点、轨迹、日志、预算不删除。
已有 pending 直接更新；没有完整轨迹的 episode 才重采，保留每次3600秒。
父实验仍运行或计时记录未结算时拒绝首次导入，不自动取消任务或修补不明账目。
同一 CPU 分支再次中断，重复相同命令恢复自己的 latest，不再从父实验重导入。

CPU 输出：

```text
research/tgppo/runs/paper-recovery-20261109/cpu-aligned/training/<variant>-s<seed>/
research/tgppo/runs/paper-recovery-20261109/cpu-trainmode/training/<variant>-s<seed>/
```

`cpu_continuation.json` 保存父检查点／预算 SHA 与迁移来源。
CPU／CUDA 的浮点计算及 dropout 随机执行不同，不承诺与持续 GPU 训练逐位等价；
这也是独立保存的原因。后续评测须使用明确的续训来源，不自动混入原冻结评测清单。

## 继续使用 GPU

使用同一入口，把 `--device cpu` 改为 `--device cuda`，申请一张 GPU。
该模式在原实验目录恢复；自动带 `--resume`，仅 CPU 型号改变时为未完成采样另建 cohort。
保留原记录，软件／绑定／任务／策略或动作种子不一致仍会拒绝。
重复作业会被执行锁拒绝，不覆盖另一个容器的状态。

该模式修复启动／跨节点错误，**不会保证 GPU 不被闲置回收**。
要保持 CUDA 更新且彻底分开资源，需另做 CPU 采样／GPU 更新的阶段调度；
目前没有平台自动提交接口，不能把手动反复重提描述为已解决回收问题。

## 三种时间限制

- `--no-total-wall-limit`：显式关闭脚本累计48小时硬停止，保持耗时台账，目标仍为500 episode。
  不带此项继续保留原48小时限制。该选项与平台是否有总时限无关。
- SCIP 每实例3600秒：保持不变。缩短它会改变训练协议与归一化，不能当同协议续训。
- watchdog4200秒：用于拒绝卡死／不完整结果，不能保证产生完整轨迹或触发 PPO 更新。

11月8日 `training_monitor_20261108/cpu_timing.py` 的55秒截止／60秒外层保护
只约束一次 CPU 诊断，不是训练防回收脚本。CPU-only 训练与“缩短 CPU 求解时限”是两件事。

## 只读检查

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh check --experiment aligned
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-recovery-20261109/custom_recover.sh check --experiment trainmode
```

准备阶段仅合成 CPU 更新／恢复检查，无真实训练、SCIP 求解或 Custom 提交。
