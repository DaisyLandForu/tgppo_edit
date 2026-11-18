# 六模型按最新检查点恢复：seed11 V100，seed22 CPU

准备日期：2026-11-16。用户手动提交 Custom；本地只做迁移与离线检查，未启动正式训练或 SCIP 求解。

启动命令见 [commands.txt](commands.txt)，一行一个作业。seed1/seed2分别是训练种子11/22。

| 方法 | seed11：V100，4 CPU，48 GiB | seed22：CPU，4 CPU，64 GiB |
|---|---:|---:|
| 新 B0 | 已提交164，下一个165 | 已提交218，下一个219 |
| MLP | 已提交284，先重放285的待更新轨迹 | 已提交172，下一个173 |
| MLP＋BC | 已提交284，先重放285的待更新轨迹 | 已提交271，下一个272 |

新结果写到 `research/tgppo/runs/paper-device-resume-20261116/{gpu,cpu}/training/`。原始来源和完整哈希保存在各条件 `device_continuation.json`；原目录、固定100/200权重和评测保留。后续中断重复本次启动命令，接新副本的最新状态。

不能仅在旧命令切换 device：旧GPU的MLP/BC落后于CPU续训分支；旧CPU的新B0也落后于最新GPU。此次六份已经独立导入并逐字段核对，避免退回旧进度。

## 实施的加速

CPU更新采用 `--learner-threads 4`，仅在同步 `Agent.learn()` 内设置 Torch 计算线程，退出或异常后恢复单线程。原代码每次 `seed_all()` 都会重新固定1线程，单独改 OMP/MKL 环境变量或启动时设置线程数无效。GPU作业仍使用1个CPU计算线程配合CUDA更新。

本地4.01核配额，对现存MLP-s11待更新轨迹中候选数731/1039/1145的完整状态分别做前向/反向微基准：1→4线程约3.39/2.67/3.47倍。详细数据见 [THREAD_BENCHMARK.json](../../../docs/methods/tgppo_device_resume_20261116/THREAD_BENCHMARK.json)。这是大候选状态的神经网络计算，不是整个PPO更新或500轮训练；不同规模、CPU和8线程尚待目标环境计时确认。

当前建议先用上述4核配置；CPU线程优化已能利用4核。若8核供给更容易，允许 `--cpus 8`，但不会自动增加采样并发。实际设置 `--learner-threads 8` 才会使用8个CPU更新线程；尚无8核实测，不建议六组一起切8线程。GPU更新主要受GPU影响，给这三作业加到8核没有已确认收益。

## 进程、线程和锁分别能做什么

- **单个SCIP搜索**：现有调用是一个求解进程，采样顺序依赖前面的分支决策。当前SCIP6配置没有用并发求解模式。给进程池设8个worker不会拆开这一棵搜索树。
- **PPO更新**：Torch内部计算线程是当前实现且检查过的加速入口。没有并行调用optimizer.step，没有改变batch256、微批8、梯度累积顺序或3轮更新。
- **多个episode并行采样**：会让几条轨迹来自同一旧策略，再批量更新，或引入过期策略轨迹；这改变现行“一个完整episode后立刻更新”的协议。锁只能避免同时写文件，不能使这些轨迹自动变成正确的on-policy数据。本轮不启用。
- **独立归一化、评测**：适合进程池按实例并行，缓存按完整CPU/软件/solver/实例/种子身份加锁。新入口复用身份完全匹配的历史归一化缓存；没有自动追加预计算或与当前采样争抢CPU。
- **训练互斥**：每个新条件一把execution.lock，并锁住该条件两个旧设备分支。六个独立条件之间可以并行；同一权重不允许多作业同时更新。无需清除锁文件。

## 保持的训练语义和限制

维持500完整episode、每次SCIP3600秒、4200秒子进程watchdog、25实例×solver0—4、原奖励/GAE/AdamW、逐episode3轮更新。新B0保持trainmode，轻量模型保持既有aligned实现且无Actor dropout/BN。BC不重新执行。

复制保留全部Actor/Critic、两个AdamW、policy/update计数、episode游标、已有随机状态、待更新轨迹和历史累计计时；只转换设备标记与待更新轨迹的清理目录。旧轨迹不会被新训练清理。跨CPU/GPU或改变线程数可能改变浮点计算和后续轨迹，不能称逐位等价续训。CPU来源没有CUDA RNG时不伪造它；原训练函数在每个PPO更新前按episode专用种子初始化随机流。

无累计48小时脚本上限；保留历史预算账本。硬件错误、平台外部终止或两小时GPU空闲回收仍可能打断任务。GPU入口记录UUID/ECC并执行原有目标设备合成更新检查，失败不会进入生产训练；不通过空跑占用GPU规避回收，不修改3600秒求解协议。故障后从本次新入口再次续训。

## 已完成核验

六份CPU只读全payload核对通过；两份pending的1557/849条transition及隔离清理目录通过；重复导入、源分支互斥通过。aligned与trainmode各完成一次4线程CPU合成更新/精确重放检查，验证线程正常/异常恢复、CPU入口覆盖原hardcoded cuda检查路径；未在本地执行CUDA。GPU目标检查由用户Custom启动后完成。

核验材料：[CHECKS_aligned.json](../../../docs/methods/tgppo_device_resume_20261116/CHECKS_aligned.json)、[CHECKS_trainmode.json](../../../docs/methods/tgppo_device_resume_20261116/CHECKS_trainmode.json)。原冻结代码的验证仍在每次启动执行。

仅看状态（不训练）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-device-resume-20261116/custom_job.sh status --variant mlp --seed 11
```
