# TGPPO 离线评分头对照：4 个并行 Custom 作业

本包用于已验收的[单步 PPO 弱响应结果](../../../docs/diagnostics/tgppo_20261102_ppo_replay/REPORT.md)之后的下一轮机制验证。用户明确选择“离线评分头对照”。**这里只训练评分头，复用既有轨迹和优势；不新增 SCIP 求解、不训练 Transformer/BiMatching/Critic、不触碰原基线。**

实验问题：在完全相同的冻结 BiMatching/树表示和真实 GAE 优势上，原 TreeGate 评分头与简单线性读出，哪个能将学习信号转成候选相对分数变化？这是条件可学习性对照，不是新的求解性能实验。

| 作业 | 评分头 | 来源种子 | 训练上限 | 资源 |
|---|---|---:|---:|---|
| original-s11 | 原 TreeGate 结构 | 11 | 20 个 head-only Adam 步 | 1 CPU / 8 GiB / 0 GPU |
| linear-s11 | BiMatching 输出直接 Linear(256,1) | 11 | 同上 | 同上 |
| original-s22 | 原 TreeGate 结构 | 22 | 同上 | 同上 |
| linear-s22 | BiMatching 输出直接 Linear(256,1) | 22 | 同上 | 同上 |

四个作业可同时提交，总计申请 4 CPU / 32 GiB / 0 GPU，**本轮无需 V100**。每个作业内部硬限制 600 秒墙时；Custom 平台时限建议填 15 分钟，给启动和日志退出留余量。四作业总计算上限是 4×600 秒，不是共享上一阶段已关闭的 600 秒账本。

**真实四作业未在本地运行。** 本地准备和测试状态以 [docs/PREPARATION.md](docs/PREPARATION.md) 为准；目标 Custom 作业成功不能由本地测试代替。

完整填写步骤和四条独立命令见 [docs/CUSTOM_JOBS.md](docs/CUSTOM_JOBS.md)。[configs/custom-jobs.json](configs/custom-jobs.json) 是 Custom 界面填写清单，不是假定存在平台提交 API。

## 配对原则

- 每种子均取原 B0 ep100 的同一份冻结表示。训练使用上一轮已核验的 256 条同版本真实 transition 及其在完整 episode 缓冲中标准化后的优势；另有 164 个训练状态只作监测，不用于更新，也不是未见实例验证集。
- 两组共享同一份缓存、动作、旧行为 log_prob、优势、逻辑 batch256、microbatch16、Adam/lr=0.00024、clip=0.16、entropy=0.003、梯度上限0.5。
- **统一重置最后的标量读出为零，并使用 fresh Adam。** 原头其余中间层保留 ep100 权重；线性头全部为零。初始动作分布因此完全相同且均匀，并检查与保存行为概率一致。这是在统一诊断初始化下比较评分头，不是原 B0 的原样续训，不混用上轮 Adam 动量。
- 原头约 241,315 参数，线性头 257 参数。容量、门控和深度差异都是整个评分头替换的一部分；结果不能唯一归因于 LN2，不能当成参数匹配的新方法优越性。
- 第一步零读出使前面头层的梯度暂时为零，这是共同初始化的预期性质，不能据第一步就宣布这些层断开。固定记录 step0–20 的全过程，不按监测结果挑步数或 checkpoint。

方案细节与反证条件见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 快速命令

同一持久挂载目录 `/home/duweiyue25`、现有 scip 镜像、Shell=bash；默认复用 `/home/duweiyue25/conda/envs/tgppo/bin/python`，不在共享环境安装依赖。

```bash
# 只读模型/轨迹并生成固定表示缓存；没有 optimizer.step
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh prepare
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh preflight
```

然后分别提交四个 Custom，每个只执行对应的一条 `run JOB_ID` 命令。四个均退出 0 后再运行 `summarize`，详见 Custom 文档。

默认输出目录：`research/tgppo/runs/offline-head-probe-20261102`。所有作业共享只读 prepared 缓存，各写自己的 jobs/JOB_ID；进程锁防止同一作业重复运行，已成功作业重提只核验后跳过。失败或中断会保留预算、checkpoint、inflight 和原日志，并拒绝静默清零重跑；不要删台账或伪造完成标志。

## 文件

- [protocol.json](protocol.json)：共同实验和预算；[configs](configs/)：四个作业的配置。
- [custom_job.sh](custom_job.sh)：Custom 前台入口；[code/runner.py](code/runner.py)：准备、作业、监控、完整性汇总。
- [code/common.py](code/common.py)：缓存/配对初始化/PPO 目标/微批；frozen_model 保留原 Actor/modules 的独立逐字节副本。
- [tests/test_head_probe.py](tests/test_head_probe.py)：缓存重放、非均匀 mask/置换、梯度累计、FP64 Adam、步数、损坏和重复保护、超时/平台中断。
- [docs/NUMERICS.md](docs/NUMERICS.md)：浮点与微批的检查边界。

全部四作业通过完整性汇总后停止，提交结果验收；不自动进入在线采样、正式训练或测试集求解。
