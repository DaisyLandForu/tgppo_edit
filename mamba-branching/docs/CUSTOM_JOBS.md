# MILP-S 单种子 Custom 作业

2026-10-29评测核查：正式475条轨迹、CL50/IL50、验证集选模及全部40对测试/汇总均已完成，单训练种子MILP-S主方法对relpscost闭环已结束。[正式评测报告](../runs/milp-s-seed0-20261027/analysis/evaluation-review-20261029/README.md)包含完整性检查、逐实例结果、论文对照及图表；[训练结果报告](../runs/milp-s-seed0-20261027/analysis/training-review-20261029/README.md)保留选模和GPU复核。当前冻结模型的公平节点及耗时高于relpscost，其他神经基线排名未验证。下文保留运行流程备查，本RUN无需重新提交已完成任务。所有命令使用独立Conda环境，不需要在Custom容器里再次安装依赖。

配置文件：[formal-milp-s.json](../configs/formal-milp-s.json)；逐作业资源和完整启动命令：[custom-jobs-milp-s-seed0.json](../configs/custom-jobs-milp-s-seed0.json)。后者是供 Custom 界面填写的配置清单，不是假定某个平台 API 的提交文件；其中依赖关系需要手动遵守或在平台中配置。

统一选择当前 **scip 镜像**，挂载相同持久目录 **`/home/duweiyue25`**，Shell **bash**。工作目录可填 `/home/duweiyue25/tgppo-transfer`；下面用绝对入口路径，工作目录不会影响数据位置。CPU任务不要申请GPU；GPU任务每个只暴露一张V100 32GB或A100 40GB，脚本使用进程内`cuda:0`，命令不变。

| 阶段 | 作业数与并行 | 每作业资源 | 建议平台时限 | 依赖 |
|---|---|---|---|---|
| 专家采集 | 8 个同时跑，每个 2 个 solver worker | 2 CPU / 8 GiB / 0 GPU | 48 小时 | 初始化完成 |
| 数据校验与索引 | 1 个 | 4 CPU / 32 GiB / 0 GPU | 24 小时 | 8 个采集全部退出 0 |
| seed0 CL→IL | 1 个 | 8 CPU / 64 GiB / 1 V100 32GB | 72 小时，可到期续跑 | 数据索引成功 |
| 冻结模型测试 | 4 个同时跑，每个串行求解 | 4 CPU / 16 GiB / 1 V100 32GB | 48 小时 | 训练退出 0 且 selected.json 冻结 |
| 汇总 | 1 个，也可当前环境执行 | 1 CPU / 4 GiB / 0 GPU | 1 小时 | 4 个评估全部退出 0 |

采样总计16个solver、16 CPU / 64 GiB。8个分片各有59–60条轨迹，单条SCIP上限1小时，2 workers；48小时是平台配额建议，并非实测完成时间承诺。训练的72小时是起步时申请值；实际已建成293,346个有效训练状态、3,130个IL窗口并完成训练。需要恢复历史运行时，重用相同入口和RUN，epoch预算不变。

## 1. 初始化一次

下述目录已在当前共享存储中初始化；命令是幂等的，可以再次核对配置。初始化只冻结配置、源码/环境指纹及任务清单，不会启动采集：

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/custom_job.sh" prepare --run-dir "$RUN" --config "$ROOT/configs/formal-milp-s.json"
```

配置冻结后，运行中的作业会拒绝源码、依赖版本或数据指纹变化。维护代码时保留本次源码快照与原始记录，另建实验或做明确迁移。

## 2. 并行提交 8 个 CPU Custom

每个作业使用下面命令，分别设置 **`SHARD=0,1,2,3,4,5,6,7`**。每个 Custom 只执行一个不同分片号，`--num-shards 8` 保持一致：

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
SHARD=0
bash "$ROOT/custom_job.sh" collect --run-dir "$RUN" --shard-index "$SHARD" --num-shards 8 --workers 2
```

每条轨迹采用 random 前缀 0/1/5/10/15 → relpscost，SCIP 8.0.4，关闭 primal heuristics，不提供 cutoff，时间上限 3600 秒、不设节点上限。共 19×5×5=475 条；solver seed0–3 的 380 条训练，seed4 的 95 条验证。HDF5 随回调流式写入，避免把整小时的轨迹累计在 Python 字典中。

已完成轨迹有 `collection/<任务ID>/done.json` 和 SHA256；重启同一命令会校验并跳过它们。未完成轨迹从该 episode 开头重跑；SCIP 树本身没有中途序列化。失败、超时 watchdog 和进程日志均保留在独立 attempt 目录，不覆盖历史尝试。SCIP 正常达到 3600 秒是可用轨迹；进程异常/硬中断不算完成。

## 3. 数据校验与索引

8 个采集作业全部退出 0 后执行：

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/custom_job.sh" build-data --run-dir "$RUN"
```

缺少任何任务时会非零退出，完整性报告在 `dataset/completeness.json`。成功后生成 `dataset/manifest.json` 和分轨迹索引；不需要把全部轨迹再拼成一个大 HDF5。训练时按索引读取有限数量的文件。

沿用上游数据规则：仅候选数 >1 的状态进入训练，先按完整轨迹划分 train/valid，再切为 100 状态窗口；末尾不足 100 的窗口保留。整条轨迹只有 1 个有效状态时按上游规则不用于 IL，数量单独记录。预处理不会把同一轨迹的不同片段随机拆入验证集。

## 4. 一个 GPU Custom 顺序训练 CL→IL

2026-10-28后续优化入口及断点迁移见[内存缓存/预取/微批加速训练](FAST_TRAINING.md)，推荐使用其中的`train_fast.sh`命令；原入口保留作来源对照。

若共享盘随机读取导致训练变慢，可从原检查点切换到[节点本地数据缓存入口](TRAINING_SPEED.md)。它使用相同冻结训练器，检查点和日志仍写回原RUN；需先退出旧训练，不能同时启动两个训练进程。

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/custom_job.sh" train --run-dir "$RUN" --resume auto
```

这是单个训练种子，CL 和 IL 存在权重依赖，在一个作业内顺序执行。需要分开申请时可先加 `--stage cl`，结束后再用 `--stage il`；两阶段不能同时写同一 RUN。

固定选择如下，均记录在配置中：

- 训练 seed=0；CL 50 epoch、IL 50 epoch，来自上游 main.py 默认值，论文没有明确完整的 epoch/选模细则。为了可独立恢复，CL 网络使用 seed0，IL 初始化使用派生 seed100000，sampler 有独立且可保存的 RNG。
- CL：全部有效训练状态、Adam/lr=1e-4、batch32。上游默认的 `max_clip_samples=200000` 不属于论文明确协议，本次不额外截去数据，差异已记录。
- IL：AdamW/lr=1e-3/wd=.01，逻辑 batch32、100 状态，默认微批1累计梯度。末批使用实际样本数作分母；每个逻辑 batch 统一按上游 max_norm=1 裁剪后更新一次。不截断候选、不缩短窗口，不对 OOM/NaN 静默跳过。
- CL 损失只包含真实候选，去掉无关常数 +2；IL 在 Mamba 前打包真实候选和历史动作，按论文每轨迹求和/轨迹间均值。padding 和因果检查已通过。
- 每个 epoch 遍历全部 seed4 验证样本。CL 选择验证 CL loss 最低的 epoch，其权重用于初始化 IL；IL 选择验证轨迹平均 NLL 最低的 epoch，并保留当时配对的嵌入器和 Mamba。严格相等时取更早 epoch；不根据测试求解表现选权重。

每 25 个逻辑更新及 epoch/阶段边界原子提交 checkpoint，保存权重、优化器、epoch、游标、样本顺序引用及哈希、sampler 和 Python/NumPy/Torch/CUDA RNG、训练统计、最佳验证信息。SIGTERM 在可完成逻辑 batch 后保存并退出 75；硬中断从最近提交点恢复，未提交的更新会重做。恢复命令就是上面的原命令，保留总 epoch 目标和已提交计数；各次尝试日志保留，未提交的计算不会冒充新进度。恢复不保证 CUDA 更新逐位相同。

输出包括：

- `training/cl/latest.json`、`training/il/latest.json`：当前可恢复 checkpoint 指针及哈希。
- `training/logs/*.jsonl`：逐次作业原始日志、更新次数、训练损失、验证损失、显存和失败批次。
- `training/cl/selected.json`、`training/il/selected.json`：阶段最佳验证选择。
- **`training/selected.json`**：两个阶段完整结束后生成的冻结模型入口。绑定最佳 IL epoch 的两套网络、验证指标、数据与配置哈希。测试只读取此文件，不读取最新或最后一个 epoch。

## 5. 冻结后并行提交 4 个 GPU 评估 Custom

这4个作业用于测试冻结模型的实际求解性能。选模已在训练中使用solver seed4验证轨迹完成；4个作业读取同一个`training/selected.json`及同一套最终权重，只将测试任务分为4片，不会训练、更新权重或选择其他checkpoint。测试solver seed0—4属于另8个测试实例，与训练/验证实例分离；测试结果不用于重新挑epoch。

分别设置 **`SHARD=0,1,2,3`**，每个作业1张V100或A100、`--workers 1`：

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
SHARD=0
bash "$ROOT/custom_job.sh" evaluate --run-dir "$RUN" --shard-index "$SHARD" --num-shards 4 --workers 1
```

共 8 测试实例×5 solver seed=40 对，每对顺序执行 relpscost 与冻结 Mamba，共 80 次求解。每个分片 10 对；每次上限 3600 秒。两策略使用同一作业的 CPU/软件/限制，避免把旧机器基线拼进本轮结果；一对失败后整对重试。不同实例/种子对可以并行，单个 GPU 作业内不并发 solver。

评估使用已知最优 cutoff、关闭 primal heuristics、根节点 relpscost、24 步历史加当前状态（49 候选/动作组）。cutoff 仅为目标限制，不冒充观察到的 incumbent；若 SCIP 在 cutoff 下返回 infeasible 且无解，该状态与 `cutoff_proof_without_incumbent` 一起保留。所有异常直接记失败，不作为正常完成结果。

## 6. 汇总与查看进度

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/custom_job.sh" status --run-dir "$RUN"
bash "$ROOT/custom_job.sh" summarize --run-dir "$RUN"
```

`status` 显示已校验的提交数量和缺失任务，不把旧 running 标志当成活进程。`summarize` 在全部 40 对结果齐全且模型一致后生成 `evaluation/summary.json`：保留逐实例/种子记录、状态和节点/公平节点/墙钟/SCIP时间/PDI 的普通几何均值。缺少结果时返回非零，不删除失败项后计算“完整”均值。比较限于本轮硬件；与论文 A100/Gold5218 的时间指标有硬件差异。

汇总后核查全部40对、异常和超时状态，再比较Mamba与同机relpscost的逐实例及总体结果，对照论文并记录硬件、训练细则等差异，形成单训练种子MILP-S复现结论。验证模仿准确率不等于求解性能；无需在这之后再选一轮模型或自动追加训练。多个solver seed是求解重复，不能当作多个独立训练种子。

训练好的模型可以直接参与求解。现有`evaluate`入口已经通过SCIP分支回调调用模型：加载冻结checkpoint中的配套embedding和Mamba（均为IL选中轮次保存的权重），提取候选变量/节点/MIP特征，用当前状态及最多24步历史选择分支变量，随后由SCIP继续搜索和判断终止；整个过程只推理。首次分支仍按本次协议使用relpscost。不能另取CL第44轮embedding与IL第50轮Mamba拼接，IL已经联合更新两套网络。

当前正式`evaluate`入口处理固定MILP-S测试清单。求解自己的新实例时，将相同特征提取和分支回调接到读入该实例的SCIP模型即可，保持匹配的SCIP 8.0.4定制绑定和独立环境；不需要每次先跑完整40对测试。论文测试使用已知最优cutoff和关闭heuristics的特定设置，未知最优值的新问题不能直接照搬该cutoff。

## 入口验证证据

入口验证报告：[entry-check-20261027-final/verification.json](../runs/entry-check-20261027-final/verification.json)。诊断只使用训练实例lseu/misc03，含solver seed4验证数据、2+2 epoch、两次真正的新进程恢复、与连续训练的数值比较、梯度累积等价性、冻结选模、求解和汇总。缺失数据、源码/数据损坏、重复训练及重复提交的检查通过。它验证入口可运行；完整475条和50+50 epoch的正式训练证据另见本文开头的训练结果报告，正式测试证据以本RUN的评估结果为准。
