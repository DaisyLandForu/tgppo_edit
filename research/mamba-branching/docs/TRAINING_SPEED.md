# 从现有断点加速训练

用户已授权进一步优化，新增[完整加速入口](FAST_TRAINING.md)：本地缓存后全量加载RAM、向量化组装、有序预取、自动验证IL微批和减少日志/检查点I/O。现在推荐其中的`train_fast.sh`；下文`train_local.sh`保留为只改变输入存储位置的方案说明。

2026-10-28 19:22（北京时间）的正式日志显示：CL 第12/50轮，IL未开始；最初每轮约16分钟，第11轮约3小时13分钟。最近30–60分钟，更新函数计时只占墙钟约3%；这个计时包含前后向、梯度检查和更新，但不包含取样、日志或检查点。因此数据/I/O是优先排查对象，不能直接把其余时间全部认定为网络存储等待，也不能据此承诺更换A100后的加速比。

当前475个HDF5共1,317,245,304字节（约1.23 GiB），加上索引和元数据仍很小。原入口逐样本随机读取共享盘，HDF5句柄的LRU只保留8个文件。可以先把不可变输入复制到节点本地盘；代码、顺序、batch32、100状态、FP32、50+50轮及选模规则全部沿用冻结入口。

## 推荐的下一次 Custom

- 1张V100 32GB、8 CPU、64 GiB内存；本地临时盘至少5 GiB。若有A100也可用，但缓存带来的收益应先实测。
- 镜像、独立Conda环境、共享挂载保持现有配置。缓存默认写`/tmp`；需确认它是节点本地盘。也可指定有足够容量的`/dev/shm`，其占用计入容器内存。
- 一个RUN仍只允许一个训练进程。旧作业还在跑时，新入口会检查同一把共享训练锁并退出，不会启动第二个训练器。

先在旧Custom节点内向训练进程发送SIGTERM，或使用会发送SIGTERM且给予保存时间的平台停止方式。原训练器完成当前逻辑batch后保存，记录`checkpointed_stop`并退出75。平台若直接强杀，则从最近已提交的25步检查点恢复，未提交更新重做。新入口不会替用户停止旧作业。

旧训练退出后，新Custom使用：

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/train_local.sh" --run-dir "$RUN" --cache-root /tmp/mamba-branching-$UID
```

若先测吞吐，可给上述命令加`--stop-after-updates 500`；这500步是实际训练，会保存进度并受控退出75。随后去掉该参数继续，不重复已提交更新。比较相邻更新的墙钟时间，排除首次环境导入、数据复制及启动校验；仍须把这些启动成本计入本次作业总耗时。

只提前缓存、暂不训练时，加`--prepare-only`。该模式允许旧训练继续运行；会产生一次约1.23 GiB的顺序读取。若已停止旧训练，也可以先缓存再用不带该参数的同一命令启动。

入口逐文件复制并校验SHA256，复用缓存时重新校验缓存；源配置、任务计划和冻结源码仍会检查。缓存目录中的`training`是指向原RUN的符号链接，所以检查点、日志、训练锁、采样顺序、最佳权重及最终`selected.json`始终持久写回原目录。原训练器还会再次校验缓存数据、环境与检查点；没有关闭或重写来源检查。启动信息保存在`training/local_launches/`。

本地缓存被回收、换节点或回到原始`custom_job.sh train --resume auto`都可以从同一共享检查点恢复。评估和汇总继续使用原RUN。不要把整个训练目录复制到临时盘后再等待训练结束时同步，因为抢占时会丢失进度。

## 后续优化顺序

1. 比较缓存前后的完整step墙钟，确认数据搬到本地后的收益。原日志`seconds`仅为更新函数耗时，不能当成完整step时间；可以用相邻`update.timestamp_utc`之差，并包含验证/检查点总耗时一起评估。
2. 如果本地盘仍受逐状态读取的CPU开销限制，再考虑全量NumPy内存缓存和按批读取；预取时只提交实际消费的数据游标，并验证恢复后样本顺序与损失。该类改动要做来源迁移与等价性验证，不能直接修改本次冻结的`formal_data.py`。
3. IL当前微批1、累积32次。进入该阶段后，可评估微批4/8及A100，但需要验证真实数据峰值显存、梯度累积数值和检查点配置迁移。不能直接改现有RUN/config.json。

缓存方案的CPU/文件一致性验证见`setup/local-training-check-20261028.json`。当前开发容器没有GPU；该入口尚无Custom上的CUDA续跑和实际训练加速比证据，不能将CPU验证当成正式单种子训练完成。

2026-10-28 19:34（北京时间）验证通过：15项复制/损坏/配置/并发锁/原入口转发检查；真实诊断数据中train/valid各128个CL样本、各4个IL窗口，在原入口和本地缓存间逐元素完全相同。报告为`setup/local-training-data-probe-20261028.json`。该小样本单次测量中CL读取为1.113→0.080秒（train）、0.322→0.081秒（valid），IL读取约快1.25–1.79倍；仅说明当前节点上的抽样I/O差异，不能承诺正式475条、GPU训练或其他Custom节点会达到相同倍数。诊断没有使用8个测试实例。

仅测数据读取时，可以在同一节点运行下面的CPU探针。`VIEW`取缓存准备输出里的`local_view`，不要填共享RUN本身。它比较同一随机索引下CL/IL、train/valid样本的逐元素一致性；小规模抽样读速不能直接当作完整训练加速比。

```bash
source "$ROOT/scripts/runtime.sh"
VIEW=/tmp/mamba-branching-$UID/缓存输出中的目录名
"$MAMBA_PYTHON" "$ROOT/scripts/benchmark_training_io.py" \
  --run-dir "$RUN" --local-view "$VIEW" \
  --report "$RUN/training/local-io-probe.json"
```

参考：[h5py数据集与分块读取](https://docs.h5py.org/en/stable/high/dataset.html)、[PyTorch性能调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。
