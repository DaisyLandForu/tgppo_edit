# 加速入口与现有断点续训

2026-10-29结果更新：A100真实Mamba微批8门禁及正式训练已完成，CL/IL均50轮，模型已冻结。当前V100完整复算seed4验证指标与训练记录一致。详见[训练结果核查及下一步评估命令](../runs/milp-s-seed0-20261027/analysis/training-review-20261029/README.md)。下面的停点和GPU待验证描述是2026-10-28实施时的历史状态；无需再启动训练。

用户于2026-10-28授权内存预加载、批量读取、有序预取及IL微批优化。使用新入口`train_fast.sh`，原冻结源码、原RUN/config.json、run.json和数据清单不修改。旧Custom已经记录`checkpointed_stop`/退出75；当前可恢复点为CL更新102875，CL第12轮游标64864。以实际latest.json为准。

## Custom配置与命令

当前scip镜像，bash，相同`/home/duweiyue25`持久挂载，8 CPU、64 GiB内存、1张A100 40GB或V100 32GB，节点本地临时盘至少5 GiB。一次只启动一个训练作业；CL和IL顺序执行。作业配置清单见`configs/custom-jobs-fast-milp-s-seed0.json`，不是平台API提交文件。

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
bash "$ROOT/train_fast.sh" --run-dir "$RUN" --cache-root /tmp/mamba-branching-$UID
```

新入口自动恢复，不需要重新采集、build-data或初始化RUN。后续重启使用同一命令。仅准备本地数据时加`--prepare-only`。若需有限试跑，加`--stop-after-updates 500`；会保存实际新增的训练进度并受控退出75，去掉参数即可继续。

## 已实现的执行优化

- 校验并缓存约1.23 GiB不可变输入到本地盘；一次批量读取全部HDF5数据，转为连续NumPy数组，实测常驻数组1,260,069,532字节（约1.17 GiB）。CL/IL共用这些数组，训练中不再打开轨迹文件。
- CL按批构造张量；IL用NumPy批量生成候选、动作、位置和mask，候选及历史动作顺序与原适配器一致，不改变100状态窗口。
- 单个CPU线程预取2个逻辑批次，只进行无随机数的读取和组装。只有主训练器在更新成功后推进游标；未消费的预取批次可以丢弃并在恢复后重建。
- IL微批默认`auto`：启动时尝试8→4→2→1。用真实窗口比较原微批1与候选微批的权重、优化器和损失，参数/优化器张量按32×float32 eps的尺度误差核对；另用最大候选宽度与最长窗口组合的合成输入检查容量。失败的探针丢弃并记录，不跳过正式数据。选定微批后持久固定；逻辑batch仍为32，损失分母和累积后统一clip norm=1不变。
- 合并梯度/权重有限值检查，减少CPU等待GPU的次数；保持FP32、TF32关闭，优化器、学习率和50+50轮不变。
- JSONL日志保持打开并逐行写入，控制台每100次更新打印一次；异常、验证和提交事件仍立即输出。
- 检查点由原每25步改为每250步或逻辑批次完成后检查到距上次提交达到60秒时保存，并保留epoch边界/SIGTERM保存。SIGKILL可能重做最多249个未提交更新；60秒不是单个超长batch期间的硬计时器。

## 审计与恢复

新执行设置及新增源码SHA256保存在`training/fast/execution.json`，源码快照保存在相邻`source_snapshot/`。checkpoint额外绑定该执行清单的SHA和实际IL微批；旧checkpoint首次迁入会记录`execution_migration`，保留优化器、所有RNG、采样顺序、epoch、游标和最佳验证结果。原配置仍记录历史执行参数，实际覆盖值以这个新增清单为准。

GPU数值和容量探针结果保存在`training/fast/il-microbatch.json`及`training/fast/gpu/`。相同GPU的同一冻结设置可复用探针；换GPU时重验已选微批。迁入后继续使用`train_fast.sh`；改变源码或执行参数需要明确迁移，入口不会悄悄重置运行记录。

`training`始终链接到原共享目录，检查点、训练锁、日志和最终选模均持久化。评估和汇总仍使用原RUN及原命令；两个阶段完成后才冻结`training/selected.json`，它会额外记录加速执行清单SHA。8个测试实例没有用于调试和选模。

## 验证边界

见`setup/fast-training-validation-20261028.json`：全量475条轨迹成功预加载；抽样train/valid各128个CL状态、各8个IL窗口的数据及collator与原入口逐元素一致；预取顺序、放弃预取后的恢复、RNG保持和异常传递检查通过。

CPU控制流测试使用真实TreeGate及因果GRU替代Mamba，验证原CL checkpoint迁移到快入口、IL中断/恢复，以及与连续原入口的权重/优化器/RNG/游标比较。CL参数尺度误差0，IL最大误差约2.58e-6，小于32×float32 eps。这不是Mamba CUDA验证；开发容器目前没有GPU，真实Mamba的GPU数值与容量检查由上述Custom命令自动执行，通过后继续正式训练。尚未测得新入口的正式GPU训练加速倍数。
