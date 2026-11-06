# 当前接续断点

2026-10-29评测核查：正式40对/80次求解和汇总全部完成，403项完整性/哈希/指标检查通过；单训练种子MILP-S主方法对relpscost闭环已结束。Mamba相对基线：公平节点1.921倍、耗时1.831倍、普通节点3.030倍。80次均为已知最优cutoff条件下的无incumbent证明结束，不是原问题无解。详见[正式评测报告](../runs/milp-s-seed0-20261027/analysis/evaluation-review-20261029/README.md)。尚未验证其他神经对照排名及多训练种子稳定性；本轮没有启动追加实验。以下为历史接续记录。

2026-10-29核查：正式CL50/IL50已完成，选定CL第44轮、IL第50轮，`training/selected.json`已冻结。IL验证准确率73.70%，每状态NLL0.952592；当前V100完整复算seed4验证结果与A100记录一致。加速作业约3小时45分钟完成；下一步提交4个测试求解评估Custom（SHARD=0..3），尚未生成任何正式测试结果。详见[训练结果核查](../runs/milp-s-seed0-20261027/analysis/training-review-20261029/README.md)。以下均为历史准备与恢复记录，不代表当前待执行步骤。

2026-10-28更新：正式475条轨迹和索引已完成。旧CL作业已受控退出75，latest为102875更新（第12轮、游标64864）；IL未开始。用户授权直接加速，已完成`train_fast.sh`及CPU检查，后续使用[FAST_TRAINING.md](FAST_TRAINING.md)的Custom命令恢复，GPU Mamba数值/容量门禁在启动时执行。原配置/冻结源码不变；实际微批、checkpoint间隔等执行覆盖值单独绑定到`training/fast/execution.json`。以下为原准备记录，不应把旧状态描述当成实时进程证据。

用户已要求通过并行 Custom 完成正式单种子闭环；训练、恢复、验证集选模和评估入口已补齐。提交配置与命令见 **`CUSTOM_JOBS.md`**，逐作业配置为 `configs/custom-jobs-milp-s-seed0.json`。

正式 RUN：`/home/duweiyue25/tgppo-transfer/research/mamba-branching/runs/milp-s-seed0-20261027`。已初始化冻结配置、475 项任务计划、源码/环境/绑定指纹与源码快照；没有替用户提交正式 Custom。以 `CURRENT_STATE.json` 和实际 done.json/checkpoint 为准，不把初始化当作采样结果。

依赖顺序：8 个 CPU Custom（各2CPU/8GiB、2worker、分片0..7）→build-data 校验全475项→1GPU训练（8CPU/64GiB、V10032GB、CL50/IL50、--resume auto）→验证集冻结 selected.json→4 个GPU评估分片→summarize。初始化目录与已完成轨迹/训练进度保留，重启同一命令接续。

训练仅用 seed0–3，选模仅用 solver seed4；8 测试实例只在冻结后用于评估。50/50 epoch 来自上游默认，不宣称论文完整规定；IL 逻辑batch32、微批1梯度累积、100状态，不截候选。CL全部有效状态而不套上游200000 cap等补充选择均在配置里注明。CL选最佳验证embedding，IL选最佳验证时配对的embedding/Mamba；两阶段不能并发写同一RUN。

最新入口验证为 `runs/entry-check-20261027-final/verification.json`：真实小数据4条采样、2+2epoch；CL第2次更新和IL第1次更新后退出并由新进程恢复，最终与连续运行在数值容差内一致；含梯度累积等价、冻结选模、求解汇总、重复锁/缺失数据/损坏数据/源码变化拒绝检查。只使用 lseu/misc03 训练实例，没有用8个测试实例调试。这不是正式475条、50+50epoch或论文性能结果。

独立 Conda 环境和 V100 核函数两节点已验证，无需重复申请调试资源。原 GPU pilot 的100状态/batch32只覆盖小候选窗口；正式入口默认IL微批1降低GPU峰值，完整数据规模仍需实际测量。CUDA更新不保证逐位一致；原失败和成功报告全部保留。历史暂停快照不表示当前仍暂停。
