# 准备作业结果与正式训练放行（2026-10-27）

结论：**可以创建六个正式训练 Custom 作业，无需重跑 prepare。** 2026-10-27 19:52（北京时间）准备流程落盘完成，19:58复核当前代码/配置及全部目标证据，36项核查全部通过，`ready=true`。本次复核只读取已产生的目标机产物，没有在开发容器运行训练。

## 验证结果

- 两组各8个episode的短程验证通过：各两次有效PPO更新，Actor-only开发求解通过。
- 新原生归一化75/75有效，两组逐条共用同一份原生结果；不是分别测出的两套归一化。实例哈希、cutoff、SCIP设置、种子和时限均匹配。
- 两组CUDA桥接各18项检查全部通过：Actor/Critic实际更新，完整Critic在GPU，CPU采样worker不初始化CUDA；中断恢复后的Actor、Critic、两套优化器、CUDA RNG与参考执行完全一致，复用完整轨迹。
- 两组pilot各覆盖全部20训练实例，完成5次有效PPO更新，无剩余未更新轨迹；最终检查点哈希匹配、模型权重全部有限、优化器和CUDA RNG完整。
- 36项复核通过；准备日志没有Traceback，已完成worker记录没有无效任务。loss全部有限，未触发非有限梯度异常；更新前概率重算误差最大4.77e-7。
- 已重新调用正式训练入口使用的证据检查函数，使用保存的目标节点preflight，确认桥接/pilot/归一化与当前正式协议匹配。当前开发容器没有GPU；六个新节点启动时仍会自行做真实硬件预检。
- 目标环境：Gold 6148、Tesla V100-SXM2-32GB、Python3.8、Torch1.13.1+cu117；两组软硬件哈希相同，初始化后的CPU/CUDA随机流相同。

## 20实例试跑实测

| 指标 | Transformer + BiMatching | BiMatching Only | 相对变化 |
|---|---:|---:|---:|
| 完整episode | 20 | 20 | 0% |
| 有效PPO更新 | 5 | 5 | 0% |
| 总耗时（分钟） | 16.32 | 13.32 | -18.37% |
| GPU学习器更新墙钟（秒） | 157.59 | 107.35 | -31.88% |
| GPU峰值allocated（GiB） | 3.958 | 0.189 | -95.23% |
| GPU峰值reserved（GiB） | 5.375 | 0.293 | — |
| 转移数 | 12,012 | 16,987 | +41.42% |
| 优化器步数（沿用原计数口径） | 147 | 204 | — |
| 500 episode线性耗时估算（小时/作业） | 6.80 | 5.55 | 预测，不是结果 |

无Transformer组本轮总体快18.4%，GPU更新墙钟少31.9%，峰值allocated显存少95.2%，说明工程计算收益已在真实在线采样/GPU更新流程中体现。其转移数反而多41.4%，因此不能把GPU更新墙钟差直接解释为每样本加速，也不能把两组不同轨迹当成完全相同计算量。

**这些数据不能证明分支策略更优。** pilot只是一个训练种子的一轮20实例循环；不同搜索树及候选数使决策数、优化步数不同。尚未进行正式训练后同测试集评估，Nodes/Gap/PDI与端到端求解收益仍待验证。当前loss有限和恢复正确只支持工程放行，不等于500 episode训练稳定性已得到证明。显存也是此轮观测峰值，不缩减两组公平资源申请。

## 六作业申请

两组分别训练种子11/22/33，每作业从头完成500个episode，每100保存，四个CPU采样worker，PPO设置保持冻结值。准备阶段的smoke/bridge/pilot不计入500预算，也不用它们的权重热启动。

每作业申请 **8 CPU / 32 GiB / 1 V100-SXM2-32GB**，CPU仍为**Gold 6148**。可申请12小时，若平台方便也可沿用24小时上限；6.80/5.55小时是初始策略试跑线性估算，不含排队，也不保证后期轨迹增长和六路共享存储下的耗时。

以下每行对应一个Custom作业，六个可以并行。镜像、共享挂载和工作目录沿用准备作业。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 33
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 33
```

不修改code或configs后再启动：这些文件参与协议哈希，改变会使已通过的准备证据不匹配。恢复时确认旧作业退出，使用相同命令，显式恢复latest.pt；不删除任何状态或失败成本。这里只放行正式训练，不自动启动选模或测试。

## 证据

- [36项复核JSON](preparation-verification-20261027.json)
- [Transformer pilot验证](../../../runs/transformer-ablation-20261027/transformer/throughput/verification.json)
- [BiMatching Only pilot验证](../../../runs/transformer-ablation-20261027/bimatching/throughput/verification.json)
- [准备作业完整日志](../../../runs/transformer-ablation-20261027/launcher-logs/20261027T105335-W5YyVD.log)
- [完整命令及训练后的选模/测试流程](../README.md)
