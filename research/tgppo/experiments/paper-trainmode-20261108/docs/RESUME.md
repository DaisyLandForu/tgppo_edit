# 同一实验支持断点续训

“拒绝从旧模式检查点续训”是本项目用于防止实验设置混用的检查，**不是论文规则，也不禁止同一实验的断点续训**。

新副本开始一次从头的正文对照；该次训练运行多久、被释放几次，都可以在原配置和剩余预算内恢复。Actor恢复后继续train更新，rollout继续eval。

## 新实验的恢复命令

CPU：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant b0 --seed 11 --resume
```

CUDA：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 11 --device cuda --resume
```

其余条件替换variant为mlp／mlp_bc、seed为11／22。保持最初选择的设备和协议。

## 旧实验也继续支持原模式恢复

旧paper-aligned脚本不改；若只想继续此前已运行的eval更新训练，仍使用它自己的入口：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant b0 --seed 11 --resume
```

这会沿用旧模式，不能用来宣称新train模式实验已完成。技术上可以另做旧权重切换训练模式的研究，但那是中途切换设置，不是这次从头对标的主实验。

## 恢复行为

| 中断位置 | 恢复行为 |
|---|---|
| 已提交episode后 | 从latest恢复Actor、Critic、AdamW、随机状态、episode游标；进入下一个episode |
| PPO更新中 | 从更新前pending恢复完整状态和轨迹，重做未提交的更新；不叠加中断进程里的部分AdamW步骤 |
| SCIP求解中 | 保留已提交权重与进度；重跑未完成的该episode，不能恢复SCIP进程中的搜索树 |
| 原生归一化中 | 已完成且身份匹配的缓存复用；未完成项保留失败记录并显式恢复 |

同一设备／软件／配置下，检查覆盖模型、优化器及RNG的精确恢复。跨CPU/GPU或不同硬件不承诺逐位相同；新脚本不允许续训时静默更换设备。

恢复不重置累计48小时上限。若预算已用尽，或强杀导致计时缺失，先核对实际成本；不要删除budget／pending／latest来绕过记账。入口会拒绝重复并发作业，已完成组不会重新训练。
