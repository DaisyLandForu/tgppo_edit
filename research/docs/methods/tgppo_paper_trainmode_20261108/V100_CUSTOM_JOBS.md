# V100 六作业启动说明

2026-11-08。用户选择V100；不增加训练脚本，不修改已冻结训练包，不自行提交作业。

## 2026-11-08 MLP重训核验后的资源优先级

当前11月6日MLP／MLP＋BC的GAE和逐episode更新已与新包相同，Actor本身无dropout；CPU局部更新对照逐位一致。**不必仅为train/eval变化重跑四组MLP。优先提交新B0两种子，旧MLP按原入口继续500。** 下列六条仍是完整统一硬件重训选项，不代表六组都必须重跑。详见[核验与决策](MLP_RETRAIN_DECISION.md)。

使用新`paper-trainmode-20261108/custom_job.sh`已有CUDA入口。**必须带`--device cuda`**，因为Python入口默认CPU。B0／MLP／MLP＋BC各两个种子11、22；每行一个Custom。

每作业：1张V100（优先32GB）、至少4个专用CPU核、64GiB容器内存、平台时限48小时。CPU型号不限，尽量保持六组软硬件配置一致。环境沿用已准备的Python3.8／torch1.13.1+cu117／SCIP6.0.1与定制绑定；共享挂载`/home/duweiyue25/tgppo-transfer`。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 11 --device cuda
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant mlp --seed 11 --device cuda
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant mlp_bc --seed 11 --device cuda
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 22 --device cuda
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant mlp --seed 22 --device cuda
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant mlp_bc --seed 22 --device cuda
```

中断后使用原命令加`--resume`：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 11 --device cuda --resume
```

六组按正文与标准GAE，从新初始化开始；旧paper-aligned训练与预算保持原样。支持新实验内部恢复，不从旧eval更新的PPO检查点接入。每条件累计48小时，到限不自动清零或扩额；500完整episode仍是训练目标，不保证48小时完成。

入口不限制为A100；启动时核对实际CUDA、CPU／内存配额，并在真实V100上验证合成PPO更新与恢复，失败即停。此前只完成本地CPU检查，不能称真实V100检查已执行。本轮交接没有启动GPU、真实训练或SCIP求解。

命令纯文本见[V100_commands.txt](V100_commands.txt)。输出位于`research/tgppo/runs/paper-trainmode-20261108/training/`。CPU求解期间GPU空闲仍可能触发平台回收，换为V100不会消除这一规则；恢复不会重复已提交更新。
