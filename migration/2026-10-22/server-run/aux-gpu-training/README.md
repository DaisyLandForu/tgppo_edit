# 独立辅助 GPU 训练

此入口是额外授权的临时工程训练：用随机生成的输入和固定教师 MLP 训练一个独立学生 MLP，执行真实 CUDA 前向、反向和 Adam 参数更新。默认运行 30 分钟后自动保存并结束，最长允许 240 分钟；没有 idle 等待或虚构的训练 heartbeat。

它不检查 CPU 品牌/型号，在 Silver 4110、Gold 6148 等节点都可运行。默认必须有 CUDA，无 GPU 时明确报错，不静默改到 CPU。它不导入 TGPPO、SCIP 或定制绑定，不读真实实例、正式模型或冻结结果，不调用原实验启动脚本。不修改现有实验代码、GPU/CPU 检查或资源上限。Crater 的调度仍由平台决定，此入口不能保证下一次申请分到 Gold 节点。

镜像沿用 `scip (user-duweiyue25/envd:11260740-c4da)`，Shell 选 bash，挂载 `/home/duweiyue25`，工作目录 `/home/duweiyue25/tgppo-transfer/research`。本辅助任务建议申请 **2 CPU / 8 GiB / 1 V100 / 45 分钟**；原正式评测继续使用原来的 8 CPU/32 GiB 配置。默认网络很小，目标是有上限的工程训练，不将它的吞吐当作正式 Actor 的性能。

每个 Custom 作业可填写同一条命令；每次都创建独立输出目录：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/aux-gpu-training/run.sh
```

指定时长/种子例如：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/aux-gpu-training/run.sh --minutes 30 --seed 22
```

输出在 `tgppo/runs/server6-20261022/aux-gpu-training/`：每次唯一的目录内保存 `config.json`、`metrics.jsonl`、`latest.pt`、`final.pt` 和 `summary.json`，启动日志为同父目录 `launch-*.log`。约每 30 秒记录实际 step、loss 和显存峰值，每 120 秒保存最近状态。SIGTERM/SIGINT 后尽量完成当前步并保存，SIGKILL 无法保证末次保存。辅助检查点没有接入正式实验，也没有在该入口实现续训参数。

停止不需要修改代码，可在 Crater 终止对应辅助作业。辅助结果及成本单独记录，不能计入 B0/R0/Rjoint 正式训练预算、开发选模、保留测试或阶段 5 方法收益；不会启动阶段 6。

当前开发容器无 GPU，本地只验证 CPU 小规模前向/反向、参数变化、输出隔离和缺少 CUDA 时的明确拒绝。目标 GPU 实际通过应以新 Custom 的 `AUX_TRAINING_PROGRESS` 和实际日志为准，不能把脚本存在当作 CUDA 验证成功。
