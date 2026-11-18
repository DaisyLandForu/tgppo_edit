# 48GiB容器内存的GPU恢复入口

每作业4CPU／48GiB内存／1张GPU；显存另算，保持原FP32、microbatch、模型和单次3600秒。
入口调用已验收的paper-recovery，原训练与恢复包均不修改。

新B0-seed22：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resource-resume-20261109/custom_job.sh run --experiment trainmode --variant b0 --seed 22 --device cuda --ram-gib 48 --resume --no-total-wall-limit
```

用户贴出的失败命令实际为seed11；该作业恢复使用：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resource-resume-20261109/custom_job.sh run --experiment trainmode --variant b0 --seed 11 --device cuda --ram-gib 48 --resume --no-total-wall-limit
```

GPU模式恢复原trainmode CUDA分支：seed11目前11、seed22目前12；不从头训练，也不导入CPU分支。
同一条件只启动一个作业。原生分母按实际CPU／软件身份匹配，跨CPU采样保留旧记录另建cohort。
`--device gpu`可作为本入口的cuda别名，但建议写cuda。

64GiB是原容器内存预检门槛，不是模型数学要求。仅在preflight副本中改为48GiB，
原配置／checkpoint哈希、GAE、Actor模式、AdamW、batch／microbatch、500／3600均不改。
worker的32GiB虚拟地址空间限制也不改；它不同于实际RSS或整容器内存。
需要64GiB可改为`--ram-gib 64`，没有自动降到更小容量的行为。

现有新B0-seed22的12份有效采样记录，单worker峰值RSS最大约2.51GiB；
seed11约2.44GiB，旧B0-seed11现存136份采样记录最大约3.52GiB。
这些不是训练父进程、页缓存和所有子进程的总峰值，因此只能支持48GiB是合理的首轮申请，
不能证明整个500训练必然够用。checkpoint文件约110MiB，也不能用文件大小推算总运行内存。

本入口每30秒写容器memory.current／memory.peak／OOM事件及进程RSS证据到：
`research/tgppo/runs/paper-resource-resume-20261109/memory/`。
容器不暴露memory.peak时只保留采样峰值，可能漏掉瞬时峰值。
日志在同一RUN的launcher-logs；内存来源写入原训练environment中的resource_override。

48GiB指容器RAM，和V100／A100显存分开。降低容器RAM不会解除平台两小时GPU闲置回收规则。
本次仅合成资源预检与只读核查，没有真实GPU、训练或新增SCIP求解。

只读校验：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resource-resume-20261109/custom_job.sh check --experiment trainmode
```
