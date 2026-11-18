# GPU启动检查后续训

用于反复出现`uncorrectable ECC error encountered`的GPU续训。入口先记录Pod、GPU UUID、PCI地址、驱动和ECC／页退役信息，再执行一个标量CUDA分配与加法并同步；通过后以原参数调用`paper-resource-resume-20261109/custom_job.sh`。原训练入口及checkpoint保持原样。

每个Custom申请4CPU／48GiB／1GPU，优先由平台安排经过健康检查的节点。seed11：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-gpu-health-resume-20261113/custom_job.sh run --experiment trainmode --variant b0 --seed 11 --device cuda --ram-gib 48 --resume --no-total-wall-limit
```

seed22可只替换`--seed 22`；CPU模式和status/check直接转发原入口。日志保存到`research/tgppo/runs/paper-gpu-health-resume-20261113/`，实际训练仍写原训练目录，并继续产生原resource日志。检查不会加载、修改生产模型；原来的目标设备完整自检仍会执行。

CUDA检查失败时不进入训练。该入口负责记录与筛查，GPU恢复／隔离仍需平台处理。历史累计ECC非零不自动判失败；单个标量检查成功也不能保证整卡长时间运行健康。20秒设备查询和180秒探针时限只作用于启动检查，原3600秒求解时限／500 episode与总预算历史均不改变。

本地已用替身命令验证六条控制流（参数／退出码转发、CUDA失败阻止训练、设备枚举失败、可选ECC查询不支持、CPU与status旁路），未在本地调用真实GPU或训练。证据见`research/docs/diagnostics/tgppo_20261113_b0_ecc_recurrence/CPU_CHECKS.json`。新GPU上的真实设备检查由用户提交Custom后执行。
