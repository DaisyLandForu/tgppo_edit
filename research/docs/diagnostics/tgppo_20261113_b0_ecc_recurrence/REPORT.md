# B0_new seed11重新申请后ECC复发

核查日期：2026-11-13，北京时间。用户引用的原续训命令参数正确；新的失败仍是GPU向CUDA返回不可纠正ECC，而非检查点恢复代码产生的训练错误。

## 本次与前次事件

| 阶段 | 时间（北京时间） | 事实 |
|---|---|---|
| 前一训练attempt | 2026-11-12 23:15:38 至 2026-11-13 08:50:03 | 从152推进到162，随后主进程在SCIP采样中收到外部SIGTERM15 |
| 本次新attempt | 2026-11-13 08:52:08 至 2026-11-13 08:53:21 | 启动target_checks的临时Agent向CUDA搬运时返回uncorrectable ECC；尚未生产latest加载或新PPO |

前次SIGTERM距最新162检查点提交约119.76分钟，与用户所述2小时GPU闲置回收规则时间吻合；仅共享程序日志仍不能确认信号发送者。本次新Pod为`sg-duweiyue25-261113-08f04-default0-0`，环境为Gold6148／V100-SXM2-32GB／4.01CPU／48GiB；与前次ECC作业型号相同，原日志没有物理节点名与GPU UUID，因此不能认定又分到了同一坏卡，也不能用新的Pod ID证明换了物理GPU。

## 检查点

seed11实际已提交162/500，下次从163继续；不存在pending.pt。最新SHA为`ec6cab2d0dd0e559e37b772c31790c14985202260d1476bb98c3e117832e880c`，mtime为2026-11-13 06:50:18，早于本次失败。已在CPU只读加载，配置SHA、有限模型／AdamW、四随机流和Actor已提交摘要全部通过，读取前后SHA不变，执行锁空闲。本次失败没有覆盖生产检查点，不能再建议从旧152或更早CPU分支继续。

内存采样OOM与oom_kill均为0；本次报错发生在GPU启动，增加容器RAM、修改PPO参数或缩短SCIP时限没有解决该ECC的证据。原四包49／53／6／6份冻结源码哈希均通过。

## 已补充的启动入口

[独立GPU检查入口](../../../tgppo/experiments/paper-gpu-health-resume-20261113/custom_job.sh)记录Pod与可见GPU UUID／PCI地址／驱动／ECC，再做一个标量CUDA分配、加法与同步；通过后原样转发到既有resource入口。该探针在独立进程中退出，不初始化训练Agent、不触及生产checkpoint或随机流；失败不会进入原训练。原完整目标设备自检仍保留。

探针只用于筛查和留证，不能修复卡；历史累计ECC非零不直接拒绝。其180秒限额只约束启动探针，不改变训练500、SCIP3600、GAE、train模式或AdamW。新增入口也不改变平台2小时GPU闲置回收策略。源码与使用说明在新包README，命令另存[RESUME_COMMAND.txt](RESUME_COMMAND.txt)。本地六个替身控制流检查通过，真实GPU检查未本地执行。

## 执行建议

请平台按Pod `sg-duweiyue25-261113-08f04-default0-0` 和2026-11-13 08:53:21的时间映射宿主机、GPU UUID，检查ECC／Xid并隔离或修复故障设备；申请健康GPU后使用新启动入口恢复163。V100和A100均可继续既有GPU协议，型号不限，但换型号不保证逐位复现。

NVIDIA对Volta及更早GPU的Xid48说明需要GPU reset或节点重启清除此类不可纠正状态：[官方说明](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-48-dbe-double-bit-error-ecc-error)。本次未读取宿主机内核Xid，不宣称已经确认Xid48或永久硬件损坏。共享GPU重置／重启应由管理员协调，训练用户不关闭ECC、不清零计数掩盖问题。单纯重复提交新Pod并不保证分到健康物理卡；CUDA_LAUNCH_BLOCKING只辅助定位，不能恢复设备。

[EVIDENCE.json](EVIDENCE.json)记录原日志、环境与哈希；[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)为实际162模型CPU只读核验；[CPU_CHECKS.json](CPU_CHECKS.json)为替身启动检查。此次没有真实GPU调用、训练、SCIP、作业提交／取消或原脚本修改。
