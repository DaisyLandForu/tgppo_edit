# B0_new 两种子启动ECC故障核查

核查日期：2026-11-12，北京时间。两作业最新失败均为CUDA返回`uncorrectable ECC error encountered`，发生在启动目标设备自检、构造临时网络后向CUDA搬运参数时。该启动路径尚未到生产latest.pt的加载，也未开始本次真实episode或PPO更新。当前应优先更换健康GPU，或请平台管理员恢复设备健康后按原命令续训。

| 条件 | 本次开始→终止（北京时间） | 已提交episode | 下次恢复 | pending |
|---|---|---:|---:|---|
| B0_new seed11 | 11/12 21:06:47→21:07:32.869 | 152/500 | 153 | 无 |
| B0_new seed22 | 11/12 21:07:37→21:07:41.521 | 156/500 | 157 | 无 |

seed11日志为`20261112T130647-6pyKbR.log`，Pod为`sg-duweiyue25-261112-ac9dc-default0-0`；seed22日志为`20261112T130737-NGtgWm.log`，Pod为`sg-duweiyue25-261112-35cd0-default0-0`。两作业记录均为Gold6148／V100-SXM2-32GB／48GiB；这些是CPU/GPU型号和Pod标识，没有物理节点名或GPU UUID，不能据此认定分配了同一物理卡。

已确认的事实：

- 两份异常栈均为`paper.train → target_checks → build_agent → actor.to / critic.to`；源码在target_checks通过后才执行生产检查点加载。两次失败用时约45.87秒与4.52秒，区别于此前长时间求解中的SIGTERM。
- 48GiB资源检查已通过，两个内存监测文件的OOM/oom_kill计数均为0，采样容器内存峰值分别约9.73GiB和0.24GiB。当前报错不是CUDA out of memory，也没有依据通过改成64GiB或缩小PPO batch解决此ECC故障。容器RAM监测不能替代GPU ECC诊断。
- 两份生产latest.pt在本次失败前就已保存（seed11为11/11 22:49，seed22为11/11 23:41），不存在pending.pt。已实际在CPU只读加载并核对配置SHA、模型参数有限、两套AdamW状态有限、Python/NumPy/Torch/CUDA随机流齐全、Actor摘要与最新已提交update一致；检查前后文件SHA相同。两执行锁空闲，可以从最新提交点续训。
- 原旧／新／recovery／resource四包冻结文件逐项SHA核验通过，没有修改训练脚本、权重或训练协议。核查没有构造Agent、执行optimizer.step、初始化CUDA或新增SCIP求解。

源码机制与证据边界：CUDA官方将`cudaErrorECCUncorrectable`定义为检测到不可纠正的ECC错误。NVIDIA针对Volta及更早设备的Xid48说明，此类GPU不可纠正错误通常需要GPU reset或节点重启清除。V100属于Volta。当前程序日志未保存宿主机内核Xid与GPU UUID，不能声称已实测Xid48，不能仅凭报错判定永久硬件损坏或归因于前次集群断电。GPU／内存硬件及驱动状态应由平台诊断。

建议执行：

1. 通过平台换到经过健康检查的GPU节点；同型号健康V100可以继续使用。原卡由管理员根据ECC/Xid、页退役与DCGM诊断决定重置、重启或进一步维修。不在共享节点上由训练用户执行GPU reset，也不关闭ECC或用清零计数冒充恢复。
2. 两个作业继续使用既有入口及`--resume`，命令见[RESUME_COMMANDS.txt](RESUME_COMMANDS.txt)。无需清空模型、删除锁或从第0轮开始；目标仍500，单次求解仍3600秒。
3. 若新作业仍分到未恢复的原卡，单纯重启Python／容器可能复发。平台可按上面的Pod标识和21:07的精确时间映射到宿主机、GPU UUID，并查看内核Xid与ECC计数。`CUDA_LAUNCH_BLOCKING=1`只能辅助定位同步报错位置，不能修复ECC。

平台可只读采集`nvidia-smi -L`、`nvidia-smi -q -d ECC,PAGE_RETIREMENT`及宿主机内核Xid、DCGM健康检查；GPU reset/节点重启由管理员协调其他作业后处理。本会话没有访问到故障卡的实时设备状态，没有代用户提交、取消或重启作业。

证据：[EVIDENCE.json](EVIDENCE.json)、[CHECKPOINT_AUDIT.json](CHECKPOINT_AUDIT.json)。

官方依据：

- [CUDA Runtime错误定义](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)：cudaErrorECCUncorrectable=214。
- [NVIDIA Xid48与Volta及更早GPU处置说明](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-48-dbe-double-bit-error-ecc-error)。
- [NVIDIA Xid日志与诊断工具说明](https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html)。
