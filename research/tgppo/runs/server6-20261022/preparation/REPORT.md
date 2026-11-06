# 当前开发容器核查与 Crater 交付记录

核查时间（北京时间）：2026-10-22T22:33:33.953974+08:00。当前为无 GPU 开发容器，正式作业由用户在 Crater Custom 界面创建。阶段 5 尚未完成或验收，阶段 6 未启动。此报告不是正式训练或测试结果。

交接要求的六份文档已完整阅读；核对了实际文件、进程、归一化统计与可用工具。初始直传包 4,997 项、364,535,640 字节全部匹配原始清单；25 个训练/开发与 66 个测试实例逐一核对 SHA、下载清单、gzip 完整性、cutoff 最优记录/来源，20/5/66 划分无名称或压缩文件哈希交叉。cutoff 核查依据包内记录，不是重新证明 91 个实例的最优性。训练统计 mean=-13.166822661165169、std=8.840326545776836，19 个有统计的实例全部属于训练集，无开发/测试混入。

原始清单与哈希台账未重写。修复涉及 6 个现有活动代码/入口文件，新增 8 个工具/说明文件；变更前副本、补丁和新哈希另存。事后核对未发现这些明确变更以外的原始条目变化；总字节差恰好等于活动文件大小差。历史 repo、stage2—5、原始实验日志和冻结协议数值保持不变。

已在当前机器重建 SCIP 6.0.1 / SoPlex 4.0.1，原生版本输出确认 githash a27ecc0 / e89d7d5，GMP、zlib、Readline、bliss 均已链接。stage4 定制绑定在独立源码副本中以 Cython 0.29.37 重编译，五接口全部导入成功；没有搬用源机 .so。Python 3.8.20、Torch 1.13.1+cu117（运行库 11.7）、NumPy 1.24.4、SciPy 1.10.1，pip check 通过。镜像使用用户确认的 scip（user-duweiyue25/envd:11260740-c4da）；Conda、solver 和代码均位于共享 /home，无需另存新镜像。

当前 CPU 为 Xeon E5-2680 v4，affinity 可见 56 个逻辑核，cgroup 配额为 16.01 核、约 64 GiB；/dev 无 NVIDIA 设备，torch.cuda.is_available() 为 false，未安装 sbatch/squeue/sinfo/sacct。CUDA 预检按预期拒绝运行，没有 CPU 静默回退。Crater GPU 节点、队列等待、作业时限和目标 GPU 镜像运行状态没有验证，不填写虚构的 Slurm 分区、账户或作业 ID。

工程检查共 53 项：51 项成功，2 项 GPU 检查因无 GPU 跳过，0 failure / 0 error。包括真实 SCIP 短任务、B0/R0/Rjoint 更新、中断半轮 PPO 后显式恢复、复用采样、模型/优化器状态一致、CPU worker 不初始化 CUDA、Actor-only 导出/加载、开发集规则与失败覆盖。检查采用小模型和 2 秒任务，不是 500 episode 正式训练。检查墙钟 106.53 秒，self+children CPU 135.76 秒；这是此轮成功检查的成本，不包含依赖下载、编译和先前失败的工具尝试，不能用来预测 V100 正式耗时。完整准备成本尚未形成平台账单。

必要修复与入口：Critic 额外模块初始化现在同时保护已初始化 CUDA 随机流；正式训练核对 cgroup CPU/内存与冻结 Python/Torch/SCIP 版本，要求匹配目标桥接和代表性吞吐证据。GPU 桥接加强为真实优化步骤后注入中断，与完整更新参考比较 Actor、Critic、两套优化器及 CUDA RNG，并确认未重复采样。新增 20 个训练实例/组三组独立吞吐试跑，记录有效更新、GPU 分配/保留峰值、worker RSS 和 100/500 episode 估算；合法的小样本缓存保留不伪造更新。训练算法、奖励、Actor 输入、原单位 value/GAE、损失、优化器、4 任务批与正式 500 episode 数值均未修改。

六个正式作业当前均未启动：B0/11、B0/22、R0/11、R0/22、Rjoint/11、Rjoint/22，均 0/500，无作业 ID。当前没有正式 normalization/baselines.json、目标 gpu-bridge/verification.json、throughput/verification.json、正式 latest.pt、五个里程碑或 summary.json。今晚能否完成未知，需要目标吞吐和排队实测。旧矩阵仍是历史上的两组各 16 episode 中断记录；旧模型不在包里，本机没有重新验证那些旧 .pt，不把缺失当预算清零或完成。

最短执行路径是创建一个 8 CPU / 32 GiB / 1 V100 的 Crater prepare 作业，使用同一镜像、挂载 /home/duweiyue25，执行 custom_job.sh prepare。它在前台依次做目标预检、小型检查、75 条原生基线、三组真实 GPU 恢复桥接和覆盖 20 训练实例的成本试跑。全部通过后，按 CRATER.md 六条命令创建六个独立训练作业，每个 8 CPU / 32 GiB / 1 V100、500 episode、100/200/300/400/500 保存；中断后同命令显式 --resume，不删除历史尝试。原 Slurm 六作业数组保留，增加了 pilot 入口，但这里没有执行 Slurm 提交。

六次训练完成后，仍须逐训练种子在同一 5 开发实例×3 求解器种子上选模，保留 actor_best.pt / actor_final.pt；两个训练种子都报告。六份选模结果冻结后，才做 Actor-only 保留测试与同硬件/同设置 relpscost 对照。当前没有额外兼容集清单；其 cutoff/来源需先核实。本轮 B0 是 120 秒/Adam/四任务更新参照，缺少 Rprod，不能单独归因联合关联收益。完成阶段 5 实际结果与成本汇报后，等待明确验收。

可复查入口与证据：

- [Crater 配置与七个作业的启动顺序](../../../../migration/2026-10-22/server-run/CRATER.md)
- [初始整包核验](./archive_audit.json)、[变更后原始文件保护核验](./archive_audit-after-changes.json)、[校准统计](./calibration_audit.json)
- [环境版本及二进制 SHA](./runtime_versions.json)、[CPU 预检](./cpu_preflight.json)、[SCIP 原生版本](./scip-version.log)、[动态依赖](./scip-ldd.log)、[pip 检查](./pip-check.log)、[当前依赖](./pip-freeze.txt)
- [本机检查原始日志](./tests-20261022T142534Z.log)、[检查与资源汇总](./tests-20261022T142534Z.log.result.json)、[CUDA 拒绝日志](./cuda_expected_failure.log)
- [六组真实检查点状态](./six_job_status-final.json)、[就绪范围汇总](./readiness.json)
- [变更补丁](./changes.patch)、[新活动源码台账](./active_sources.json)、变更前副本 source-before/

已解决的构建/工具异常均保留：最初相对路径构建指向错误目录，随后绝对路径脚本构建成功；依赖安装期间两次绑定导入分别缺 NumPy/SciPy，最终 build-attempt-20261022T142053Z-20415 完整通过。最初测试计时调用缺少 /usr/bin/time，随后从 stdin 启动的计时包装与 multiprocessing spawn 不兼容，产生 1 failure / 1 error；改为有 main guard 的 verify_environment.py 文件入口后，上述 53 项重新完成。此前失败日志 tests-20261022T142358Z.log、tests-20261022T142431Z.log 及 build-attempt-* 均保留；没有把失败记录重写为正常退出。
