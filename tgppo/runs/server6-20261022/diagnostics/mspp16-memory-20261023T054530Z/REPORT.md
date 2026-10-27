# mspp16 内存失败诊断，2026-10-23

截图四个评测作业各有 195/198 条有效结果保留，仅 mspp16 的三个求解器种子失败。其他模型与原生 relpscost 也已观察到同样失败。日志先报 SCIP Insufficient memory / No memory，正式 worker 随后以 -11 退出；其余任务执行结束后，cached_jobs 因仍有无效结果而返回失败。这不是训练或选模丢失，也不表示全部 198 条都无效。

原 worker 配置 resource.RLIMIT_AS=8 GiB。这是虚拟地址空间，不是 Crater 的 32 GiB 容器物理内存额度；直接增加容器 RAM 不会修改内部上限。

本次在 E5 开发容器进行两次隔离工程诊断：复用同一原生 mspp16/solver-seed0 输入及 120 秒求解设置，仅将 worker 地址空间上限分别设为 8/24 GiB，每个单线程。没有改动源代码、原结果、模型或正在运行的正式作业。诊断结果不计入 Gold 6148 的正式测试。

| 地址空间上限 | 观察到的虚拟地址峰值 GiB | 实际驻留峰值 GiB | 结果 |
|---:|---:|---:|---|
| 8 | 7.9993 | 6.0311 | valid=false，SCIP insufficient memory |
| 24 | 11.3630 | 9.2015 | valid=true，正常 timelimit |

24 GiB 诊断的 load_wall_seconds=27.7064、optimize_wall_seconds=120.1943、worker_wall_seconds=160.6244。证明突破原 8 GiB 内部限制可解决该原生任务的内存失败。模型 Actor-only、其他 solver seeds 仍需在 Gold 上实际补跑验证，不能把本工程诊断冒充正式成功。

建议的最短恢复方案：保留每路其余 195 条有效记录及全部失败原始记录；新增独立补跑入口，对六份冻结 Actor 和原生 relpscost 的同一 mspp16/0,1,2 总计 21 条任务统一使用 24 GiB 地址空间、单 worker。容器可继续 32 GiB；不让四个增额 worker 同时竞争容器 RAM。保留原 Gold 6148、SCIP/绑定/模型 SHA、cutoff、求解种子和 120 秒求解时限。可将进程 watchdog 放宽至 300 秒以覆盖大实例读取/释放，不扩大 SCIP 的 120 秒搜索预算；这是需要单独记录的资源例外。补跑结果单独保存、链接父协议和旧记录，不把它们伪装成原 8 GiB 协议的成功尝试。

本轮尚未编写或启用该补跑入口：用户此前明确要求不改代码。现有四个运行入口及全局冻结清单的 SHA 已再次核对未改变。无需重训、重新选模或清空缓存重跑。

详细工程数据、结果与日志见本目录 verification.json、result-8GiB.json、result-24GiB.json 和 probe-*.log。verification.json 的父进程 elapsed_seconds 包含等待其他探针结束，不用于单个任务成本；任务成本使用 result 内的 worker_wall_seconds。阶段 5 未完成，阶段 6 未启动。
