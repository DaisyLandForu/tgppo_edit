# 开发集三时间点CPU Custom评估

当前状态：入口已准备，等待用户手动提交；本地没有新增求解或训练。

执行方式见[Custom说明](../../../../tgppo/experiments/dev-curve-20261105/CUSTOM_JOBS.md)，统计和停止规则见[PLAN.md](PLAN.md)。一个4CPU／32GiB／0GPU作业，执行预计20–30分钟、累计上限40分钟；自动输出本目录REPORT.md及CSV后停止。不会重复原训练、扩大66测试或继续后续阶段。

复用当前MLP＋BC的BC初始化/PPO20/PPO100，两训练种子×5开发实例×两solver种子，并新测10原生，共70项；全部同主机，不与旧CPU的绝对时间混算。最终事实以共享盘原始JSON、哈希、CPU身份与报告为准。
