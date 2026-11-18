# 旧训练跨 GPU／CPU 恢复入口核验

独立入口：`research/tgppo/experiments/paper-resume-20261108/custom_resume.sh`。
支持旧 paper-aligned 的 B0／MLP／MLP＋BC、种子 11／22；不修改旧训练包或新 trainmode 包。

seed11 三组检查点已提交 91 episode，第 92 次原采样无有效结果，原 CPU 是 Platinum8358。
旧缓存严格检查 CPU 身份，直接换节点可能在重新采集时拒绝恢复。
新入口仅对 CPU 型号变化的未提交采集建立独立 cohort，保留旧 identity 与 attempt；
任务、checkpoint、动作种子、绑定与软件必须一致。原生分母仍按新 CPU 身份重新获取。

原程序继续负责模型、AdamW、RNG、pending、预算和执行锁。
采样／更新模式、GAE、更新组织、500 episode、3600 秒均不变。
当前已有 full pending 的其他条件仍按原程序恢复 pending，适配器不干预更新。
不同 CPU 在墙时约束下可能改变轨迹和分母，跨 GPU 也不保证逐位一致，需保留硬件来源。

8 项合成缓存路由检查全部通过；原 49、新 53 个冻结文件校验无变更，
入口完整校验通过。只运行合成 worker，没有训练、GPU 或新增 SCIP 求解。
证据为本目录 `CPU_CHECKS.json` 与 `cpu_checks.log`；V100 实机检查留在 Custom 启动时执行。

最终核验时发现 B0-seed11 已增加一个未结束续跑记录，目标 A100／Silver4216，
state 为 normalization。不能仅因本机没有进程就判断远端停止，不应重复提交该条件。
其预算的未结束 attempt 按原程序保守预留全部剩余额度，不能将该预留误写为实际耗尽。
MLP 与 MLP＋BC 的已结算剩余累计预算约 17h59m／20h50m；适配器不会清零或追加。

交付后停止，实际作业由用户手动提交。

## B0-seed11 后续失败核验

用户报告旧入口带 `--resume` 的 B0 作业失败。原 launcher 日志和 state 明确为
`Cached task identity changed: .../episodes/92`：新旧 stack 仅 CPU 从 Platinum8358
变为 Silver4216，仍使用 A100；不是本次 GPU 释放或 CUDA 错误。
latest 文件 SHA 与 episode92 的策略身份一致，已提交 91、无 pending，未写入第92次更新。
此次续跑 3673.458 秒后失败，累计已结算 123949.463 秒，剩余 48850.537 秒（13h34m11s）。
应改用独立 `paper-resume-20261108/custom_resume.sh run --variant b0 --seed 11 --resume`，
保留原记录并重新采集未完成第92次；原训练源码／模式／预算不改。
该入口完整只读校验再次通过，本次没有训练／求解／提交或修改 checkpoint。
详细快照见 `B0_FAILURE_FOLLOWUP.json`。
