# B0-seed11：用户提供的2574a作业配置与执行日志核对

## 对应关系和明确结论

用户提供的是`batch.volcano.sh/v1alpha1 Job`配置，name为`sg-duweiyue25-261111-2574a`、namespace为`crater-workspace`，正好匹配budget中11月11日10:15:47启动、11:53:28停止的attempt，不是后来新建的abd82作业。原附件已按原样保存在`B0_S11_JOB_2574a.yaml`。

配置中真实容器command与task-template注解里的command逐字相同，均为已交付resource-resume入口，trainmode/b0/seed11/cuda/48GiB/resume/no-total-wall-limit参数正确。

这次训练主进程的直接退出原因是收到SIGTERM15。没有发现Job运行时限、错误启动命令、内存门槛不匹配或SCIP watchdog触发的证据。SIGTERM的发送方及原始触发原因并未记录在本Job配置和训练日志内，不能据此唯一认定为某条平台回收策略。

## 实际执行到哪里

| 北京时间 | 证据 |
|---|---|
| 10:15:44 | Job creationTimestamp |
| 10:15:47 | 训练入口开始，预算覆盖为无总墙时限 |
| 11:17:21 | 第122个episode开始collect，实例rmatr200-p5、solver seed1；CPU型号迁移cohort已成功建立 |
| 11:53:28 | 父进程等待collect worker时收到SIGTERM，日志末尾为paper.Stopped |

总运行5861.25秒，约1小时37分41秒；当前采样只进行约2166.55秒，尚未到3600秒SCIP时限或4200秒watchdog。进入采样前主要是当前CPU上的新原生归一化，约耗时一小时。这一轮未完成第122次采集/更新，因此latest仍为已提交121的安全状态，没有新增pending。不是“脚本完全没启动”。

日志：`tgppo/runs/paper-resource-resume-20261109/launcher-logs/20261111T021547-bsFOGB.log`，最后调用链为train → cached_job → run_jobs → 等待worker → 收到信号后抛出Stopped。日志中collect路径迁移成功，未出现Cached task identity changed。

## 配置逐项排除

- request为4CPU/48GiB/1V100，limit为4001m/49153Mi/1V100。49153Mi与该失败作业内存监测到的51,540,656,128字节完全一致。采样内存峰值约13.09GiB，OOM/oom_kill均0。
- Job和Pod配置均未设置activeDeadlineSeconds；容器未配置livenessProbe或startupProbe，没有看到探针杀容器的声明。
- `maxRetry: 1`和task的`maxRetry: 3`是重试配置，不能读成一小时或三小时运行限制。[Volcano Job说明](https://volcano.sh/docs/v1.16.0/concepts/volcanojob/)
- `ttlSecondsAfterFinished: 604800`为完成/失败后七天清理，不会在运行97分钟时触发。[Volcano TTL说明](https://volcano.sh/docs/userguide/user_guide_how_to_use_job_ttl/)
- `PodEvicted → RestartJob`是发生该事件后的策略，配置里存在该条并不证明本次发生过驱逐。
- 实际入口安装unlimited_budget，无运行计时器。当前环境可读取的`/crater-start.sh`也只做用户环境准备和sudo启动命令，未发现timeout；未取得旧Custom挂载的ConfigMap历史内容，不能保证该同名文件就是旧作业运行时的逐字节副本。当前可读shell启动文件未发现限时设置，部分sudoers文件无读取权限；没有提升权限。

## 新增的源码机制线索：平台清理不写在Job YAML里

已读取Crater官方公开源码，固定提交`367718b388e0193d8e633f3710622810a9267ab3`：

- [charts/crater/values.yaml](https://github.com/raids-lab/crater/blob/367718b388e0193d8e633f3710622810a9267ab3/charts/crater/values.yaml)：公开默认低GPU检测90分钟、等待30分钟。
- [backend/pkg/cleaner/low_gpu_usage.go](https://github.com/raids-lab/crater/blob/367718b388e0193d8e633f3710622810a9267ab3/backend/pkg/cleaner/low_gpu_usage.go)：清理判定窗口是两者之和，命中后删除Volcano Job；`AlertEnabled`只控制发信，不是禁止清理的开关。
- [backend/pkg/monitor/query.go](https://github.com/raids-lab/crater/blob/367718b388e0193d8e633f3710622810a9267ab3/backend/pkg/monitor/query.go)：通过DCGM指标的窗口最大值及窗口起点是否已有该Pod指标来判断低GPU利用率。

这是确认存在的源码机制，不是本次已确认根因。无法获知集群实际部署版本、实际配置、命中指标及删除审计；公开默认删除窗口为120分钟，而本次仅运行约97.7分钟，不能直接按默认值解释这次终止。CPU任务的同期SIGTERM也不能据此统一归因。

同理，修改`alert-enabled`、maxRetry或TTL并不能据现有证据修复本次失败。不缩短SCIP、不增加GPU空转、不改原训练或全局平台策略。

## 当前状态与证据缺口

后来新建的B0-seed11作业`sg-duweiyue25-261111-abd82-default0-0`已在13:44:16启动、执行锁活动；读取时正在第122轮原生归一化，保持其运行，没有重复提交或停止。

要唯一定位2574a的SIGTERM，需要实际Pod终止状态、Crater清理/操作审计或节点事件。用户这份文件没有status，不含terminated.reason、exitCode、finishedAt等实际容器终态信息。当前账户无集群事件读取入口；本次不重复要求用户提供前端不存在的退出码，也不将未确认原因写成事实。

机器可读结果：`B0_S11_JOB_CONFIG_AUDIT.json`。本轮仅只读配置/日志、核对公开平台源码并形成说明，没有新增训练、SCIP、GPU使用或改动冻结脚本。
