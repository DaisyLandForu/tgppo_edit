# 固定200权重：已有四份的完整66实例CPU评测

用户于2026-11-13确认：先评已有4份200权重＋relpscost。已经冻结新B0-seed22、MLP-seed11、MLP＋BC-seed11／22；仅读取actor-0200.pt与对应episode-0200.pt，不使用latest或将100权重替充200。旧100轮与原500正式入口独立保留。

四个CPU Custom：每作业4CPU／32GiB／0GPU，共16CPU／128GiB，CPU型号不限。运行命令见[commands.txt](commands.txt)，详细说明见[docs/CUSTOM_JOBS.md](docs/CUSTOM_JOBS.md)。用户手动停止训练、提交评测；本地准备没有新增求解。

66实例×（四权重＋新relpscost）×solver_seed0＝330项，分片85／85／85／75。SCIP3600秒，watchdog4200秒；mspp16在最后分片独立单worker、24GiB地址空间，其余4worker、各6GiB。无累计训练48小时等总墙时停止规则。

仅在同一实例的五方法同环境结果全部齐全时形成配对；换机恢复保留已完整面板，对不完整面板重新配齐，避免跨环境拼绝对时间。不能用上轮旧原生结果代替新Custom的配对原生。全部330有效才输出完整效果表。

种子覆盖不齐：同种子可识别BC增量（seed11）和MLP＋BC相对新B0的整体增量（seed22）；缺新B0-seed11与MLP-seed22，不报告纯结构的同种子因果对照或跨种子稳定结论。MLP类来自已审计GPU→CPU续训分支，B0来自CUDA训练，训练设备边界保留。实际评测一律CPU Actor-only。

结果输出到`research/docs/methods/tgppo_paper200_test66_20261113`，原始结果到`research/tgppo/runs/paper200-test66-20261113`。停止条件为固定清单完成或失败退出；不自动扩种子、重试、训练、读取未来200权重或根据测试选择checkpoint。
