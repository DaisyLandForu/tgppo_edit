# 2026-11-11 六训练任务中断与固定100评测核查

检查时间：2026-11-11T09:59:08.300995+08:00（北京时间）。本次仅只读核查、写诊断文档和刷新评测进度；无新增训练／求解，无停止当前评测，无修改冻结训练或评测包。

## 结论

六个截图任务最新匹配日志均为`paper.Stopped: Signal 15; checkpoint retained`，是进程收到外部SIGTERM后退出。现有证据没有显示本次由Python训练错误、NaN、GPU显存不足或脚本48小时上限直接触发。SIGTERM的发出原因必须靠平台／节点事件确认；不能仅凭程序日志把物理断电认定为已证实。

两份BC作业于11月10日23:23:25相隔约0.07秒停止，CPU与GPU多个节点也在相近窗口退出，与用户报告的集群关停／断电事件相容。六个实际停止时间跨度23:22:59—次日00:46:57，并非同一秒发生的硬断电证据。

新B0-seed22最后一次已提交GPU PPO更新距SIGTERM仅约17.43秒；seed11约58分51秒，且同期CPU训练也停止，因此不支持用此前“GPU连续两小时未使用”统一解释这六次中断。仍不能排除平台其他回收策略，需要平台事件才能唯一归因。

四份有资源监测的作业（两新B0、两BC）OOM/oom_kill计数均0，采样容器峰值约11.47—15.06GiB；采样峰值不是完整峰值保证，另外两份MLP没有同样的资源监测记录。

## 已提交检查点与恢复位置

| 方法 | seed | 已完成／500 | 外部终止时间（北京时间） | 续训首先执行 |
|---|---:|---:|---|---|
| b0_new | 11 | 121 | 11-11 00:34:25 | 重新开始／恢复第122未提交episode的必要采样 |
| b0_new | 22 | 121 | 11-10 23:24:47 | 重新开始／恢复第122未提交episode的必要采样 |
| mlp | 11 | 163 | 11-11 00:46:57 | 重新开始／恢复第164未提交episode的必要采样 |
| mlp | 22 | 138 | 11-10 23:22:59 | 从更新前pending重放第139完整PPO更新，6795状态，无需重新采样 |
| mlp_bc | 11 | 181 | 11-10 23:23:25 | 从更新前pending重放第182完整PPO更新，1402状态，无需重新采样 |
| mlp_bc | 22 | 142 | 11-10 23:23:25 | 重新开始／恢复第143未提交episode的必要采样 |

八份实际checkpoint（六latest＋两pending）均已只读加载：配置SHA正确，Actor／Critic／AdamW数值有限、动量状态存在、Python／NumPy／Torch随机状态存在。最新Actor张量摘要与已提交update记录一致。两pending的模型／AdamW／policy_version／update_counter与各自latest逐项相等，含已完成轨迹；源文件读取前后SHA一致。未构造Agent或执行optimizer.step。

MLP-seed22在第139轮第3epoch、73/81逻辑minibatch时中断；BC-seed11在第182轮第1epoch、6/18逻辑minibatch时中断。中间更新仅在原进程内存中，不从73/81或6/18继续叠加，而从pre-update pending重放该轮全部更新。固定更新随机流及既有恢复机制保留。

六个执行锁检查均空闲，最近预算attempt均已结算，不需要删除锁、清理缓存或修改预算账目。`--resume --no-total-wall-limit`恢复各自现有分支，保留500episode、单次3600秒、GAE与逐episode三epoch AdamW。CPU MLP分支必须继续CPU recovery目录，不能回到旧GPU父目录丢失新增进度。

## Custom续训命令与资源

完整六条自包含命令见[RESUME_COMMANDS.txt](RESUME_COMMANDS.txt)，每条分别提交一个Custom。

- 新B0两组：各4CPU／48GiB容器内存／1 V100（A100也兼容），`--experiment trainmode --variant b0 --device cuda --ram-gib 48`。恢复原trainmode/training/b0-s11或b0-s22，目标从121继续至500。
- MLP及MLP＋BC四组：各4CPU／64GiB／0GPU，`--experiment aligned --device cpu --ram-gib 64`。恢复paper-recovery/cpu-aligned现有四份latest。
- 所有作业CPU型号不限。原恢复入口处理仅CPU型号改变的采样cohort；软件／配置／策略／动作种子发生其他变化时仍拒绝错误缓存。跨硬件不承诺逐位相同。
- 两小时低GPU利用率回收的既有风险仍在；本次不缩短3600秒或更改训练方式，也不因这次中断重新初始化。

## 固定100 episode的66实例评测

当前评测仍在原终端机器运行：父调度flock PID3786、shell3792，分片1主进程3995及四CPU worker可见。没有证据显示这条现有评测也被此次关停打断，不应重复提交。

快照：106 / 330原始结果有效，0失败；100 / 330项已形成完整五方法配对，覆盖20 / 66实例。

| 分片 | 计划项数 | 已落盘有效项 | 完整实例配对 | 状态 |
|---|---:|---:|---:|---|
| 0 | 85 | 85 | 17 | complete |
| 1 | 85 | 21 | 3 | running |
| 2 | 85 | 0 | 0 | 等待前序分片 |
| 3 | 75 | 0 | 0 | 等待前序分片 |

分片0为17实例85项，已完整耗时约10.55小时；分片1正在推进，分片2／3还未开始，因为用户当前只有一台4CPU／32GiB环境，采用顺序执行。该进度不等于全测试集完成。

现仅刷新progress.json，未发布局部总体PDI／结构／BC输赢排名，避免将执行顺序形成的测试子集当完整66结论。保持现有后台任务，全部330项齐全后再统一分析新旧B0、MLP结构、BC、相对relpscost及部署成本。训练续跑不修改这四份冻结100权重。

## 证据与边界

- `EVIDENCE.json`：六份匹配日志、状态、预算、停止时间、内存监测及源码冻结SHA。
- `CHECKPOINT_AUDIT.json`／`check_checkpoints.py`：实际八份checkpoint只读核验；无训练、无新增SCIP、无CUDA初始化。
- `EVAL_PROGRESS_SNAPSHOT.json`：读取进行中的worker原始JSON及同环境完整配对快照，后续结果还会增加。
- 旧49、新53、recovery6、resource6文件SHA通过；固定100评测包及其输入／保护源码验证通过。
- 未访问平台电源／scheduler事件。可确认的是外部SIGTERM；物理断电、维护驱逐或其他平台策略的唯一根因仍缺平台证据。
- 本次诊断与续训命令交付完成，实际六个Custom仍由用户提交。
