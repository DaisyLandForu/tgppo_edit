# 新B0恢复失败与48GiB资源核验

用户后续贴出的实际命令是 `paper-trainmode-20261108/custom_job.sh ... --seed 11 --device cuda --resume`，
因此该失败属于seed11；平台名称如果叫seed22，也不会改变命令实际使用的种子。

2026-11-09 13:47（北京时间）日志报 `Cached task identity changed: .../b0-s11/episodes/12`。
新旧缓存环境仅CPU从Gold6148换为E5-2680v4；GPU为V100-PCIE-32GB。
最新checkpoint SHA仍与该未完成采样策略身份相同，已提交11，未丢权重或优化器。
seed22的原GPUcheckpoint仍为12；未找到能够归因于这次用户所述“seed22 GPU失败”的另一份新日志，
不能把seed11错误记到seed22头上。详见 FAILURE_EVIDENCE.json。

## 48GiB是否可用

原协议 `ram_gib_per_training_job=64` 在GPU preflight中形成硬门槛；
这不是模型数学要求，也不是V100显存要求。原worker虚拟地址空间上限32GiB。

现存有效采样记录的单worker峰值RSS：

| 条件 | 记录数 | 峰值最大值 |
|---|---:|---:|
| 新B0 seed22 | 12 | 2.506GiB |
| 新B0 seed11 | 11 | 2.444GiB |
| 旧B0 seed11的原路径记录 | 136 | 3.516GiB |

以上未统计训练父进程、所有子进程及页缓存的整容器峰值，不能用来保证500全过程的内存。
但没有证据支持必须申请64GiB，48GiB是合理的本轮资源选择。
MEMORY_EVIDENCE.json保存原始结果路径与checkpoint哈希。

## 独立修复入口

`research/tgppo/experiments/paper-resource-resume-20261109/custom_job.sh`：

- 复用已验收recovery入口的显式恢复、CPU cohort、执行锁和可选取消累计墙时停止。
- `--ram-gib 48`仅覆盖资源预检副本；原训练配置／checkpoint哈希、模型、GAE、AdamW、
  batch／microbatch、500episode、SCIP3600和worker32GiB均不变。
- GPU模式继续原trainmode CUDA分支，不从CPU分支迁移权重。
- 接受cuda与gpu别名，日志明确打印实际variant／seed，减少作业名和执行参数混淆。
- 每30秒保留容器usage／peak／OOM事件，记录资源覆盖来源，主机RAM与GPU显存分别解释。

申请4CPU／48GiB／1GPU，命令见新包commands.txt。实际失败的seed11选对应行；
要运行b0_new_seed_22则必须使用`--seed 22`。
该修复不解除平台两小时GPU闲置回收规则。

## 验证范围

6项合成资源检查：原64门槛拒绝48、新入口接受48、仍拒绝32、64选项恢复原门槛、
CPU／软件检查保留、GPU别名／显式resume转发、内存峰值与事件记录。
测试直接执行冻结preflight函数的AST，替换硬件查询；不导入Torch或求解器、不使用真实GPU。
最初依赖导入检查90秒到限，日志另存；不把它当训练失败，也未改变训练时限。
最终结果见CPU_CHECKS.json。原49／新53／恢复包6个冻结文件无变更，见SOURCE_AUDIT.json。
本次没有真实训练、SCIP求解、作业提交／取消或checkpoint写入。
