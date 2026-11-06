# 六作业训练后的选模与 Crater 评测操作单

2026-10-23。六个正式训练均已核验：各 500 个完整 episode、五个稳定里程碑、最终空缓存、Actor/Critic 更新、有限模型与优化器状态、CUDA RNG 和最终 SHA 均通过。训练事实及原始日志保留。本文件对应阶段 5；阶段 6 未启动。

**当前六份开发集选模已完成，可以直接执行第二节的七个测试作业。** 450 次开发求解全部有效，六份 `actor_best.pt` 和六份 `actor_final.pt` 均已核验。全局权重于 **2026-10-23 11:16:43（北京时间）** 冻结，冻结文件 SHA 为 `a8489c97fa042f980fc6f2b770ad1e1597524b4b38d8b91f721f25df1c32bd1a`。无需重跑预检或选模；每个新测试作业仍会自行检查实际节点。当前 E5 开发容器不运行正式求解。

| 组别 | 训练种子 | 最佳里程碑 episode | 开发集平均 PDI，越低越好 |
|---|---:|---:|---:|
| B0 | 11 | 100 | 3534.3566 |
| B0 | 22 | 100 | 3543.3142 |
| R0 | 11 | 300 | 3531.4287 |
| R0 | 22 | 500 | 3541.1484 |
| Rjoint | 11 | 100 | 3534.8249 |
| Rjoint | 22 | 400 | 3530.1016 |

六份模型均按原规则分别选出；Rjoint/22 的开发分数最低，但不据此从两个训练种子中挑赢家，也不提前断言测试优势。所有训练都实际完成 500 episode，选中早期里程碑不减少已发生的训练成本。`actor_final.pt` 继续保留。

复查记录与可复制的七作业参数表：`tgppo/runs/server6-20261022/post-training-20261023/selection-review-20261023T032024Z/selection_review.json`、同目录 `evaluation_jobs.tsv`。下方第零、一节保留已经完成的预检/选模操作步骤，用于复查和必要恢复。

新入口：`post_job.sh` / `post_training.py`。没有改动训练运行时代码，协议指纹仍为 `32bed3e2cce3dd36e30e5bde1d3c95321b322081ce123296f00b92e3cdf18b80`。旧 `custom_job.sh` 的训练、准备入口继续保留；本次后处理请使用下列新入口，以获得训练审计、CPU、选模冻结和实例/cutoff 校验。

**2026-10-23 上午（北京时间）历史操作记录：默认仍为六个选模、随后七个测试作业，不要求创建 30 个选模作业。** B0 的两次提前选模曾因分配到 E5-2680 v4 而在求解前退出；原日志保留。随后目标 Gold 6148 节点的短预检通过，六个选模作业于 10:42—10:44 启动并已全部完成。申请 V100 不等于平台一定分配同款 CPU，应以各作业日志为准。

## 已核实的训练结果

| 组别 | 训练种子 | 完整 episode | 完整 PPO 更新 | 逻辑优化步 | 训练耗时小时 |
|---|---:|---:|---:|---:|---:|
| B0 | 11 | 500 | 125 | 3807 | 7.04 |
| B0 | 22 | 500 | 123 | 3786 | 7.22 |
| R0 | 11 | 500 | 125 | 3768 | 7.21 |
| R0 | 22 | 500 | 123 | 3714 | 7.14 |
| Rjoint | 11 | 500 | 125 | 3741 | 7.16 |
| Rjoint | 22 | 500 | 123 | 3723 | 7.24 |

六作业累计约 43.00 GPU 占用小时；这是训练进程记录的墙钟和，不是平台精确计费，不含准备、排队、历史提前启动失败、选模或测试。seed 22 各有两批在 episode 244/396 因累计决策不足 256 暂缓更新；已核实缓存保留并被后续批次消费，最终没有未更新转移。相同训练种子三组的初始 Actor 与 CPU/CUDA 随机流配对一致。

证据：`tgppo/runs/server6-20261022/post-training-20261023/training_audit.json`。它记录全部六个 latest、30 个里程碑的哈希、参数更新、所有 3000 个完整采样任务和更新缓存守恒。25 个训练/开发、66 个保留测试实例的内容 SHA 与已记录最优 cutoff 也已核对；没有运行保留测试集求解。

## 零、先用一个短 Custom 作业验证节点

镜像仍选 `scip (user-duweiyue25/envd:11260740-c4da)`；**8 CPU / 32 GiB / GPU 数量 1 / GPU 类型 V100**，工作目录和挂载同下文。不要使用 CPU-only 配置。若平台提供节点限制，选择原训练的 Gold 6148 节点类别。时限 15 分钟足够做这项预检，但排队时间另计。

启动命令：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh preflight
```

本项不运行 SCIP 求解；检查真实 CPU/软件指纹、CPU/内存配额、共享目录、归一化基线、30 个里程碑路径、六个最终稳定检查点的 SHA 和 CPU Actor 加载。看到 **`PREFLIGHT_PASSED`** 且平台正常结束后，再创建下面六个选模作业。通过记录在 `post-training-20261023/preflights/`；日志在 `custom-logs/post-preflight-*.log`。

若仍显示 E5-2680 v4，不要批量复制提交：需在 Crater 选择/约束 Gold 6148 节点，或请平台管理员确认对应节点类别。换镜像、把 GPU 数量填成 1 都不能单独保证 CPU 型号；脚本不会跳过同硬件公平性检查。后续六个选模、七个测试也各自执行真实节点检查，因此一次预检通过不能替其他节点担保。OOM、节点中断等仍可能发生；有效任务可由同一命令续跑。

## 一、再创建六个开发集选模 Custom 作业

统一镜像：`scip (user-duweiyue25/envd:11260740-c4da)`。Shell：bash。工作目录：`/home/duweiyue25/tgppo-transfer/research`。用户空间挂载点：`/home/duweiyue25`。

用户已确认 CPU-only 无法指定节点，因此本轮统一继续申请原先的 **8 CPU / 32 GiB / 1 V100**，时限 **4 小时**。实际 CPU 须为 **Intel Xeon Gold 6148**；本入口仍只使用 CPU。资源预检会明确拒绝不同 CPU/软件身份，不会自动放宽冻结协议。选模与七个正式测试作业使用同一配置。

以下六行分别填写到六个独立作业的启动命令框，可同时运行。不要把六行放入同一个作业。

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select B0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select B0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select R0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select R0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select Rjoint 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh select Rjoint 22
```

每作业在 **100/200/300/400/500** 五个里程碑上，使用相同 **5 个开发实例 × 求解器种子 0/1/2**，共 75 次完整求解。按实例等权平均 120 秒 PDI 最低选择，精确同分取较早里程碑。每个“组 × 训练种子”单独选，不从两个训练种子中挑赢家。评估加载 Actor，不构建 Critic，不提取额外前沿信息。

输出在原训练目录 `training/GROUP-sSEED/`：

- `development/episode-*/`：完整开发评估和失败/重试记录。
- `selection.json`：五个分数与选中里程碑。
- `actor_best.pt` / `actor_final.pt`：开发集最佳与固定 500 episode 的纯 Actor 权重。
- `selection_verified.json`：重新核对开发集覆盖、规则、源模型与导出哈希之后的记录。

中断后用同一命令创建续跑作业；按同一身份复用有效开发结果，保留旧尝试。不要删除任何缓存或锁文件，不重跑训练。

最后一个选模作业完成后自动冻结全部六份权重到：

`tgppo/runs/server6-20261022/post-training-20261023/frozen_selection.json`

若其他选模仍在运行，单个已完成作业可正常退出，但会说明尚未全局冻结。全部完成后，可在开发容器再次执行下列命令核验并补做冻结，不运行任何求解：

```bash
bash migration/2026-10-22/server-run/post_job.sh freeze
```

冻结后选模入口只核对并保留原权重，不再重新选择。任何模型、选择记录、训练协议或软件身份变化都会拒绝继续。

## 二、六份选模冻结后，再创建七个测试 Custom 作业

使用与选模一致的镜像、挂载、CPU 型号和资源配置：**8 CPU / 32 GiB / 1 V100 / 4 小时**。选模和测试程序均只使用 CPU；申请 V100 是为了获得与训练一致的节点 CPU，不能把 GPU 空闲误判为程序没有工作。

以下七行分别对应六份最佳 Actor 和一次原生 relpscost，可同时运行。每份 Actor 评估 **66 个保留实例 × 3 个求解器种子 = 198 次求解**；原生参照也为相同 198 次，总计 **1386 次**。每个任务 120 秒，四 worker 并行；单作业最坏求解时限合计约 99 分钟，另加加载、进程与保存开销，4 小时时限留有余量。

本次仍直接使用已经完成选模验证的 `post_job.sh` / `post_training.py`，未修改运行代码或算法。**每行对应一个 Custom 作业，共七个；不要将七行填进同一作业。** 程序根据冻结清单自动加载上表对应的 `actor_best.pt`，不需要手工填写模型路径。时限中断后仍使用同一行命令续跑，不删旧输出。

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate B0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate B0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate R0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate R0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate Rjoint 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate Rjoint 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/post_job.sh evaluate native
```

入口会先核验全局冻结文件，缺少任意一份选模或导出时以 75 退出；**不会先运行测试再补选模型**。它还核对实例与 cutoff 未改变、CPU/软件身份一致、每个实例三个求解器种子覆盖完整、CPU worker 未初始化 CUDA，以及 Critic/额外搜索特征开销为零。两个训练种子的六份模型均保留并评价。

原始结果与缓存：`tgppo/runs/server6-20261022/test/official66-best/`，按模型哈希分目录，原生参照目录以 `relpscost-native` 开头。完成记录：`post-training-20261023/evaluation/`。中断后同一命令恢复，仅重试缺失/无效任务；全部历史尝试保留。记录 Crater 的真实作业 ID、资源与实际节点，脚本无法代替平台计费记录。

七个作业及时并行调度时，预计测试墙钟约 **2—3 小时**，排队与失败重试另计。按 11:20 左右开始、无长排队估算，可争取 13:20—14:20 得到完整对照；这不是平台完成时刻保证。主指标为相同 120 秒时限下的原生 PDI；同时保留证明完成、PAR-2、节点数与 Actor 回调耗时，按实例配对并同时报告两个训练种子。测试前没有使用测试结果参与选模。

## 三、查看状态与结果边界

开发容器可执行：

```bash
bash migration/2026-10-22/server-run/post_job.sh status
```

所有作业日志：`tgppo/runs/server6-20261022/custom-logs/post-*.log`。

当前脚本覆盖包内 66 个保留实例。额外兼容评测集尚无独立实例清单及核实的 cutoff，因此没有加入新数据或混称已完成；补充时需冻结独立清单、排除训练/开发内容重合，并增加同设置原生对照。本次保留 `actor_final.pt`，主表先按冻结规则评价 `actor_best.pt`；若另测 final，须单列另加的 1188 次求解成本，不根据 best 测试结果再挑选。

训练检查通过只说明实验链路和预算完整，方法优劣仍需测试结果。B0 是 120 秒 / Adam / 四任务批更新参照；本轮没有 Rprod，结论不能单独归因于联合关联收益。阶段 5 实测交付后等待用户验收，阶段 6 未启动。
