# 原论文与官方代码初步核对

核查对象：ICLR 2026 正文与附录；作者 commit `66ab2d6572eaf83c279a07c21445272e8e3cc34f`。本记录是准备阶段发现，尚未代表修复后的完整训练流程已验证。

| 项目 | 已核实情况 | 处理安排 |
|---|---|---|
| 数据 | 论文 Table 20 明确 MILP-S 19 train / 8 test；仓库未包含实例或 HDF5 数据 | 独立复制并核验 27 个公开实例；专家轨迹需重新采集 |
| 专家与验证划分 | random 前缀 0/1/5/10/15，再由 relpscost 采集；solver seed 4 仅作验证 | 按论文生成采样清单，按完整轨迹划分，禁止随机拆同一轨迹混入验证 |
| SCIP | 论文 8.0.4；镜像已具备库/头文件；上游代码要求定制接口但未附绑定 | 将公开 TreeGate 25/8/53 维特征移植到独立 PySCIPOpt 4.3.0，编译后实测 |
| primal heuristics | 实测原版 utils 仍启用 `adaptivediving/freq=5` 和 `padm/freq=0`，与论文关闭全部 primal heuristics 不一致 | 工作副本保留原设置后，将全部 `heuristics/*/freq` 设为 -1；`fixed-heuristics-audit.json` 已验证全部关闭 |
| Mamba | `GroupedMamba` 实际构造 `Mamba`；d=8、d_state=64、d_conv=4、expand=2 | 保持 Mamba-1，GPU 目标为 V100；不能把 Mamba2 当作原模型替代 |
| 训练上下文 | 论文 99 步历史 + 当前，共 100 状态 | 显式配置 100；batch=32；正式启动前用真实候选规模测显存 |
| 评估上下文 | 论文 24 步历史；已追踪回调：deque 交替保存候选组与动作组，`max_seq_len=49` 对应 24 个完整历史步骤加当前候选组 | 保持上游参数 49=2×24+1；此前将 49 误记为历史步数的判断已更正；配置区分步数与 deque 项数 |
| 评估根节点 | 论文主表使用 relpscost 根节点；另有 -p 纯神经版本 | 分开记录，正式主表不得遗漏根节点初始化 |
| 优化器 | CL Adam/lr=1e-4；IL AdamW/lr=1e-3/wd=.01 | 按 Table 6；不套用 TGPPO 的 PPO 预算或优化器 |
| epoch 和选模 | 上游默认 CL50/IL50，评估默认 epoch99；论文未给出完整选模细则 | 在真实闭环前明确来源和选择规则；不得把脚本默认值标成论文已明确给出 |
| padding | `real_clip_process` 没有使用返回的 candidate mask；IL 的 CE 没有屏蔽 padding 候选；Mamba 也没接收 `new_masks` | 用不同候选数的真实样本验证影响；以论文有效候选集合为准明确修复，并与上游差异一起报告 |
| HDF5 标签类型 | HDF5 转换器把动作索引同特征一起存成 float32；当前 Torch 对 `LongTensor([numpy.float32])` 报 TypeError，真实 pilot 已复现 | 读取时验证标签有限、非负、为整数，再显式转为 long；真实 HDF5 转换/加载/批处理已通过 |
| 样本跳过 | `finetune_process_large` 对超候选上限/NaN 返回 None；训练直接 continue | 正式入口需要计数、报错或明确策略；不能静默减少数据 |
| 随机种子 | main.py 有 --seed，但训练入口没有完整设置 Python/NumPy/Torch/CUDA RNG | 正式入口统一设置并随 checkpoint 保存 |
| checkpoint | 上游主要保存 state_dict，没有完整训练中断恢复状态 | 正式作业需要优化器、进度、随机状态和数据顺序，单独验证恢复 |
| 异常 | 求解评估的宽泛 except 可能把失败计成正常结果 | 适配入口保留失败状态、traceback 和退出码 |
| 设备 | 默认 GPU 编号为 1；Custom 通常只暴露一张卡 | 适配入口使用可见卡内的 cuda:0，不覆盖平台分配 |
| 路径与依赖 | 多处硬编码 /home/data1；顶层导入部分未用于 Mamba 的 Ecole/PEFT/GNN 依赖 | 独立工作副本去掉 Mamba 路径未用的导入，保留补丁；后续使用显式数据/输出路径 |

绑定移植的兼容性细节：新版 `getLPBranchCands()` 返回六元组，旧版返回变量列表；node 特征中的候选数必须保留旧语义。节点边界和深度使用 SCIP-8 accessors；内部统计使用本机 SCIP-8 实际头文件编译。`getFairNNodes` 同时接收 bytes/str。有限值公式保持来源版本，闭合区间位置定义为 0。三组特征都必须在真实根节点和非根节点回调里检查。

本轮已修复并实测 SCIP-8 的启发式关闭遗漏和 HDF5 标签类型兼容问题。真实 pilot 的 15 条训练侧样本包含 8–48 个候选，批处理填充了 384 个候选位置；同一固定初始化下，原版 CL 损失为 1.8288034，只在有效候选上计算则为 1.9691664（比较时保留同一个常数项）。这是无参数更新的诊断，确认 padding 会改变目标；不代表已经修复训练，也不代表性能结果。

2026-10-27 GPU 验证补充：独立适配层 `code/paper_training.py` 已修正诊断路径的 padding：CL max-pool 和负样本只含有效候选；IL 在进入 Mamba 前去掉候选 padding，仅在整条序列末尾补齐，输出 CE 只包含真实候选和真实步骤。保留原候选顺序、历史专家动作和分组位置编码，末步目标动作不作为输入。测试覆盖污染 padding 值、候选数不等、序列长度 8/3 不等、单条与批处理输出一致，以及修改当前专家动作不会改变当前及过去的预测。损失与梯度比较考虑 FP32 数值误差，原始误差在报告中保留。

损失按论文明确选择：CL 使用式 (2)，省去上游常数 +2（不改变梯度）；IL 使用每条轨迹内求和、轨迹间均值，而上游默认对状态取均值，完整 100 状态时相差 100 倍。这是论文公式与公开实现的差异，不能把两种归一化产生的原始 loss 数字直接比较。HDF5 仍沿用作者转换器的 `候选数 > 1` 筛选，再划分 100 状态窗口；这项显式数据规则与静默跳过坏批不同。

固定数据 pilot 保存了两套网络、优化器、Python/NumPy/Torch/CUDA RNG、数据顺序和游标。加载状态逐位相同，下一批及下一步 loss 一致；CUDA 更新后参数/优化器有微小浮点差异。先前严格逐位/逐元素阈值失败报告保留；最终检查另报告按张量尺度计算的误差（32×FP32 epsilon），不能声称逐位确定性。它不是正式多 epoch 或跨进程作业恢复验证。

原版 main.py 保留作对照；正式入口另外实现为 `scripts/formal_jobs.py / formal_data.py / formal_train.py`，已集成适配层、梯度累积、上游 max_norm=1、原子 checkpoint、完整进度/RNG/样本顺序恢复和验证集选模。诊断数据上完成 2+2 epoch 和两阶段跨进程恢复，与连续运行数值一致，见 `runs/entry-check-20261027-final/verification.json`。正式 50+50 epoch 与 475 条采样仍待 Custom 执行；不能把入口验证当作正式结果。

epoch50/50 来自上游默认值；全量 CL 状态（不套用上游 200000 cap）、CL/IL 各自派生初始化种子、验证 loss 最小且并列取最早等补充选择已在 `configs/formal-milp-s.json` 冻结并注明来源。完整工作副本补丁为 `setup/working-source.patch`；正式执行另保存全部脚本/代码快照和包版本、绑定/输入 SHA256。作业顺序和命令见 `CUSTOM_JOBS.md`。
