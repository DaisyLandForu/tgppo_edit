# 航空布线最大实例：三方法无求解时限对比

用户确认：`k=4`，`edges.csv` + `pairs-4-246.csv`，TGPPO 使用原 B0、训练种子 11 的开发集选定 Actor。默认 SCIP seed=0。三作业独立求解，不共享解、不训练、不用已知最优 cutoff。

## 提交方式

沿用 **scip 镜像**，挂载 `/home/duweiyue25`。将 [commands.txt](commands.txt) 的前三条命令分别提交为 **3 个 Custom 作业**，可以同时启动，工作目录不限。

| 作业 | 方法 | 建议申请 |
|---|---|---|
| `sssp` | C++ 原图化简和 MILP 建模，SCIP 默认原生分支 | CPU 作业，至少 64 GiB RAM |
| `mamba` | 冻结 Mamba-Branching，24 步历史、根节点 relpscost | CPU 作业，0 GPU，至少 64 GiB RAM |
| `tgppo` | 冻结原 TGPPO B0 Actor，确定性 argmax | CPU 作业，0 GPU，至少 64 GiB RAM |

SCIP 和数值库都固定 **1 线程**；平台若只提供 8 CPU 档也可使用。尽量选同型号 CPU，汇总会检查硬件差异。**新提交默认 `--device cpu`**，两种神经方法均能在 0 GPU 的 CPU Custom 上推理。SCIP 在根节点、LP、割平面和启发式阶段可能连续数小时不调用神经网络；因此本实例推荐 CPU 作业，避免 GPU 长时间空闲被回收。只改命令但仍申请 GPU 作业不能解决该回收原因。

重新提交已被回收的 Mamba：

```bash
bash /home/duweiyue25/tgppo-transfer/scip/airwiring_compare/custom_job.sh mamba --device cpu
```

申请 **0 GPU、至少 64 GiB RAM**，选择合适的 CPU 队列和平台允许的运行时长。它会保留原失败/中断尝试并建立新 `attempt-*`，复用 CIP 和冻结权重，从头搜索；没有可恢复的搜索树。GPU 空闲回收的原因被移除，但 CPU 作业仍受平台最长时限、内存和抢占规则约束。原来正在运行的作业不受本次文件修改影响，不会自动迁移设备。

如明确使用允许间歇 GPU 空闲的队列，可显式加 `--device cuda:0` 继续使用原 CUDA 路径；遵守 `CUDA_VISIBLE_DEVICES`，不会后台制造 GPU 负载。

脚本没有求解时限或外层 watchdog；SCIP `limits/time=1e20` 为其默认无限值，同时 `gap=0`、`absgap=0`、节点/解数/重启数限制为 `-1`。平台的 Custom 最长运行时间仍由平台控制，提交时选择平台支持的最长档。

每个神经作业启动后会先在本作业节点运行一个 **40 变量的合成检查**，最多 30 节点，确认权重、真实特征、分支回调和选定推理后端能够工作，通过后才求解航空大实例。这个节点上限只属于合成检查；正式航空求解没有该限制。指定 CUDA 而 GPU 不可用、或模型回调异常时会明确失败，不会静默改用原生方法继续计算。

## Mamba CPU 推理适配

[mamba_cpu.py](mamba_cpu.py) 只替换本次加载的 Mamba-1 实例的前向执行：保留全部冻结参数、TreeGate 嵌入、位置嵌入、24 步历史、初始 relpscost 和确定性 argmax。卷积使用 PyTorch CPU，selective scan 按官方 v2.2.2 [Mamba 前向](https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/modules/mamba_simple.py) 和 [参考递推](https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/selective_scan_interface.py) 实现；递推使用 TorchScript 并只保留当前状态，避免按全部 token 展开高维状态数组。每次回调仍从零状态重算当前历史，保持原窗口语义。没有使用 GRU 替代或重训模型，也没有修改 `research/mamba-branching/code` 或已冻结 checkpoint。

CPU 与 CUDA 的浮点实现可能在近并列候选上选择不同动作；本次是相同权重的 CPU 部署，不能称为旧 GPU 搜索轨迹的逐动作重放，也不保证求解更快。汇总另列 `device`、`inference_backend`，新结果记录 CPU 适配器 SHA；分析耗时时披露设备差异。

可独立复核（仅参考前向与小型合成求解，不启动航空大实例）：

```bash
bash /home/duweiyue25/tgppo-transfer/scip/airwiring_compare/custom_job.sh check-mamba-cpu
```

报告保存在 [checks/mamba-cpu/validation.json](checks/mamba-cpu/validation.json)：比较官方 PyTorch 参考递推及原 `Mamba.forward`，核对冻结权重下不同候选数/完整历史的输出与动作，测试长序列，并实际执行小模型的 CPU 分支。测试中禁止 CUDA 初始化和 CUDA scan 调用。CPU 对 CUDA 的直接数值比较需要 GPU，当前未完成，不能由 CPU 参考检查代替。

本地验证已通过：4 组递推参考、5 组冻结网络参考（含 batch=2、不同候选数和完整 49 组历史），在 `rtol=1e-5, atol=1e-5` 容差内一致且测试动作相同；小模型实际 29 次分支，CUDA 未初始化。143,424 token 的完整历史合成测试约 6.80 秒，全检查进程峰值 RSS 约 572 MiB；这不是航空求解的耗时或总内存估计。原源码、数据与权重等 16 项保护哈希未变。TGPPO CPU 小模型 30 次分支及完整 CIP 读回/无限时参数/汇总回归也通过，记录在 `checks/mamba-cpu/`。

## 比较口径

`sssp` 使用上传源码的**原建模 + 原生 SCIP 后端**。源码原命令默认选 HiGHS；本次指定 SCIP，以便三组共用相同求解器。为了让三组输入逐字节一致，C++ 负责构建并导出原始 CIP，三个作业都由同一个 Python/SCIP 驱动读入 CIP；`sssp` 不挂学习分支器。它不是直接原样执行原文件中带 1800 秒和 5% gap 限制的命令。

三个作业统一：SCIP 8.0.4、定制 PySCIPOpt 4.3.0、同一模型 SHA、求解器随机种子、容差 `1e-6`、SCIP 默认 presolve/cuts/primal heuristics。保留 primal heuristics 是为了实际寻找可行解。历史 Mamba 的 MILP-S 评估关闭启发式、设置已知最优 cutoff，本次不复用那套评估限制。

TGPPO 原权重在 SCIP 6.0.1 / Torch 1.13.1 环境训练，本次部署到 SCIP 8.0.4 / Torch 2.2.2，与 Mamba 共用已有 25/8/53 特征绑定。保留原 Actor、ActorPolicy 和 Brancher 的正常决策路径；只适配 PySCIPOpt 4 的中断 API。维度相同不代表跨 SCIP 的数值完全相同，因此这是**冻结模型在航空实例上的迁移比较**。不包含最新 MLP+BC 改进模型，也不以航空结果重新选权重。

实测同一 B0 权重在原 Torch 1.13.1 与部署 Torch 2.2.2 的三组合成 CPU 输入上，概率最大绝对差为 `1.49e-8`。31 候选案例出现近乎并列，argmax 从 2 变为 0；因此**概率接近不能保证动作相同**，不能把迁移后的求解路径当作原环境逐动作复跑。原始前向输出保存在 `checks/actor-forward-{original,deployment}.json`；CUDA 上的数值差异也不能由该 CPU 检查排除。

模型固定为：

- Mamba：`research/mamba-branching/runs/milp-s-seed0-20261027/training/selected.json` 指定的 CL44 / IL50 配套 checkpoint，SHA `c1d1761e1085f8e7c92947a40d765d0d05e5a471bca618aa3fdc38e725ac2a74`。
- TGPPO：`research/tgppo/runs/server6-20261022/training/B0-s11/actor_best.pt`，开发集选 episode 100，SHA `8b605db5be6bc4a70d8f8ab7c7fa84552f8542b0676a4365a19041a101875397`。

## 输入适配和模型产物

原文件只读。新目录内的导出适配器保留原图化简、选点、变量、目标和约束生成逻辑：

1. 上传 pairs 每行有 7 列（含尾空列），实际起点/终点/density 为 0-based 列 3/4/5；原源码读取 3/5/9，会越界或读错。准备阶段验证端点和数值，生成 10 列 `pairs.normalized.csv`，映射到原读取布局，保留全部 246 点对及原 density 字符串。
2. 去掉本次不用的 HiGHS 转换、XLS 后处理和 optimize 调用，避免要求额外的 HiGHS/xlslib 开发库；使用源码内的 SCIP 变量/约束导入代码导出 CIP。没有用替代的航空建模公式重写模型。
3. 编译/导出用文件锁防止三个作业重复写入；按源码、输入、k、适配器、编译器、libscip 的哈希缓存，命中后校验文件哈希。求解时再次验证实际加载的 libscip 与导出一致。

产物在 `artifacts/model-<hash>/`：原源码快照、生成 C++、差异补丁、规范化 CSV、编译/导出日志、`model.cip`、`manifest.json`。第一次无需另交准备作业；三条正式命令都会自动准备或复用缓存。

已导出本实例：**546 个原始节点、641 条原始边、246 点对；258,164 个变量（5,736 binary + 252,428 continuous）、162,800 条约束，CIP 约 81 MiB**。这是建模结果，不是最优解或求解结果。

## 查看结果

```bash
bash /home/duweiyue25/tgppo-transfer/scip/airwiring_compare/custom_job.sh summarize
```

默认在 `runs/aviation-k4-seed0/` 保存：

- `sssp-s0/`、`mamba-s0/`、`tgppo-s0/`：各作业独立目录，每次启动生成新 `attempt-*`。
- `scip.log`、完整 `settings.set`：SCIP 原始日志和实际参数。
- `result.json`：状态、最优性证明、上下界、gap、节点、解数、首解时间、SCIP/墙时、PDI、神经分支次数、回调错误、硬件和 checkpoint/绑定/库/源码 SHA。
- `progress.jsonl`、`progress.json`：改善 incumbent 或距上次快照至少 60 秒且完成搜索节点时更新；长根节点/长 LP 期间可能暂时不更新，查看 `scip.log` 和平台状态。
- `incumbent.sol`：保存改善的最好解，结束时再保存最终最好解；最终解经过原问题约束、边界和整数性检查。目标值同时记录模型单位与原代码的 `/1e6` real 单位。
- `summary-s0.csv`、`summary-s0.json`：汇总三组已完成或当前已有的记录；未完成状态不冒充仍然存活的进程。

无 incumbent 时上下界/gap 的无穷值写为 JSON `null`，不会把未求解写成 gap=0。`proven_optimal=true` 需要 SCIP optimal 且最终解检查通过。PDI记录各自实际求解时长，不能直接当作统一 120 秒/7200 秒截断指标。监控写盘耗时也单列记录。

重复提交已完成的方法会校验并跳过；同一方法同时重复提交会被锁拒绝。被平台终止/失败的尝试全部保留；重新提交会从新搜索开始，**不宣称恢复 B&B 搜索树**。`incumbent.sol` 是解备份，不是搜索树 checkpoint。原源码的路线还原/XLS 不属于本次求解比较输出。

自定义实验可加 `--k`、`--seed`、`--edges`、`--pairs`、`--results`，三条作业务必使用相同参数。改变 k/输入时给出新的 `--results`；原生状态归一化、120秒训练时限等历史入口不参与本次运行。环境覆盖项为 `AIRWIRING_SCIP_PREFIX`、`AIRWIRING_PYTHON` 和 `CXX`。

## 本地验证范围

初始交付检查见 [validation.json](validation.json)，新增 CPU 检查见 [checks/mamba-cpu/validation.json](checks/mamba-cpu/validation.json)。CPU 适配开发没有在本地启动航空大实例求解；长时间运行仍由用户提交 Custom。旧 GPU 尝试与中断记录均保留。
