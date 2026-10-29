# 环境准备与 GPU 验证记录

独立 Conda 环境与 GPU 验证已完成；正式采样、训练恢复、验证集选模和评估入口已补齐，并通过真实小规模闭环/跨进程检查。下一步可提交并行 Custom，详见 **[作业配置与启动命令](CUSTOM_JOBS.md)**。正式 MILP-S 的 475 条采样和 50+50 epoch 尚未启动。

本轮完整结果、报告链接和验证边界见 [GPU_VALIDATION.md](GPU_VALIDATION.md)；机器可读接续状态在 `CURRENT_STATE.json`，历史暂停快照保存在 `PAUSED_STATE.json`。

## 已完成

- 新建 `/home/duweiyue25/conda/envs/mamba-branching`，Python 3.10.20；基础依赖安装成功。没有修改 TGPPO 环境。
- 固定作者源码 commit `66ab2d6572eaf83c279a07c21445272e8e3cc34f`，保留原版和独立工作副本；完整差异在 `setup/working-source.patch`。
- 独立移植和编译 PySCIPOpt 4.3.0 特征接口，链接镜像自带的 SCIP 8.0.4。最终编译补丁和来源哈希在 `setup/scip8-features.patch`、`setup/binding-port.json`。
- 复制并核验 MILP-S 全部 19 个训练实例、8 个测试实例及测试 cutoff。实例来源和哈希在 `data/milp-s/manifest.json`。
- 按论文生成 475 条待采样任务清单，380 train / 95 validation；正式任务尚未提交。
- 修正官方启发式关闭清单遗漏的两个 SCIP-8 插件，实测全部 `heuristics/*/freq` 已关闭。
- 追踪并核对评估窗口：上游 `max_seq_len=49` 个交替候选/动作组，对应 24 步历史加当前状态。已更正此前把它记作 49 步历史的错误判断。
- 更新后 CPU 特征预检通过，最新报告路径见 `CURRENT_STATE.json`；之前的成功和失败报告均保留。
- Torch 2.2.2+cu118、Mamba-SSM 2.2.2、causal-conv1d 1.4.0 及全部 CUDA 依赖已安装。12 个配套包匹配 PyPI SHA256，Torch 主包匹配官方索引 SHA256；缓存清单位于 `setup/wheels/`。
- `pip check` 通过；环境锁定文件已生成：`setup/pip-freeze.txt`、`setup/conda-explicit-linux64.txt`。Conda explicit 文件记录本共享存储的本地包来源；不是任意外部机器都可直接访问的 URL 清单。
- CUDA 扩展导入、CPU 模型构造、候选编码前向通过，环境报告为 `runs/environment/20261027T143644.084074Z.json`。候选编码器 19,816 参数、序列模型 4,400 参数。
- 两组真实专家采样通过：misc03 / seed0 / 随机前缀0 采到15条，misc03 / seed4 / 随机前缀5 采到9条；仅为各最多10秒/30节点的诊断。没有正式采样任务提交。
- HDF5 转换、训练/验证分开读取及批处理通过；修复了浮点存储标签转 long 的类型兼容错误。数据诊断报告为 `runs/data-pilot/20261027T144123.379346Z/report.json`。实际序列只有15状态，不能冒充已验证完整100状态批。
- 8 个测试 cutoff 已独立核对 MIPLIB/CORAL 官方记录，原始来源另存 `sources/`，核对结果在 `setup/cutoffs-audit.json`。另发现旧参考文件中的一个训练实例值不一致，见 `DATA_AUDIT.md`。

## 当前验证边界

开发服务器已实际执行 CUDA 前后向、真实 HDF5 数据上的 CL/IL 更新、FP32 的 100 状态/batch32 更新、固定数据恢复，以及根节点 relpscost + 24 步历史的神经 SCIP 求解。misc03 和 lseu 均求解到最优；只使用训练实例。Custom 节点目前验证到合成 `gpu-ready`，真实训练诊断在开发服务器完成。

最终 swath1 容量诊断为一个真实 100 状态窗口重复 32 次，每状态 2–27 候选，峰值分配显存 194.63 MiB。这不是 32 条独立轨迹，也不能保证全部实例的候选规模均适用。恢复检查的加载/RNG/下一窗口/下一步 loss 一致；更新结果有浮点差异，不能声称逐位确定性。

已验证的 `code/paper_training.py` 修复 padding，IL 按论文轨迹内求和、轨迹间均值；与上游差异见 `SOURCE_AUDIT.md`。正式入口已集成损失适配、独立失败记录、checkpoint/数据顺序恢复与验证集选模。新入口在 2+2 epoch 诊断中分别中断 CL、IL 并用新进程恢复，参数/优化器与连续运行数值一致；梯度累积等价性、重复作业锁和缺失/损坏数据拒绝检查通过，证据在 `runs/entry-check-20261027-final/verification.json`。完整正式数据规模尚待运行。8 个测试实例未用于调试或选模。

## GPU 接续入口

用户已提交的 Custom 预检通过，同一节点不必再次运行。新节点沿用 scip 镜像和共享挂载，可执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/mamba-branching/custom_job.sh gpu-ready
```

具体范围和后续顺序见 `GPU_HANDOFF.md`。`gpu-ready` 不自动启动正式训练；真实数据诊断另有 `gpu-training-pilot --capacity-instance swath1`，现已有成功证据，无需日常重复执行。
