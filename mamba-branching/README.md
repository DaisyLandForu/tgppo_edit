# Mamba-Branching：MILP-S 原协议复现

本目录独立于 TGPPO。2026-10-29已完成MILP-S单训练种子闭环：475条专家轨迹、CL50/IL50、验证集选模、40对冻结模型测试及汇总。启动与恢复入口见[Custom作业指南](docs/CUSTOM_JOBS.md)和[加速训练说明](docs/FAST_TRAINING.md)。

当前模型的验证动作准确率为73.70%；测试几何均值为公平节点1704.76、耗时80.42秒，同轮relpscost为887.64、43.91秒。Mamba相对基线的公平节点和耗时分别为1.921倍、1.831倍。与论文中MILP-S弱于relpscost的方向一致，但相对差距更大；尚未验证优于其他神经基线或多训练种子稳定性。全部测试是已知最优cutoff下的无incumbent证明结束，不能说原问题无解或本次找到最优可行解。

- [正式评测报告、逐实例表格与图表](runs/milp-s-seed0-20261027/analysis/evaluation-review-20261029/README.md)
- [训练与选模核查报告](runs/milp-s-seed0-20261027/analysis/training-review-20261029/README.md)
- [仓库收录范围与迁移说明](REPOSITORY_CONTENTS.md)

用户已确认：先复现原论文的小数据集；使用 V100；新建 Conda 环境；沿用 scip 镜像和共享持久存储。GPU 已接入；开发和 Custom 节点实测均为 **1 V100 32GB / 约 8 CPU / 32 GiB RAM**。专家轨迹采集计划合计 **16 CPU / 64 GiB**，先完成一个训练种子再扩并发。

## 环境和数据

- 环境：`/home/duweiyue25/conda/envs/mamba-branching`，Python 3.10.20。
- 官方代码：`upstream/Mamba-Branching`，固定 commit `66ab2d6572eaf83c279a07c21445272e8e3cc34f`。
- 工作副本：`code/`，完整差异在 `setup/working-source.patch`，初始导入清理另存 `setup/mamba-only-imports.patch`，逐文件来源在 `setup/source-lock.json`。
- 原论文：`sources/paper.pdf` / `paper.txt`。
- 实例：`data/milp-s/`，19 train + 8 test；从已有公开实例副本复制，均匹配原下载台账 SHA256；未复用 TGPPO 轨迹或权重。
- SCIP：镜像自带 `/usr/local/scip-8.0.4`；定制接口基于 PySCIPOpt 4.3.0 独立编译。生成脚本 `scripts/port_binding.py`，来源记录 `setup/binding-port.json`。
- CUDA 依赖选择：PyTorch 2.2.2+cu118、Mamba-SSM 2.2.2、causal-conv1d 1.4.0。它们是本次兼容性选择，作者未提供完整版本锁。实际安装及检查结果见 `docs/PREPARATION.md`。

激活环境：

```bash
conda activate /home/duweiyue25/conda/envs/mamba-branching
```

Custom 入口使用环境的 Python 绝对路径，不依赖平台默认激活哪个环境。新镜像仍需挂载同一共享目录，并具备匹配的 SCIP 8.0.4 库；因此当前优先沿用 scip 镜像。

## 已准备的核查入口

CPU 检查会核对独立环境、27 个实例哈希、SCIP 版本，并在训练实例 misc03 上最多进行 10 秒、30 节点的特征/原生专家回调诊断。它不训练模型。

```bash
bash /home/duweiyue25/tgppo-transfer/research/mamba-branching/custom_job.sh cpu-preflight
```

GPU 节点到位后，使用 scip 镜像、1 V100 / 8 CPU / 64 GiB，挂载 `/home/duweiyue25`；工作目录不限，Shell 用 bash：

```bash
bash /home/duweiyue25/tgppo-transfer/research/mamba-branching/custom_job.sh gpu-ready
```

脚本先检查独立环境和当前节点 SCIP 绑定，再检查 V100 Mamba 前向/反向、有限梯度、参数更新、权重/优化器保存加载。它尊重平台的 `CUDA_VISIBLE_DEVICES`，在进程内使用 `cuda:0`。无 GPU 时返回非零并保存失败原因。单独检查 CUDA 可用原有 `gpu-preflight` 子命令。完整接续说明见 `docs/GPU_HANDOFF.md`。

**`gpu-ready` 是合成预检。** 后续的真实数据诊断已通过，命令为：

```bash
bash /home/duweiyue25/tgppo-transfer/research/mamba-branching/custom_job.sh gpu-training-pilot --capacity-instance swath1
```

它要求已有对应的采样诊断文件，覆盖 HDF5、CL/IL、100 状态/batch32、固定数据恢复和训练实例求解。容量检查重复同一个真实窗口 32 次；不是 32 条独立轨迹或收敛训练。新节点先跑 `gpu-ready` 即可，不必重复这些诊断。正式训练/恢复和验证集选模入口已经另行实现，命令为 `prepare / collect / build-data / train / evaluate / summarize / status`。

每次预检独立写入 `runs/preflight/*.json`，保留失败记录；不会覆盖成功证据。

## 协议与已完成的流程

论文设置和来源见 `configs/paper-milp-s.json`；上游差异见 `docs/SOURCE_AUDIT.md`。单种子冻结配置为 `configs/formal-milp-s.json`；逐 Custom 资源/命令在 `configs/custom-jobs-milp-s-seed0.json`。训练默认 `--resume auto`，CL 与 IL 分别 50 epoch，只用 solver seed4 选模，冻结后才做测试。

本轮已经完成：

1. 独立环境、SCIP-8定制特征、CPU采样、V100与A100核函数和真实数据诊断。
2. 全475条轨迹采集与索引，380条训练、95条验证，按100状态上限分窗。
3. CL50/IL50及中断恢复；加速入口预加载数据、有序预取并验证IL微批8，逻辑batch32及FP32不变。
4. 仅按solver seed4验证损失选模（CL第44轮、IL第50轮），冻结同轮配套embedding/Mamba后执行全部40对测试并汇总。

论文采样矩阵为19实例×5个随机探索前缀×5个求解器种子，共475条轨迹；本轮已全部完成。CL共293,346个有效训练状态，IL共3,130个训练窗口。5个测试求解器种子不是5个独立训练种子；本轮只有训练seed0。

正式评估使用 1 小时时限、关闭 primal heuristics、提供已知最优 cutoff；默认主表在根节点使用 relpscost，纯神经版本另外标记。TGPPO 的 120 秒结果及其归一化基线不能直接作为本轮结果。

## 来源

- [ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/b2f56a345fe0ce337df3bdfb31e2b350-Paper-Conference.pdf)
- [作者仓库](https://github.com/doctor-watson626/Mamba-Branching)
- [PySCIPOpt 4.3.0](https://github.com/scipopt/PySCIPOpt/tree/v4.3.0)
- 特征公式来自保留的 T-BranT / TreeGate 定制绑定；编译源、修改脚本和哈希记录都在本目录。
