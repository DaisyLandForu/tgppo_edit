# Git收录范围与迁移说明

本模块提交到`DaisyLandForu/tgppo_edit`的独立分支`mamba-branching-reproduction-20261029`，仓库内路径为`mamba-branching/`。服务器上的原工作目录仍为`research/mamba-branching/`。第三方源码的来源、固定版本及差异见`setup/source-lock.json`、`setup/binding-port.json`和补丁。

## 收录内容

- 工作代码、正式采样/训练/恢复/评估脚本、内存缓存/预取/微批优化、Custom资源配置及文档。
- SCIP8定制PySCIPOpt的可编译源文件、上游源文件参考、特征移植补丁、环境版本与依赖安装脚本；不含编译输出。
- 27个公开MILP-S实例、cutoff、来源清单、论文及公开最优值资料。
- 正式运行和诊断的轻量配置/提交记录/选模指针/哈希、训练验证逐轮CSV、全部80次求解指标、评测图表及报告。诊断失败记录保留，不以成功记录覆盖失败历史。

## 保留在原持久存储的内容

Git不收录模型/优化器checkpoint（`*.pt`）、专家轨迹（`*.h5`/`*.pkl`）、采样顺序和数据索引（`*.npy`）、逐更新JSONL日志、Conda环境、依赖wheel、SCIP二进制、生成C文件或编译缓存。部分原始元数据会引用这些文件；这些引用和哈希用于溯源，不表示被引用的大文件已在Git中。

最终权重保存在原服务器RUN中，入口为`runs/milp-s-seed0-20261027/training/selected.json`。要恢复原训练或重新调用原模型，需保留/另行复制完整RUN（含权重、轨迹和索引）及匹配环境。旧TGPPO的`full-project-20261027` Release不包含后来创建的本模块，不能作为Mamba模型备份。

## 新机器使用

不要覆盖或重新生成已冻结RUN的配置、源码指纹和run.json。原命令中的`/home/duweiyue25/...`是实测服务器路径，迁移时根据实际位置设置`ROOT`、`MAMBA_ENV`等；绑定路径、包版本与二进制哈希变化会使原RUN严格校验失败。重新复现实验应使用独立的新RUN，而不是改写历史来源记录。

先创建Python3.10的独立Conda环境，准备SCIP8.0.4，使用`setup/PySCIPOpt-4.3.0-features`源文件和`scripts/install_dependencies.sh`安装依赖，再执行CPU/GPU预检。环境版本见`setup/pip-freeze.txt`及`setup/conda-explicit-linux64.txt`。`prepare_source.py`和`port_binding.py`属于来源生成工具；现成工作代码和绑定源文件已收录，无需对它们重复生成或覆盖。

本次提交是代码和轻量结果同步，不是完整可恢复的二进制运行快照。逐文件收录清单及SHA256见`REPOSITORY_MANIFEST.json`。
