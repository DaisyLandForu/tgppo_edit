# TGPPO 第一阶段诊断交付

先读 [一页组会摘要](MEETING_BRIEF_20261030.md)，再按需查看：

1. [00：实验身份、协议与证据缺口](00_inventory_and_protocol.md)
2. [01：论文局限与本地证据](01_paper_limitations_evidence_matrix.md)
3. [02：PPO、H3、Actor 的实测诊断](02_training_reward_actor_diagnostics.md)
4. [03：成本、实例配对与求解归因](03_runtime_and_solver_diagnostics.md)
5. [04：最多三项假设、最低成本方案与停止条件](04_hypotheses_and_low_cost_validation.md)

E1=已测事实，E2=代码/论文机制，E3=待验证假设，E4=目前支持不足/被否定。报告是待验收交付；第二阶段干预未执行，不把验收等待当作已获批准。原基线、实验权重、日志和数据划分未修改。

## 复现命令

工作目录为 `/home/duweiyue25/tgppo-transfer/research`。脚本仅写本目录，重新执行会更新这里的派生产物；不会更新历史结果。没有 SCIP solve 或 optimizer step。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -u docs/diagnostics/tgppo_20261030/scripts/analyze_existing.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -u docs/diagnostics/tgppo_20261030/scripts/audit_protocol_reward.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/duweiyue25/conda/envs/tgppo/bin/python -u docs/diagnostics/tgppo_20261030/scripts/actor_offline.py
MPLCONFIGDIR=docs/diagnostics/tgppo_20261030/logs/mpl-cache PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /tmp/mamba-result-plot-env-20261029/bin/python docs/diagnostics/tgppo_20261030/scripts/plot_diagnostics.py
PYTHONDONTWRITEBYTECODE=1 python docs/diagnostics/tgppo_20261030/scripts/validate_delivery.py
```

绘图复用已经存在的独立 Matplotlib 临时环境，没有给训练 Conda 环境安装依赖；该 `/tmp` 路径未来可能消失，可替换为具有 NumPy/Matplotlib 的独立 Python 环境。本轮实际版本见 [environment.json](environment.json)。Actor 诊断使用 CPU Torch 1.13.1，与历史 Torch 版本相同；没有导入/运行求解器，不能把这当作当前 SCIP ABI 检查。

论文按交接指定 URL 联网取得 v1，保留 [HTML](sources/tgppo_arxiv_2511.12986v1.html) 和 [可检索文本](sources/tgppo_paper.txt)；原文 CC BY 4.0，作者与来源在 [01](01_paper_limitations_evidence_matrix.md) 标明。没有查询/实装 HSTU。

## 指标、图与核验

原始源文件哈希见 [source_hash_audit.csv](source_hash_audit.csv)；历史完整结果入口见 [原权威报告](../../阶段5_服务器六作业结果与验收报告_TGPPO_2026-10-23.md)。新报告中每个统计均链接对应 CSV/JSON；固定观测的来源行号、轨迹哈希与 checkpoint SHA 均落盘。

- [图 1：学习链与开发曲线](figures/01_learning_diagnostics.png)
- [图 2：配对结果与成本](figures/02_runtime_solver_diagnostics.png)
- [图 3：历史 gap–time 示例](figures/03_gap_time_examples.png)

三图各有同名 SVG/PDF 可供组会导出。图的计时/采样局限在 [02](02_training_reward_actor_diagnostics.md) 与 [03](03_runtime_and_solver_diagnostics.md) 说明。数值/链接/逐任务交叉核验见 [validation.json](validation.json)；全部新交付文件的最终 SHA 见 [delivery_manifest.json](delivery_manifest.json)。manifest 不包含自身；CPU/GPU/SCIP 成本没有混加。

本轮已有统计约 230 秒、Actor 检查约 41 秒，另有文献读取、元数据核查和绘图开销；均不是新增训练或求解预算。后续仅在第一阶段验收后，按 [04](04_hypotheses_and_low_cost_validation.md) 选定范围继续。
