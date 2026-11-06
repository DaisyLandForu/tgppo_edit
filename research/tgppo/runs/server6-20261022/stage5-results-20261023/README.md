# 阶段 5 实际结果（2026-10-23）

[中文完整报告](../../../../docs/阶段5_服务器六作业结果与验收报告_TGPPO_2026-10-23.md)。

七路补齐：66 实例、3 个求解器种子、1386 条有效记录。R0/Rjoint 相对 B0 未显示稳定收益；原生 relpscost 整体更好。详见 `main_results.csv` 与 `per_training_seed.csv`。

- `audit.json`：覆盖、冻结、资源例外与软件身份。
- `per_task_with_sources.csv`：所有最终任务及其原始文件和 SHA。
- `preserved_failures.csv`：原 21 条 mspp16 失败，没有删去。
- `per_instance.csv`、`paired_instances.csv`：按实例比较。
- `main_results.png/svg/pdf`、`development_curves.png/svg/pdf`：由真实聚合数据生成；后者仅开发集曲线。
- `training_cost.csv`、`deployment_cost.csv`、`summary.json`：成本、计时及各训练种子结果。
- `source_ledger.json`：只读输入来源；`deliverable_sha256.json`：本次交付文件哈希。

复算（不启动求解、只写本目录分析结果）：

```bash
cd /home/duweiyue25/tgppo-transfer/research
source migration/2026-10-22/server-run/env.sh
/home/duweiyue25/conda/envs/tgppo/bin/python tgppo/runs/server6-20261022/stage5-results-20261023/analyze.py
```

绘图需要 Matplotlib；本次使用独立临时环境 `/tmp/tgppo-stage5-plot-20261023`，不依赖其持续存在，已保存 PNG/SVG/PDF。运行 `plot_results.py`、`write_report.py` 可由分析产物重建图和报告。重新生成后需更新交付文件哈希。阶段 5 待验收，阶段 6 未启动。
