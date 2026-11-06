# 本轮训练结果审阅

> 后续状态：用户已授权本地 CPU 追加 90 分钟，使用独立 Silver4214 cohort 统一重测 160 模型＋10 新原生。本目录保留原训练审阅与 Gold6148 不完整评估口径；最新执行/完整结果入口见 [../continuation/README.md](../continuation/README.md)，不要从本目录旧进度判断当前任务是否完成。

用户已进一步授权：收齐现有评估并分析各增量，证据满足跨种子/跨实例条件后才冻结候选做小规模跨实例确认。新[推进规则](../continuation/PLAN.md)和完整评估后的`../continuation/DEVELOPMENT_REVIEW.md`为后续决策入口；以下动态报告仍保留前次交付口径，不将其旧“待授权”说明当作当前限制。

入口：[报告](REPORT.md)、[周五汇报摘要](MEETING_BRIEF_20261106.md)。训练种子11、22的16组已全部完成；开发求解进度以[evaluation_status.json](evaluation_status.json)及原始RUN的`results/complete.json`为准。

当前只读汇总进程等待**已经由最后一个Custom启动的CPU评估**，每30秒更新报告；不会启动GPU、PPO、SCIP或新作业。完整结果或现有评估停止/到限后退出。进程状态与日志：[review_process.json](review_process.json)、[summarize.log](summarize.log)。不能仅凭旧进程状态宣称评估仍在运行。

文件包含[训练统计](training.csv)、[BC统计](bc.csv)、[PPO后专家信号](retention.json)、[来源SHA与核验](verification.json)。统一评估完整后生成`development_summary.csv`、`development_by_instance.csv`、`paired.csv`；部分评估只导出已提交批次，不据此排名。

若仅需从已有文件重新生成报告，在项目根目录运行（标准库，无需求解器环境）：

```bash
python research/docs/methods/tgppo_light_actor_20261105/results_review/summarize.py
```

不要为此重跑训练或`finalize`。未进入66实例测试，没有启动下一阶段实验。
