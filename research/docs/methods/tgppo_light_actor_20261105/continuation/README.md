# 轻量 Actor 方法评估续接入口

> 后续阶段已完成20项BC-only消融，见[新阶段入口](../bc_only_ablation_20261105/README.md)。当前新证据尚未支持PPO相对BC本身的稳定收益；本页以下为先前开发＋8实例复核阶段的冻结交付口径。

**本阶段已完成并停止：开发170/170、跨实例144/144项全部有效。** 优先阅读[周五汇报摘要](../MEETING_BRIEF_20261106.md)和[结果解读](RESULTS_REVIEW.md)。MLP+BC相对B0跨实例PDI−6.25%，BC相对MLP再降2.18%；证明率仍37.5%，原生75%，尚未超过原生。下一步20项BC-only CPU对照仅为提案，未执行。

用户已授权本地 CPU 追加累计 90 分钟统一重测：160 项冻结模型任务、10 项本轮新测原生 relpscost。当前 cohort 为 Silver4214；旧 Gold6148 的不完整结果和预算保留，不合并绝对计时。

- 当前执行状态：[progress.json](progress.json)。`valid_model_tasks_raw` / `valid_native_tasks_raw` 是实际有效的逐任务进度；整组提交数在全部任务结束后才更新，不能把其暂为 0 解释为没有进展。
- 本轮结果位置：[CURRENT_EVALUATION.json](CURRENT_EVALUATION.json)。只有其 `results_directory` 中存在 `complete.json`，且运行状态为 complete，才视为开发评估完成。
- 实际网络、专家初始化、PPO和消融口径：[METHOD.md](METHOD.md)。
- 面向方法汇报的结果解读：[RESULTS_REVIEW.md](RESULTS_REVIEW.md)。
- 完整开发评估后的结论：[DEVELOPMENT_REVIEW.md](DEVELOPMENT_REVIEW.md)、[decision.json](decision.json)、结构/BC分解 `independent_effects.csv`、成本 `deployment.csv`。报告尚未生成时，不从进行中的子集给最终排名。
- 推进与停止规则：[PLAN.md](PLAN.md)。仅通过后执行冻结结构的完整 2×2 跨实例探索复核；否则新增跨实例任务为 0。条件式阶段仍有独立累计 90 分钟和北京时间 22:00 截止，不自动重试。
- 开发评估已完成，选择MLP+BC完成已授权的跨实例确认；终态见[cross_progress.json](cross_progress.json)，完整报告见[CROSS_INSTANCE_REPORT.md](CROSS_INSTANCE_REPORT.md)。开发完成及170项归档核查见[development_completion_audit.json](development_completion_audit.json)。
- 恢复过程及存储问题：[RECOVERY.md](RECOVERY.md)；冻结源码检查：[source_verification.json](source_verification.json)。无需新 Custom / GPU / 训练。
- 最终完整性、源码/权重哈希、归档及进程退出核验：[FINAL_AUDIT.json](FINAL_AUDIT.json)。

原训练、BC、专家信号检查及 Gold6148 的历史局部审阅见 [../results_review/](../results_review/README.md)。既有报告中的“尚未授权”或原生“缓存”描述不适用于本次已授权的同机新测 cohort。
