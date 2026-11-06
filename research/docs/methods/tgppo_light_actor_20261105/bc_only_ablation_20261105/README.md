# BC-only小消融交付

用户已授权并完成20项本地CPU消融，16模型＋4本轮原生全部有效，累计701.26/1200秒，无新增训练/GPU/自动重试，源权重和冻结基线不改。

- [结果报告](REPORT.md)：逐实例、逐训练种子PDI与证明率。
- [训练、选模、评测流程与时间](TRAINING_AND_EVALUATION.md)：100episode用途、为何暂缓盲目长训、GPU分工和资源估计。
- [分析数据](analysis.json)、[配对CSV](paired.csv)、[全部任务](tasks.csv)、[事先写定的范围与停止条件](PLAN.md)。

结果：PPO100对BC-only在4个实例×训练种子配对中1改善、3退步，合并PDI+3.53%；这是两个已查看实例的机制消融，不是新的总体泛化结论。

当前已停止。先复用已有BC/episode20/episode100在5开发实例补70项CPU学习曲线，累计40分钟，只是下一阶段提案；没有启动该阶段、长训练或全66评测。

本次RUN：`research/tgppo/runs/light-actor-20261105/bc-only-ablation-20`，保存完整原始JSON、部署副本、来源和预算。`raw`为本地缓存链接，永久归档为同目录`raw_results.tar`及`evaluation_rows.json`。
