# TGPPO 代码与报告

代码与文档更新：2026-11-18。本分支在2026-11-06完整项目快照上追加增量提交，保留现有历史和目录结构。

- [最新六作业续训命令](research/tgppo/experiments/paper-device-resume-20261116/commands.txt)：seed11 V100、seed22 CPU，按最新设备分支检查点续接；CPU PPO更新支持多线程。
- [论文正文主设置对齐实现](research/tgppo/experiments/paper-trainmode-20261108/docs/TRAINING_DIFFERENCES.md)：明确列出与作者公开入口的差异，不能理解成已复现作者正式模型和指标。
- [固定200轮、完整66实例结果](research/docs/methods/tgppo_paper200_test66_20261113/RESULTS_REVIEW.md)。
- [seed11训练曲线与进度快照](research/docs/methods/tgppo_seed11_training_20261118/REPORT.md)：2026-11-18 13:24北京时间快照，不是实时状态或500轮完成声明。
- [B0弱响应原因与复现边界](research/docs/methods/tgppo_seed11_training_20261118/CAUSE_AND_REPRODUCTION.md)。
- [组会材料](report/README.md)，含方法说明、幻灯片、图表及评测摘要。

本次追加更新源码、脚本、配置、诊断/方法报告和report材料；没有上传模型权重、checkpoint、原始PT/PKL轨迹或新的运行目录。旧提交中已有的实验记录和依赖源码保留。工作区原件与正在执行的训练保持原样；断点续训仍需实验机器上的模型、优化器、随机状态和轨迹文件。

这不是不需要环境准备即可执行的二进制归档。需要按研究文档准备SCIP定制绑定、Python依赖和数据。详细增量清单见[snapshot/CODE_UPDATE_20261118.json](snapshot/CODE_UPDATE_20261118.json)；旧完整快照清单仍描述2026-11-06时点。
