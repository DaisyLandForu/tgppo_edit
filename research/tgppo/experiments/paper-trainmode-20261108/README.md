# TGPPO 按正文训练：恢复 Actor train 模式

2026-11-08。用户已选择“论文正文＋标准 GAE”。独立准备包，不从旧 PPO 检查点续训；旧六作业保持原样。

- [启动与恢复](docs/CUSTOM_JOBS.md)、[六条命令](docs/commands.txt)、[断点续训说明](docs/RESUME.md)。默认 CPU 更新，避免平台因 GPU 闲置回收；CUDA 可显式选择。
- [冻结协议](configs/protocol.json)、[正文对齐与公开代码边界](docs/PROTOCOL_ALIGNMENT.md)。
- [哪些差异影响结果](docs/TRAINING_DIFFERENCES.md)。
- 输出：`research/tgppo/runs/paper-trainmode-20261108/`。
- 本地检查与准备报告：`research/docs/methods/tgppo_paper_trainmode_20261108/`。

B0／MLP／MLP＋BC，种子11／22；25实例×solver0–4，500完整episode、3600秒、每episode后3epoch AdamW。正式主比较固定500，不按66测试结果选模。

Actor 采样／部署为eval，PPO更新为train；B0有效训练dropout=.05，MLP结构无dropout，仍为0。所有方法使用相同GAE／episode和优化规则。新日志分别记录初次更新的dropout概率比扰动，以及eval分布更新前后变化。

真实训练和正式测试尚未执行；CPU检查通过不等于方法有效或已完整复现作者实验。
