# TGPPO 论文主协议对齐：B0／MLP／MLP＋BC

2026-11-06。独立新实验，旧基线、权重与报告只读。准备完成后由用户手动提交，当前没有真实新训练或求解结果。

- [Custom启动命令](docs/CUSTOM_JOBS.md)：6个训练作业；恢复、状态及后续CPU评测入口。
- [对齐协议与剩余边界](docs/PROTOCOL_ALIGNMENT.md)：全部25实例、5个solver seeds、500完整episode、3600秒、AdamW、逐episode更新。
- [冻结配置](configs/protocol.json)、[作者easy/hard发布数据核查](docs/author_subset_audit.json)。
- 运行输出：`research/tgppo/runs/paper-aligned-20261106/`。
- 准备检查与未来评测报告：`research/docs/methods/tgppo_paper_aligned_20261106/`。

只比较固定episode500，不使用原5开发实例或66测试集选权重；训练完成后另行执行CPU正式评估。不能把“脚本已就绪”写成“论文性能已复现”。
