# TGPPO 复现与搜索信息 Critic 实验

本目录的学校服务器直传压缩包包含可迁移的**源码、协议、阶段报告、25 个训练/开发 MPS 实例、66 个测试 MPS 实例及交接材料**。它不包含旧模型权重、轨迹或已编译求解器。解压成功只代表拿到材料，不能代表服务器环境和实验已验证。

今晚的明确训练范围是 B0、R0、Rjoint × 训练种子 11/22，共六个独立 GPU 作业；每个 500 个完整 episode，在 100/200/300/400/500 保存检查点。按开发集为**每个训练种子**选最好权重，保留固定 500 episode 权重，再用冻结 Actor 在未见测试集上评估。目标 V100/Slurm 尚未验证，旧阶段 5 九作业矩阵未完成。

先读：

1. [学校服务器交接文档](migration/2026-10-22/HANDOFF_SERVER_2026-10-22.md)
2. [可复制给服务器 Codex 的任务说明](migration/2026-10-22/PROMPT_服务器Codex.md)
3. [六作业启动包](migration/2026-10-22/server-run/README.md)
4. [项目约定](AGENTS.md)及[阶段验收安排](docs/阶段交付与验收_TGPPO复现与增量改进.md)

当前研究代码在 `tgppo/experiments/server-20261022/`；历史研究版本保存在 `tgppo/experiments/stage2` 至 `stage5`，上游原版在 `tgppo/repo/`。定制 PySCIPOpt 源码在 `tgppo/experiments/stage4/metrics_binding/`。不要用原版启动脚本代替研究入口。

直传包的文件范围与 SHA 清单见[迁移清单](migration/2026-10-22/ARCHIVE_CONTENT_MANIFEST.json)。服务器上重建 Python、SCIP/SoPlex 与定制绑定，再按交接文档的顺序验证、提交作业。后续 GitHub 同步另行完成，不把它当作今晚运行的前置条件。
