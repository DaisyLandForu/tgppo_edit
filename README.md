# tgppo-transfer 项目快照｜2026-11-06

这是当前工作目录的新分支快照，保留原有Git历史，未修改主分支或旧研究分支。与旧分支将research内容平铺在根目录不同，本分支保留当前工作目录层级，使report的相对引用可用。

- [组会材料目录](report/README.md)：18页PPT、离线网页幻灯片、完整汇报、讲稿、CSV和图表。
- [完整组会汇报](report/TGPPO_组会汇报_20261106.md)：从系统诊断到轻量Actor及全66结果、持续监督消融。
- `research/`：研究源码、实验配置、训练及评测日志、结果、文档和实例数据。接续前先读[研究约定](research/AGENTS.md)。
- `scip/`：当前独立SCIP／布线相关源码、数据和结果。

## 收录范围

report目录全部文件按原样上传，包括PPT、HTML、PDF／SVG／PNG图、底表与来源审计。其余收录当前源码、脚本、配置、文档、MILP实例和CSV／JSON／文本结果；不启动任何训练或求解。

模型权重／checkpoint、原始大轨迹（PT／PKL／H5等）、安装wheel、依赖环境、编译产物和临时缓存暂不上传，原件仍在实验机器。所有省略项与原因见[snapshot/OMITTED_FILES.json](snapshot/OMITTED_FILES.json)。这不是可直接恢复全部训练状态的全量二进制归档。

一份超过Git单文件限制的JSONL日志无损gzip保存，映射和SHA见[snapshot/COMPRESSED_LARGE_LOGS.json](snapshot/COMPRESSED_LARGE_LOGS.json)。需要时运行`python snapshot/restore_large_logs.py`还原原路径，脚本拒绝覆盖既有文件。机器临时目录链接不上传；指向本项目内部的绝对链接在Git中改为相对链接，原始目标记录在清单中。

逐文件Git对象、模式和源文件信息见[snapshot/WORKSPACE_MANIFEST.json](snapshot/WORKSPACE_MANIFEST.json)；report另有逐文件SHA-256。仍在运行的scip实验输出取只读时间点副本，不停止作业、不把快照当作任务完成，记录见[snapshot/LIVE_OUTPUT_SNAPSHOTS.json](snapshot/LIVE_OUTPUT_SNAPSHOTS.json)。历史报告里的模型／原始轨迹链接可能指向本次省略的文件；这不表示它们已重新训练或丢失。环境与SCIP绑定需按research中的说明重建，不承诺clone后无需准备即可运行。

## 当前结论

无BC轻量MLP相对共同120秒预算B0，Actor在线成本降低92.83%，证明率提高3.79个百分点，共同Actor57实例PDI改善3.10%；全66中点改善1.08%但积分测量界跨0。BC初始化与持续专家监督未形成全量／跨种子稳定增量。当前不是论文完整训练协议的性能复现，也未全面超过relpscost。以report和最新阶段核验为准，旧快照中的计划状态仅代表当时进度。
