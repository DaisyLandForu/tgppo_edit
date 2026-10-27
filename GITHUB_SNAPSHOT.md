# 2026-10-27 完整项目快照

本次上传来自 `/home/duweiyue25/tgppo-transfer/research`，保留当前项目文件和历史实验记录。根目录原有 `README.md` 是 10 月 22 日迁移时的说明，按原样保留；本次 GitHub 收录范围以此文档及 `snapshot/PROJECT_MANIFEST.json` 为准。

Git 中收录源码、冻结配置、全部实例数据、文档图表、CSV/JSON 结果、原始日志和构建材料。模型权重 `*.pt` 和原始轨迹 `*.pkl` 放在完整项目归档中，避免超过普通 Git 的单文件限制。归档同时保存 Git 中的原始项目文件，便于直接还原整个研究目录；它不包含研究目录之外的 Conda 环境。

完整归档位于 [full-project-20261027 Release](https://github.com/DaisyLandForu/tgppo_edit/releases/tag/full-project-20261027)。归档在全部上传、核对后发布；发布前的草稿不代表交付完成。`snapshot/PROJECT_MANIFEST.json` 记录每个原始文件的 SHA-256、大小、权限和每个符号链接的目标；`snapshot/SHA256SUMS` 记录归档分片及清单的哈希。

## 获取完整目录

需要 GitHub CLI、Python 3.8 或更新版本，以及 `zstd` 命令。还原目标必须是新的或空的目录，以保护已有实验记录。

```bash
git clone https://github.com/DaisyLandForu/tgppo_edit.git
cd tgppo_edit
gh release download full-project-20261027 \
  --repo DaisyLandForu/tgppo_edit --dir snapshot-download
python snapshot/restore_snapshot.py \
  --assets snapshot-download --output ../restored-research
```

只验证归档而不解压：

```bash
python snapshot/restore_snapshot.py \
  --assets snapshot-download --verify-archive
```

还原程序先核对清单与每个分片的 SHA-256，再检查归档内所有文件、目录和符号链接，并在解压时逐文件验证哈希。归档保存的已编译求解器与绑定属于当时服务器的环境产物，迁移到其他环境时仍按交接文档重建。

## 研究入口

- 当前服务器代码：`tgppo/experiments/server-20261022/`
- 服务器运行与 Crater Custom 作业：`migration/2026-10-22/server-run/`
- 阶段 5 汇总结果：`tgppo/runs/server6-20261022/stage5-results-20261023/`
- 六个训练作业、选模、测试及失败后补测记录：`tgppo/runs/server6-20261022/`
- 本周汇报、方法图和结果图：`docs/` 与 `docs/weekly_20261023/`

本次操作只做项目归档与上传，不启动新实验、不改变阶段验收状态。原始目录中的日志、报告、配置及历史 SHA 台账保持原样。
