# 实验室服务器迁移清单

## 2026-10-22 最新直传包（优先阅读）

用户改为先直接上传压缩包到学校服务器，GitHub 同步待仓库写入授权完成后另做。最新包是本目录 `tgppo_research_handoff_2026-10-22.tar.gz`，校验文件为同名 `.sha256`，逐文件清单在 `ARCHIVE_CONTENT_MANIFEST.json`；包内根目录为 `research/`。它包含六作业新代码与脚本、全部 25 个训练/开发及 66 个测试 `.mps.gz` 实例、cutoff、文档和 SCIP/SoPlex 源码，因而**不需要再单独传 MPS 数据才能训练**。旧 `.pt`/`.pkl`、编译二进制、Git 内部元数据和 Conda 环境不包含。

服务器上先用 `sha256sum -c tgppo_research_handoff_2026-10-22.tar.gz.sha256` 验证压缩包，再解压到选定目录：`tar -xzf tgppo_research_handoff_2026-10-22.tar.gz -C /目标父目录`。进入解压出的 `research/`，执行 `python migration/2026-10-22/verify_server_archive.py` 核对逐文件 SHA，再阅读 `HANDOFF_SERVER_2026-10-22.md`，把 `PROMPT_服务器Codex.md` 中的整段任务交给服务器 Codex。包内源码不等于已编译/已运行环境；正式作业前必须重建绑定、测原生基线和通过目标 V100 桥接。

下面的旧清单与旧压缩包说明是迁移过程的历史记录。旧 `/mnt/data/research_2026-10-21.tar.gz` 不含此轮六作业代码，不能作为今晚的交接包。

日期：2026-10-22。当前是“工程闭环已打通、阶段 5 正式对照未完成”。本文件给出迁移材料与步骤，不启动新实验，也不表示已在目标服务器验证。

## 需要带走什么

| 材料 | 路径 | 作用 |
|---|---|---|
| 文档、冻结协议、阶段报告、原始审计和补丁 | `docs/` | 研究设计与证据，避免新会话误解已完成范围 |
| 原版及修复/增量代码（保留隐藏 .git/worktree 数据） | `tgppo/repo/`、`tgppo/experiments/` | stage5 为当前主线；不能仅从 GitHub clone 原版 |
| 定制绑定源码 | `tgppo/experiments/stage4/metrics_binding/`、`tgppo/setup/PySCIPOpt/` | 需包含 .pyx/.pxd/.pxi/setup.py；普通同版本号发行包不能代替 |
| 求解器源码、版本与构建资料 | `tgppo/setup/scip-6.0.1/`、`soplex-4.0.1/`、`source_versions.json`、构建日志 | 新系统 ABI 不一致时重新编译 |
| 数据、cutoff、来源和哈希 | `tgppo/datasets/` | 25 个原训练实例、66 测试实例；本轮20/5划分另见冻结协议 |
| 配置及入口 | `tgppo/configs/`、`tgppo/activate.sh`、`docs/stage5/` | 根路径、绑定优先级、任务配置 |
| 模型、轨迹、缓存、评估和消耗记录 | `tgppo/runs/` | 可复查/恢复；包含基线、校准统计、离线结果及阶段5局部模型 |
| 项目接续记忆 | 根目录 `AGENTS.md`、本目录 `CODEX_HANDOFF.md` | 新 Codex 会话先读 |
| 环境描述 | 本目录 `environment/`、原 `tgppo/setup/` 环境清单 | Conda 环境在 `/root/anaconda3/envs/tgppo`，不在旧工程压缩包里 |
| 原始 Codex 项目会话（可选） | 补充包 `codex_sessions/` | 仅本项目 cwd 对应的会话备份，不含登录凭据文件 |

已有 `/mnt/data/research_2026-10-21.tar.gz`（4,006,540,352 字节）是原工程快照，包含当时目录中的隐藏文件，但不包含之后新增的实验产物、本次交接说明、Conda 环境及 `/root/.codex`。

迁移补充包只补充交接说明、环境描述和项目会话备份，**不包含新训练权重**，也不是环境镜像。若要带走目前最新模型，迁移当天还需同步最新 `tgppo/runs/`，或重新打包最新完整工程。现有快照作为代码/数据基包可继续保留。

## 阶段 5 的真实状态

2026-10-22 检查时，原调度器、监控器和两次训练进程已不在；停止原因未确定。两个 latest.pt 已实际读取，分别为 R0/s11、Rprod/s11，均为 16 个任务、4 轮完整 PPO 更新。其余七次未启动，阶段 5 最终开发评估未落盘。

详见 `observed_status.json`。旧实时页不是当前进程监测；禁止把中断矩阵称为九次实验完成。

## 环境与镜像怎么选

当前系统 Ubuntu 22.04 x86_64，系统 ldd 为 glibc 2.35；训练环境 Python 3.8.20、Torch 1.13.1+cu117、NumPy 1.24.4、定制 PySCIPOpt 1.4.9、SCIP 6.0.1、SoPlex 4.0.1。报表另用 base Python 3.11.7、NumPy 1.26.4、Matplotlib 3.8.0。

1. 默认路线：迁工程，在服务器通过环境描述重建 Python 依赖，并从源码构建相同求解器/定制绑定。当前没有项目已构建且经过验证的 Docker 镜像可直接下载。
2. 同类 Linux 且系统兼容、希望减少联网安装：可以另做 conda-pack 环境包，并在目标目录 conda-unpack。不要直接压缩复制 Conda 目录冒充可迁移环境。此包仍不覆盖工程外的系统库、独立 solver 和 GPU 驱动。
3. 若实验室允许容器且希望长期复现，可另构建包含依赖和定制绑定的 Docker/Apptainer 环境。不是随便下载一个 PyTorch 镜像就能得到本项目定制 SCIP 接口。先确认服务器容器权限、CPU 架构、GPU/驱动与作业系统，再定镜像。

`environment/tgppo_rebuild.yml` 去掉旧 prefix 与普通 PySCIPOpt 安装项；自定义绑定应单独构建。该文件仍为源机导出配方，未在目标机实际安装验证。`environment/conda_explicit_linux64.txt` 是同平台 Conda 包记录，不含全部 pip 包和系统依赖；应与 pip 清单一起使用。

## 接续 Codex

推荐先使用工程根目录 AGENTS.md + CODEX_HANDOFF.md，新会话明确读取它们。官方文档说明 AGENTS.md 可为会话提供项目工作约定：
https://learn.chatgpt.com/docs/agent-configuration/agents-md

原始会话备份位于补充包 `codex_sessions/`。如希望尝试恢复原会话，在目标账户将其中内容合并到 `~/.codex/sessions/`，安装/登录 Codex，再在项目目录运行 `codex resume --all`。不同客户端、版本和索引实现不保证仅复制 JSONL 就能恢复原 UI 会话；如果不能识别，用新会话读交接文件继续即可。原始日志的旧工作路径也需要在恢复时选择新目录。

Codex 会话恢复说明：https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-resume

不需要迁移源机 auth.json、浏览器登录状态、系统凭据或整个用户缓存；新服务器正常登录。会话记录是可选档案，项目接续文件才是跨客户端的主要依据。

## 正式运行前必须补齐的迁移工作

- 路径：`/root/research`、`/root/anaconda3` 在多个入口中写死。新服务器若非相同目录，需要在独立迁移副本整理统一根路径和入口，保持旧日志/哈希不变。
- Git：worktree `.git` 文件和管理元数据含绝对路径。保留所有未提交源文件和补丁，检查工作树关联；不要只复制 repo/.git，也不要丢失 experiments 的未提交修改。
- 绑定：按目标 Python ABI/系统编译，先确认 getCandsState/getMIPState/getNodeState/getIntegralSnapshot/getOpenNodeSnapshot 可用。导入路径需优先使用 stage4/metrics_binding/src 和 stage5/code。
- 二进制：源机 libscip 还依赖系统 libz/readline/gmp/bliss/stdC++ 等；单搬 solver/ 不等于完整运行环境。
- CPU 配额：旧 affinity=[0,1,2,3] 不保证在 Slurm 分配内。按作业允许 CPU 选择，并统一各组并发、线程和内存限制。
- 恢复：当前 load_agent 支持恢复 checkpoint，但 stage5 CLI/调度器尚无完整断点接续入口。先设计恢复游标、有效更新边界、已用预算和失败记录；不要删除旧结果后隐式从头重跑。
- 托管：用实验室 Slurm 或持久的 tmux/服务托管作业，不依赖聊天会话的生命期。定时检查真实进程和心跳。
- 测量：新硬件上重测正式归一化基线。120 秒和 3600 秒不能共用同一基线；历史原生数据仅作源机证据。
- 验证：目标机先跑既有 46 项检查，再做每组三组真实小闭环；检查 Actor-only、原单位 GAE、模型保存恢复和 PDI 统计。
- 正式规模：另存正式协议，明确任务时限、episode/更新、CPU/GPU资源、训练/开发/测试划分和共同预算。不能把源机 2 CPU 核时的先导配置直接当论文全规模配置。

当前未改变模型、训练代码或被冻结的主实验配置；本次只补充迁移与状态材料。

## 参考

- Codex 状态目录及配置：https://learn.chatgpt.com/docs/config-file/config-advanced
- Conda-pack 迁移与限制：https://conda.github.io/conda-pack/
- Docker 镜像包含内容：https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-an-image/
