# TGPPO Transformer 最小消融

本目录是 `server-20261022` 已验证 B0 的独立副本。原版入口、历史 Critic 改进、权重和日志均未修改。当前用户要求训练交由集群 Custom 作业；本地没有启动任何真实 MILP 训练。

- `transformer`：Transformer + 原 BiMatching。
- `bimatching`：Actor 和 Critic 都移除 Transformer，候选编码直接进入原 BiMatching。
- 默认 `use_transformer=true`，兼容旧 B0 checkpoint；false 必须全新独立训练。
- 当前证据、形状及限制见 [报告](docs/REPORT.md)、[改动前检查](docs/INSPECTION.md)、[正式结果表](docs/RESULTS.md)。

## Custom 作业配置与命令

共享挂载 `/home/duweiyue25`，镜像沿用实验室 `scip (user-duweiyue25/envd:11260740-c4da)`，Shell=bash，工作目录 `/home/duweiyue25/tgppo-transfer/research`。脚本使用共享 Python `/home/duweiyue25/conda/envs/tgppo/bin/python`。换路径时通过 `RESEARCH_ROOT`、`TGPPO_PYTHON` 指定，不修改历史文件。

**每行命令对应一个 Custom 作业。不要在开发容器执行 smoke、prepare、train。** 当前容器没有 GPU，也没有可用的 Crater 作业提交接口；下面的作业尚未提交。

| 阶段 | 作业数量 | 每作业资源 | 依赖 |
|---|---:|---|---|
| prepare（含短程 smoke、两组真实 CUDA 更新/中断恢复、成本试跑） | 1 | 8 CPU / 32 GiB / 1 V100 | 无 |
| train | 6，可并行 | 8 CPU / 32 GiB / 1 V100 | prepare 全部通过 |
| select | 6，可并行 | 4 CPU / 32 GiB / 无 GPU | 各自训练完成 |
| freeze | 本地只读核查后落盘 | 无 GPU | 六份选模全部完成 |
| evaluate | 6，可并行 | 4 CPU / 32 GiB / 无 GPU | freeze 完成 |
| report | 本地 | 无 GPU | 测试结果到齐，未到齐则报告缺失 |

训练、归一化、开发与测试须使用**相同 CPU 型号**（例如此前 Gold 6148）、相同 SCIP/绑定/Torch；配置在目标节点测量指纹，跨型号直接拒绝。准备及训练申请 V100；脚本拒绝无 GPU 的正式训练，不自动降级为 CPU。可申请平台最长时限（历史模板24h）；旧 B0 每500 episode约7小时仅为参考，新两组真实耗时由 prepare 的20实例 pilot估算。测试需32GiB是为保留历史大实例的求解空间，不因旧8GiB worker限制而跳过实例。

先申请一个准备作业：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh prepare
```

此命令先确认目标GPU/配额，再自动做两组短程训练：每组8个完整episode，四个训练实例轮换，5秒时限，两个完整PPO更新，2个开发实例、两个里程碑的Actor-only在线评估。架构256/5/8、Adam、batch256、epochs3等保持原值，短程不代表效果。之后同硬件重测75次120秒原生归一化任务（两组共用准确匹配的原生记录），各组独立验证CUDA更新/中断恢复，并各跑20实例成本pilot。数据/模型没有使用历史B0权重热启动。

如只想先检查短程训练，可先申请一个同资源作业：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh smoke
```

prepare 会验证已有 smoke 的代码/协议/软硬件指纹并复用通过证据，不再次训练它。

准备通过后，分别创建六个训练 Custom 作业，前三个和后三个仅架构开关不同：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train transformer 33
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh train bimatching 33
```

每组/种子500个完整episode、相同20训练实例随机日程、每100保存、4个CPU采样worker、CUDA FP32 PPO。两组保留公共层逐位相同初始化及后续RNG流，但轨迹会随策略分化。固定episode预算，记录实际转移数和有效更新数，不伪称两组获得完全相同轨迹或优化步数。

中断后确认旧作业退出，再用**同一条训练命令**创建续跑作业。它显式恢复 latest.pt、优化器、RNG和完整轨迹，不删除旧结果；文件锁拒绝双启动。不编辑正在运行的代码/配置：内容哈希改变会拒绝续跑。只有匹配的summary.json才算训练完成；不能用running标志代替。

训练完成后，分别创建六个CPU选模作业：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select transformer 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select transformer 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select transformer 33
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select bimatching 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select bimatching 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh select bimatching 33
```

每种子同样五里程碑 × 五开发实例 × 三求解器种子，按实例等权120秒PDI最小选模，精确相同取较早者；保留best和final。开发求解共450次，不能从三个训练种子中挑一个赢家。

六份完成后可在开发容器执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh freeze
```

冻结六份选模及Actor哈希后，再分别创建六个CPU测试作业（同样CPU型号）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate transformer 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate transformer 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate transformer 33
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate bimatching 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate bimatching 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh evaluate bimatching 33
```

CPU Actor-only随机采样，与现有评估方式一致；相同动作随机流配对，3个求解器种子，66个冻结测试实例，120秒，共1188次求解。测试集在历史Critic实验中已使用，不能称为从未看过的新数据。所有超时、原始Gap、PDI上下界和失败尝试保留；无效任务导致失败，不能静默删去。重启命令只复用身份完全匹配的有效结果，并保留失败成本。

沿用历史已验证的共同资源例外：`mspp16` 在普通65实例结束后单worker串行运行，地址空间24GiB、容器至少32GiB、进程watchdog300秒；SCIP搜索仍120秒。两组规则完全相同，其余实例与训练保持8GiB worker上限。

## 本地检查与汇总

以下命令不启动MILP训练：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh tests
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh report
```

`report` 输出 docs/RESULTS.md、comparison.json、per_instance.csv，包含种子均值/标准差、以实例为单位的配对bootstrap、超时、未知Gap计数、调用延迟、训练峰值显存和有限loss检查。未跑完时写缺失清单与“未测量”，不会把随机策略的在线探针填入正式表。

固定输入CPU前向微基准（新测量会覆盖同名微基准文件，正式训练结果不受影响）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/custom_job.sh benchmark --device cpu --repeats 30
```

在已分配GPU节点上可改为 `--device cuda`。使用10次warmup、两端CUDA同步、交替模型顺序。该命令只测同一组合成输入，不训练、不评价策略；不能替代端到端求解结果。

曲线脚本隔离报告依赖，不修改训练Conda：

```bash
conda env create -f /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/environment/reporting_environment.yml
conda run -n tgppo-reporting python /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/transformer-ablation-20261027/plot_results.py
```

有真实训练日志后生成Actor/Critic loss、entropy、clip fraction、episode reward和决策数的PNG/PDF/SVG；没有训练时明确退出，不生成虚构曲线。当前报告环境未安装，曲线渲染未执行。

## 文件及输出位置

| 位置 | 内容 |
|---|---|
| configs/transformer.json、bimatching.json | 两组正式冻结配置，唯一方法变量use_transformer |
| configs/*-smoke.json | 单独的短程工程验证，不能混入正式对照 |
| code/project/policy/actor.py、critic.py | 开关、原前向旁路，保留原BiMatching和头 |
| code/project/runtime.py | 配置透传、旧checkpoint默认true |
| code/server.py | 复用原训练，仅透传开关与动态时限标签 |
| code/project/agent/agent.py、brancher/brancher.py、environment/environment.py | 两组共用的只读计时、Fair Nodes和墙钟采集 |
| code/ablation.py | 复用server的准备、训练、选模、评估；配对协议及汇总 |
| tests/test_ablation*.py | 张量、mask、梯度、旧权重、配对协议及超时汇总检查 |
| docs/source_inventory.json、verification.json | 受保护源码哈希、验证索引 |
| tgppo/runs/transformer-ablation-20261027/{variant}/training/B0-s{seed}/ | 各自config、latest、里程碑、updates、采样和选模日志 |
| tgppo/runs/transformer-ablation-20261027/{variant}/test/official66/ | 各自逐实例测试记录与失败尝试 |
| tgppo/runs/transformer-ablation-20261027/launcher-logs/ | 每次Custom启动日志 |
| tgppo/runs/transformer-ablation-20261027-smoke/ | 短程真实训练验证；当前尚未运行 |

以上runs路径均相对于research。当前全部改动位于本独立Git实验目录；初始提交 `c56cfd8` 保存未消融的服务器副本，后续差异可用 `git diff c56cfd8` 审查。回退时切换回已有server原入口或另用初始提交的工作副本，不需要删除任何历史实验。
