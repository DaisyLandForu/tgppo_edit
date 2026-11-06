# 六作业服务器训练与评估启动包

2026-10-22。用户本轮选择：**B0/R0/Rjoint × 种子 11/22，六个独立作业，各 500 个完整 episode，每 100 保存，按开发集选最好权重。** 不再把之前建议的四组×三种子当作本轮必跑范围。

本包已补齐代码与 Slurm 模板，本机 CPU 闭环和恢复测试通过；**没有在实验室 V100/Slurm 上验证，也没有提交六个正式训练作业**。GPU 更新与断点 RNG 恢复必须通过目标机桥接后才能放行。

## 目录与本轮协议

- 新研究副本：`tgppo/experiments/server-20261022/`。原版和 stage2—5 全部保留。
- 新入口：`code/server.py`；GPU 桥接：`code/server_bridge.py`。
- 本目录 `protocol.json`：三组、两种子、500 episode、里程碑 100/200/300/400/500。
- 训练：4 名单线程 CPU SCIP worker 采集同一个策略的四个完整任务，然后 GPU 做完整 PPO 更新，3 epochs、逻辑 batch 256、微批 16、FP32。Actor/Critic 分开优化，新增输入仅用于 Critic，value/GAE 保持原单位。
- 仍为 120 秒任务、20/5 划分、H3、MSE；公共优化器沿用修复主线 Adam。**B0 为统一短时协议下的原方法参照，不能称为原论文 AdamW/3600 秒全规模复现。**
- 每训练作业申请 8 CPU/32G/1 GPU，六路并发共 48 CPU/192G/6 GPU。实际 Slurm 账户、分区、V100 GRES 类型、CPU 约束需按实验室设置填写。
- 原 50 episode/2 CPU 核时的入口保持原样；新目录按本轮 500 episode 配置执行，Slurm 模板最长 24 小时。24 小时是作业模板上限，不是已经测得的训练时间或“一夜必跑完”的保证。
- 两个种子是初步重复证据，均应报告，不从两个种子中只挑成绩最好的一份。

数组映射：0=B0/11，1=B0/22，2=R0/11，3=R0/22，4=Rjoint/11，5=Rjoint/22。

## 迁移时必须补带本轮文件

之前生成的工程包/迁移补充包不自动包含这些新文件。请重新同步本目录和 `tgppo/experiments/server-20261022/`，以及依赖的数据、求解器、stage4 定制绑定源码与校准统计。目录中的 `.git` worktree 指针可能需要修复，运行 Python 不依赖该指针，但不要丢掉未提交源文件。

本包使用相对项目根路径，**不要全局替换历史日志路径**。`env.sh` 自动推断项目根，也可设置 `RESEARCH_ROOT`。它优先导入 stage4 定制绑定，再导入本次服务器代码，不走原版启动脚本。

## 第一步：服务器环境与检查

在服务器先重建匹配的 Python/SCIP/定制绑定，具体见上一级迁移说明。至少应有五个接口：getCandsState/getMIPState/getNodeState/getIntegralSnapshot/getOpenNodeSnapshot。源机 `.so` 不一定可跨 ABI 直接使用。

```bash
cd /你的路径/research
export TGPPO_PYTHON=/你的Conda环境/bin/python
export SERVER_LAUNCH_DIR="$PWD/migration/2026-10-22/server-run"
source "$SERVER_LAUNCH_DIR/env.sh"
"$TGPPO_PYTHON" -m unittest discover -s "$RESEARCH_ROOT/tgppo/experiments/server-20261022/tests" -v
```

这些是小型工程检查，包含真实短任务，不会启动 500 episode；其中已有离线缓存测试可能使用 GPU 纯张量拟合。完整 GPU 在线桥接由下一步单独执行。根据显存和内存峰值决定是否需要共同修订微批或资源申请，不采用候选截断。

## 第二步：归一化基线与目标 V100 桥接

先按实验室实际填写分区/账户。基线、训练和选模应放在同类 CPU 和同套软件环境，必要时使用一致 CPU constraint。代码校验 CPU 型号、求解器/绑定哈希、Torch/ABI 和协议；不拿旧机器的 120 秒基线冒充新服务器归一化基线。

```bash
export GPU_PARTITION=your_v100_partition
export CPU_PARTITION=your_cpu_partition
export LAB_ACCOUNT=your_account

"$SERVER_LAUNCH_DIR/submit.sh" baselines --partition="$CPU_PARTITION" --account="$LAB_ACCOUNT"
"$SERVER_LAUNCH_DIR/submit.sh" bridge --partition="$GPU_PARTITION" --account="$LAB_ACCOUNT"
```

若集群需显式 GPU 类型，可为 GPU 作业加 `--gres=gpu:v100:1`（类型名按本集群实际设置）。模板默认 `gpu:1`，桥接会核对实际设备名包含 V100。没有 Slurm 的环境也可以在已经分配好的目标 GPU 节点直接运行 `gpu_bridge.sh`；不要在登录节点运行计算。

`baselines` 在新运行目录产生 25×3=75 条原生任务。`bridge` 用完整 256/5/8 模型，对 B0/R0/Rjoint 分别做四条真实完整任务及一次 CUDA PPO 更新，核对 CPU rollout/GPU 重算概率、完整 Critic 更新、优化器/CUDA RNG 恢复、CPU worker 不初始化 CUDA、Actor-only CPU 加载。桥接只产生工程证据，不代替正式训练。

目标成功证据：`tgppo/runs/server6-20261022/gpu-bridge/verification.json` 的 passed 为 true；同时正式 `normalization/baselines.json` 已产生。未满足时，正式训练会拒绝开始。

请根据目标试跑补做覆盖 20 个训练实例的吞吐估算；目前“一夜完成500”的时间窗口仍属估算。GPU 桥接首批小实例用于正确性，不是整场成本预测。

## 第三步：六个正式训练作业

完成环境、基线、桥接和资源核对后提交。这里仅提供命令，当前没有执行：

```bash
"$SERVER_LAUNCH_DIR/submit.sh" train --partition="$GPU_PARTITION" --account="$LAB_ACCOUNT"
```

也可以在提交时传 `--dependency=afterok:基线作业ID:桥接作业ID`。依赖和提前结束信号按 [Slurm sbatch 官方文档](https://slurm.schedmd.com/sbatch.html) 配置；具体队列策略需要目标集群验证。

每个运行目录如 `training/Rjoint-s11/`：

- `latest.pt`：最近原子恢复状态，包含 Actor/Critic、两套优化器、CPU/CUDA RNG、完整轨迹、游标、策略版本；可能是“已采集、尚未完整更新”的状态。
- `episode-0100.pt` … `episode-0500.pt`：五个稳定里程碑；记录真实已训练任务数。某批决策不足 256 时会保留缓存，不伪造有效更新。
- `updates.json`：实际转移数、逻辑优化步、CPU/GPU event/墙钟、显存峰值。
- `attempts/` 与 `collection/`：每次启动和所有采样尝试，包括失败及重试记录。
- `summary.json`：仅在完成 500 个任务后产生。没有它时不能把残留 running/日志当作训练完成。

如果作业被限时或中断，重新提交对应数组编号，例如 `submit.sh train --array=1,4 ...`。脚本使用显式 `--resume`，不会删除旧状态。半轮 PPO 的参数不提交；恢复从更新前的完整轨迹和 RNG 重做这一轮，已完成的有效采样可复用，失败尝试仍保留。

`--requeue` 不是无条件自动续跑承诺，具体重排取决于 Slurm 配置。若旧依赖因失败失效，需要为后续选模作业重新设置依赖。SIGKILL 可能使最后一段心跳成本未落盘，费用统计还需结合 sacct/集群记账，不把心跳当作精确总核时。

## 第四步：开发集选最好权重

训练完成后用 CPU-only 数组评估五个里程碑，不占 GPU。六个作业每个都比较完全相同的 5 个开发实例×3 个求解器种子，共 6×5×5×3=450 次开发求解。除训练外要预留这部分计算。

```bash
"$SERVER_LAUNCH_DIR/submit.sh" select --partition="$CPU_PARTITION" --account="$LAB_ACCOUNT" --dependency=afterok:训练数组作业ID
```

固定规则：每个“组×训练种子”分别选开发集实例等权平均 120 秒 PDI 最低的里程碑，精确相同时取较早者。任何失败或覆盖不全都不能静默删去。开发集只能用于选择，不能再称为最终独立测试。实际比较保留测量区间，小差值不等于稳定提升。

输出：

- `selection.json`：五个候选分数、选择原因、数据集/种子、检查点哈希。
- `actor_best.pt`：该训练种子的开发集最佳 Actor，纯 Actor 权重，没有 Critic 或优化器。
- `actor_final.pt`：同一运行第 500 episode 的 Actor，作为统一终点对照保留。

六份 `actor_best.pt` 都应进入最终比较，不在两个训练种子中继续挑赢家。主结果若采用开发集最佳，表述为“同一选模规则后的未见测试结果”；同时报告固定 500 的结果更便于判断选模影响。

## 第五步：Actor 接入 SCIP 和未见测试集

接入路径已经存在：`load_actor → ActorPolicy → Environment → Brancher.branchexeclp → SCIP.branchVar`。训练时就在 SCIP 中采样；训练后换成冻结 Actor-only 路径即可，无需重新实现分支规则。正式 CPU 评估不构建 Critic、不取联合前沿、不更新参数。

本包没有自动启动测试或阶段 6。选模规则和权重锁定后，才提交：

```bash
"$SERVER_LAUNCH_DIR/submit.sh" evaluate --partition="$CPU_PARTITION" --account="$LAB_ACCOUNT"
```

默认使用 `test66.json` 的 66 个保留实例、每个三个求解器种子。六份 Actor 合计 1188 次求解。`CHECKPOINT_KIND=final` 可另跑固定 500 终点；其余条件保持一致。测试检查名称和文件哈希，防止重命名的训练/开发实例混入测试。

多个评测集可分别设置 `TEST_INSTANCES_DIR`、`TEST_NAMES_JSON`（实例名列表）、`TEST_LABEL`。当前入口要求 `.mps.gz` 且有 `cutoffs.json` 中已验证的最优 cutoff，沿用受控 sandbox 任务；任意新增数据集/无 cutoff/默认完整 SCIP 设置需要另配协议和原生对照，不能冒称即插即用的通用评测。

同一评测集还应跑 relpscost 原生参照（CPU 作业内执行）：

```bash
"$SERVER_LAUNCH_DIR/run.sh" evaluate --native \
  --instances-dir "$RESEARCH_ROOT/tgppo/datasets/test_instances" \
  --names-json "$SERVER_LAUNCH_DIR/test66.json" --label official66-native
```

输出每次原始 PDI/区间、证明状态、节点、时间和回调开销。各组使用相同时限、求解器种子、动作随机流配对规则、硬件类别及求解器设置。不要在测试集上重新挑检查点或只报告有利实例。两个训练种子属于初步证据；没有 Rprod，本轮仍不能把整套收益都归因于联合关联结构。

## 交付与验收边界

本次交付是服务器代码、启动模板和本机工程验证。目标 V100/ABI、真实 Slurm 提交/重排、完整 500 episode 耗时尚待服务器验证。旧阶段 5 中断矩阵没有恢复完成，历史记录未改写为成功；新六作业有独立输出目录，完成后按阶段约定交付并等待验收，不自动进入阶段 6。


本机交付证据：[50项最终检查](./tests-final.log)、[检查汇总](./verification.json)、[CPU入口检查](./local_cpu_preflight.json)、[相对stage5增量补丁](./server_from_stage5.patch)、[相对原版完整补丁](./server_complete.patch)。目标GPU/Slurm状态始终单列为待验证。
