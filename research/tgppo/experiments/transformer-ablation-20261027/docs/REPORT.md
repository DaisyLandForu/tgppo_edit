# TGPPO Transformer 消融实验报告

日期：2026-10-27（北京时间）。**实现与本地正确性验证已完成；按用户最新要求，真实训练只交付集群 Custom 脚本，没有在开发容器执行。尚无训练后的两组性能对照，当前不能决定 Transformer 是否值得保留。**

## 1. 检查与基线选择

以 `tgppo/experiments/server-20261022` 的已验证 B0 为基线，独立复制到本目录。原版 `tgppo/repo`、stage2—5、历史 R0/Rjoint、全部权重和上传快照保持原状。三份网络文件 Actor/Critic/modules 与原版逐字节一致。原有 B0-s11/s22 已各完成500 episode、125/123次有效PPO更新；R0/Rjoint 属于之前的 Critic 研究，不进入本次对照。早期中断 stage5 和后来完整 server6 是不同实验，不能混淆。原B0测试数据不能与新硬件、新初始化的消融直接拼成对照。

完整检查及源码位置见 [INSPECTION.md](INSPECTION.md)，改动前哈希和现存训练摘要见 [source_inventory.json](source_inventory.json)。当前14项检查通过，已验证所有受保护源码哈希未变；见 [verification.json](verification.json)。

实际在线调用为 `runtime._worker → load_actor → ActorPolicy → Environment → SCIP.optimize → Brancher.branchexeclp → getCandsState/getMIPState/getNodeState → Actor.forward → Categorical.sample → branchVar(cands[action])`。保留 SCIP 候选对象与特征的同序索引，cands_pos 不用作策略action。

## 2. 最小架构变化

| 路径 | Transformer输入/输出 | 进入BiMatching | 输出 |
|---|---|---|---|
| Actor | `[L,B,256] → [L,B,256]` | tree `[B,256]`，candidates `[B,L,256]`，mask `[B,L]` | logits/probs `[B,L]` |
| Critic | `[L,B,256] → [L,B,256]` | 独立tree/candidates，相同形状 | value `[B,1]` |

输入候选特征25维，node8维，mip53维。Actor 的 LayerNorm+候选线性编码、树编码，以及候选/树拼接后的 global_embedding 全部保留；false 旁路的是 Transformer。Critic 没有 Actor 的 global_embedding，保留其原编码、BiMatching、masked mean、aggregation 和 Tree-Gated value_head。

Transformer为5层、hidden256、8头、FFN1024、GELU、post-norm、两处残差、dropout .05，无位置编码、无额外末层norm。false 删除整个encoder，包括内部norm；输入输出同维，没有新增适配器、额外归一化或交互模块。BiMatching六个原有线性层、两路候选轴softmax聚合与sigmoid融合逐字保留。

一个 `use_transformer` 同时控制 Actor/Critic；因此结论针对整套架构，无法单独归因到 Actor 或 Critic，也不能区分候选交互与容量、encoder内部norm的影响。为严格配对初始化，false 构造时消耗原始初始化RNG后释放encoder，state_dict/优化器中没有Transformer参数，前向绝不执行它。默认true的权重与原模型输出逐位相同，旧checkpoint缺开关时按true加载，不改写其配置哈希。

## 3. 公平训练及评估协议

正式两组各训练种子11/22/33，从头初始化，不复用旧B0权重。配置 `configs/transformer.json` 与 `configs/bimatching.json` 只有use_transformer、variant标签、输出目录不同，自动检查拒绝其他变量差异。

共用20 train / 5 dev / 66个既有test；原特征与H3；Adam，lr Actor=.00024/Critic=.00012，gamma=.97，lambda=.92，clip=.16，entropy=.003，batch256，epochs3，gradient clip=.5；每四个完整episode更新，微批16，不截断候选。Actor rollout和PPO重算均eval（不关梯度），Critic rollout eval、更新train。B0 Critic原单位MSE，无历史搜索增强或校准模块。

每运行500完整episode，100/200/300/400/500保存，同一训练日程/预算。轨迹分化后转移数与有效更新数可能不同，应如实记录。各训练种子在同一5 dev×3 solver seeds上独立选择最低实例等权120秒PDI里程碑，平分取较早者，不能从三个训练种子中再挑赢家。六份Actor与选模哈希全部冻结后才测试66×3任务/模型，共1188次。66实例在历史研究中已经使用，本轮称冻结既有测试集，不称新留出数据。

SCIP6.0.1/SoPlex4.0.1/定制PySCIPOpt1.4.9、sandbox+最优cutoff、120秒、CPU Actor推理、CUDA FP32 PPO。各阶段强制相同CPU型号及求解器/绑定/Torch哈希，并做新的原生归一化、CUDA恢复桥接和20实例pilot。当前容器为Silver4214、4.01 CPU配额、约32GiB、无GPU，不复用旧Gold节点归一化。现有B0仍是120秒/Adam参照，不是论文3600秒/AdamW规模复现。

已核对历史mspp16内存故障：两组共同保留相同资源例外，仅此实例改为单worker、24GiB地址空间、容器至少32GiB、watchdog300秒；SCIP搜索仍120秒，其余65实例/训练维持8GiB。不因为资源错误剔除该实例。内存例外只影响资源容纳，不引入策略变化，单元测试覆盖任务分流及保持时限。

## 4. 实际完成的验证

[14项检查日志](unit-tests.log)覆盖默认true逐位等价、公共初始化/RNG配对、参数确实移除、生产256维前向和logits概率、可变候选数、padding无影响、单候选、Actor/Critic有限非零独立梯度、置换等变/价值不变及索引对应、空集合/全padding/NaN拒绝、两组checkpoint读写、旧B0加载、协议控制变量、冻结防篡改、超时/未知gap保留、mspp16资源处理。这些是单元测试，不是MILP训练。

[在线验证](online-check/verification.json)实测两组随机初始化Actor及旧已训练B0，在lseu上各做2秒在线求解；三次均合法结束，有实际分支，fallback=0，Critic耗时=0。证明接口可运行，**不比较随机策略节点、Gap或求解质量**。第一次新Fair Nodes采集因旧绑定要求bytes而失败，修为bytes规则名后通过；首次失败日志和原始结果均保留，未混入正式结果。

[CPU环境](preflight-cpu.json)与[固定输入微基准](benchmark-cpu.json)已实测。真实短程训练、CUDA梯度/中断恢复及500-episode对照均未运行，需用户创建Custom作业；GPU显存与延迟未测量。完整训练曲线只有真实日志到齐后才能生成。

## 5. 正式性能结果（当前未测量）

| Metric | Transformer + BiMatching | BiMatching Only | Relative Change |
|---|---:|---:|---:|
| Parameters (Actor) | 4.7389e+06 | 790095 | -83.33% |
| Parameters (Actor+Critic) | 9.54358e+06 | 1.64598e+06 | -82.75% |
| Inference Latency (ms) | 未测量 | 未测量 | 未测量 |
| Inference P50 (ms) | 未测量 | 未测量 | 未测量 |
| Inference P95 (ms) | 未测量 | 未测量 | 未测量 |
| Solve Time (s) | 未测量 | 未测量 | 未测量 |
| Worker wall incl. checkpoint load (s) | 未测量 | 未测量 | 未测量 |
| Nodes | 未测量 | 未测量 | 未测量 |
| Fair Nodes | 未测量 | 未测量 | 未测量 |
| Primal-Dual Integral | 未测量 | 未测量 | 未测量 |
| Final Gap | 未测量 | 未测量 | 未测量 |
| Solved Instances (all solver seeds; /66) | 未测量 | 未测量 | 未测量 |
| Solved runs (/198) | 未测量 | 未测量 | 未测量 |
| Timeout runs | 未测量 | 未测量 | 未测量 |
| Feature extraction per instance (s) | 未测量 | 未测量 | 未测量 |
| Branch callback per instance (s) | 未测量 | 未测量 | 未测量 |


参数来自真实256维模型。正式表的延迟/求解指标保持未测量，不把下面的合成输入微基准冒充训练后部署结果。训练稳定性及不同种子波动同样待测。

## 6. 已实测的固定输入 CPU 前向开销

同一个Silver4214开发容器、Torch1.13.1、单Torch线程，batch=1，固定随机输入，10次warmup，30次重复，交替测试顺序；两组从同种子初始化，纯Actor forward，不含特征提取/张量转换/采样。该工作量没有训练或求解质量指标。共享容器的调度噪声未作跨时段重复，不能外推V100或Gold延迟。

| Candidates L | Transformer mean ms | BM Only mean ms | Relative Change | Transformer P50 / P95 ms | BM P50 / P95 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.485 | 1.451 | -73.55% | 5.442 / 5.803 | 1.477 / 1.541 |
| 8 | 8.043 | 1.871 | -76.74% | 8.001 / 8.331 | 1.832 / 2.116 |
| 32 | 15.135 | 2.450 | -83.81% | 15.008 / 16.090 | 2.456 / 2.505 |
| 128 | 38.880 | 4.494 | -88.44% | 38.425 / 41.916 | 4.398 / 5.161 |
| 512 | 215.674 | 12.494 | -94.21% | 214.878 / 218.212 | 12.489 / 12.942 |

Actor参数从4,738,895降至790,095（-83.33%）；Actor+Critic从9,543,582降至1,645,982（-82.75%）。这些数据说明删除encoder能减少固定输入前向计算。它们不能证明完整MILP求解更快，因为策略会改变搜索树、候选数量和超时行为。真实在线forward、原Actor调用、特征提取、branch和整个callback分别计时，原始样本提供P50/P95；CUDA计时前后同步。

MILP Solve Time口径为实例加载+optimize（含回调）+finalize+释放；不含模型checkpoint加载/构建，后者连同求解另以worker_wall_seconds报告。CPU在线评估没有GPU显存；训练峰值allocated/reserved由每次真实CUDA更新保存。

## 7. 如何根据结果作判断

1. **性能贡献：目前未知。** 必须完成独立训练后比较节点、Fair Nodes、固定时域PDI及上下界、最终Gap、证明完成、超时和训练曲线。时间截断下更少节点也可能只是吞吐较低，不能单独当作质量收益。
2. **计算收益：固定输入CPU微基准已证实开销降低；实际求解收益未知。** 用同一批测试实例、相同时间限制与硬件，保留超时；按实例配对，先平均solver seeds，再报告训练种子均值/样本标准差和按实例重采样的区间。未知Gap保留SCIP infinity sentinel并单列数量，不能当作0或丢弃。
3. **研究价值：BiMatching已经包含集合聚合，从实现上具备独立工作接口；是否足够有效仍需训练数据回答。** 若三种子下质量接近且求解时间降低，才支持继续轻量交互研究；若质量明显下降，需结合容量/norm减少分析。当前不推荐仅凭本地微基准永久删除Transformer，也不引入新架构来补偿。

## 8. 复现与交付索引

- [README中的完整Custom命令、资源、恢复和输出目录](../README.md)。首个作业为prepare（包含短程smoke），随后6训练、6开发选模、freeze、6测试、report。
- [自动正式表](RESULTS.md)、[缺失模型和逐种子结果JSON](comparison.json)、[本地检查](unit-tests.log)、[完整改动前检查](INSPECTION.md)。正式结果目前0/6模型。
- `code/ablation.py report` 在数据到齐后生成真实逐实例CSV、均值/波动、超时/Gap、调用延迟、显存、配对实例bootstrap；不完整时标明缺失。
- `plot_results.py` 使用独立reporting环境生成loss/reward等PNG/PDF/SVG；当前无训练数据且报告依赖未安装，没有虚构图。
- 修改限于本独立Git目录：actor.py/critic.py开关，runtime.py透传/旧权重兼容，server.py透传及共同资源处理，agent/brancher/environment只读计时与指标，新增配置、薄封装脚本、测试和文档。BiMatching、TreeGate、PPO.learn、memory、reward、特征绑定均未修改。

本轮可以交付可运行实现及检查证据、可直接提交的训练/评估命令。**训练、选模、完整测试及“是否值得保留”的性能结论尚未完成，等待用户集群作业的实际产物。**
