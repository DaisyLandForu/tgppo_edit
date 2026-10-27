# Transformer 消融结果

正式对照尚未完成；未测量项不能用于推断性能。

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

以实例为统计单位，先平均求解器种子，再按训练种子报告均值/样本标准差。
Solved Instances 为三个求解器种子都在时限内证明完成的不同实例数，Solved runs 为198次任务中的完成数。
超时全部保留；Final Gap 保留 SCIP sentinel，并同时报告未知 gap 数。PDI 为 percent-seconds，保留上下界。
纯网络延迟跨调用汇总，实际访问候选集合可能不同；固定输入微基准另见 benchmark-cpu.json。

