# 训练硬件记录与CPU型号放宽

用户授权不同CPU型号的训练节点。每个CPU/软件指纹使用匹配的原生归一化；最终全部模型在同一评估机器上比较。异构训练的轨迹、吞吐和时间截断会受硬件影响，本轮不宣称严格控制训练CPU的消融。

| 作业 | CPU | GPU |
|---|---|---|
| b0-s11 | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | Tesla V100-SXM2-32GB |
| b0-s22 | Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz | Tesla V100-PCIE-32GB |
| candidate-s11 | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | Tesla V100-SXM2-32GB |
| candidate-s22 | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | Tesla V100-SXM2-32GB |
| score-s11 | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz | Tesla V100-SXM2-32GB |
| score-s22 | Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz | Tesla V100-PCIE-32GB |
