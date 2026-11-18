# TGPPO：200 episode评测摘要（2026-11-16）

330/330项有效，66实例、solver seed0、每项3600秒。四个CPU Custom各9.68—11.82小时，11月14日01:20全部完成。本次只复核现有结果，没有新增训练或求解。

| 条件 | 平均PDI↓ | 证明完成↑ | Actor决策路径均值↓ |
|---|---:|---:|---:|
| 新B0 seed22 | 33878.76 | 25/66（37.88%） | 55.79ms |
| MLP seed11 | 29459.59 | 33/66（50.00%） | 3.17ms |
| MLP＋BC seed11 | 26222.10 | 41/66（62.12%） | 3.08ms |
| MLP＋BC seed22 | 26469.15 | 40/66（60.61%） | 3.06ms |
| relpscost | 14345.32 | 60/66（90.91%） | — |

**可报告的两个同训练种子对照：**

- seed11的BC增量：PDI−10.99%，证明净增8例（新增9、丢失1），平均optimize墙时−13.50%；44例PDI改善、22例退步。
- seed22的完整组合相对新B0：PDI−21.87%，证明净增15例且未丢失原已证明实例，平均optimize墙时−17.82%。Actor决策路径观测约快18.24倍，不是纯forward或端到端倍数。

组合的全66节点SGM增加10.95%，不能说所有指标都提升。作者固定easy33节点SGM／hard32 PDI SGM上，组合分别改善5.40%／34.19%。

结论：本轮支持“轻量Actor＋BC初始化＋PPO”相对本地复现新B0的正向改进，仍未超过relpscost；缺新B0-seed11及MLP-seed22的200，尚不能识别纯结构的同种子增量或确认BC增量跨训练种子稳定。

探索性实例bootstrap：BC平均PDI变化区间含0，均值稳健性仍不足；组合区间未跨0。它不替代新增solver或训练种子。100与200评测硬件不同，不能凭绝对数值推断越训越好／越差。

建议先冻结200，预先固定8例补solver seeds1、2做80项CPU配对确认；之后补齐两份200消融，再决定是否扩至500。该后续仅为提案，本轮交付后等待安排。

完整来源：[结果复核](../research/docs/methods/tgppo_paper200_test66_20261113/RESULTS_REVIEW.md)、[原始主表](../research/docs/methods/tgppo_paper200_test66_20261113/summary.csv)、[只读审计](../research/docs/methods/tgppo_paper200_test66_20261113/review/REVIEW.json)。
