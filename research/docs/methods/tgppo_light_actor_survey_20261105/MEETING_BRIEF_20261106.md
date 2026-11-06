# 周五汇报：轻量候选集合 Actor＋专家约束表示＋PPO

方案形成：2026-11-05（北京时间）。**这是方法设计摘要；新方案尚无训练/求解结果，不能将文献效果或旧瓶颈核验计作本方法实验成绩。**

## 方法问题

TGPPO 已使用廉价 25/8/53 维特征，接下来主要减少 Actor 模型计算，同时让候选表示可被专家决策有效约束。轻量化可重选模型族，不必只减少五层 Transformer 的层数。

## 两个优先模型

| 模型 | 计算路径 | 为什么值得验证 | 主要边界 |
|---|---|---|---|
| **首选：树条件化 Deep Sets 式 Actor** | 共享候选 MLP→mean/max 汇聚→各候选结合自身表示、树状态与集合摘要→共享标量评分 | 无自注意力、支持动态候选数和排列等变，原生 CPU 运算容易落地 | 汇聚表达力有限；PDI 和延迟优势未测 |
| **第二候选：单 ISAB Actor** | 候选→少量诱导摘要→逐候选读回→相同评分 | 比单一汇聚有更灵活的候选交互，固定诱导数时对 L 线性 | 两次交叉注意力，小 L 未必胜过小 Transformer |

以共享 MLP/TreeGate 类模型和 1 层 64 维 Transformer 为必要结构参照，避免只与原 474 万参数模型比较。首选草案约 3.27 万推理参数是静态计算，**不是实装结果，也不等于 145 倍加速**。

文献：[TreeGate，AAAI 2021](https://arxiv.org/html/2002.05120v3)、[Deep Sets，NeurIPS 2017](https://arxiv.org/html/1703.06114v3)、[Set Transformer，ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)、[SetNorm / DS++，ICML 2022](https://proceedings.mlr.press/v162/zhang22ac.html)。这些模块已有先例，贡献需由成本—求解质量与独立消融建立。

## 为什么暂不直接换成 Mamba

Mamba-Branching 是“窄嵌入＋专家约束预训练＋历史序列模型”的整体设计。论文 d=8，评估最近 24 步＋当前状态；0.93 ms 来自 GPU 设置。当前候选是集合，历史才是序列，不能据此认定 Mamba 在单线程 CPU 上最合适。[Mamba-Branching，ICLR 2026，附录 E/F/H](https://proceedings.iclr.cc/paper_files/paper/2026/file/b2f56a345fe0ce337df3bdfb31e2b350-Paper-Conference.pdf)

Mamba-2/3、TabM、符号策略均有参考价值，但当前不并行铺开；先证明现有输入下轻量集合模型的质量与成本，再判断是否需要历史或更强表示。

## 独立增量与实验顺序

第一关：合法专家数据→普通 BC；第二关：同标签/预算比较 CL 是否比 BC 额外有效；第三关：固定结构、原 H3/Critic/PPO 做真实求解对照。

|  | 原 B0 Actor | 选定轻量 Actor |
|---|---|---|
| 无预训练 | PPO | PPO |
| 有专家预训练 | BC→PPO | BC→PPO |

这四组分别检验结构与预训练增量；CL 在此之后单独消融。专家一致率是中间指标，最终报告配对 PDI、证明率、墙时及 Actor 路径延迟。两种训练种子只作初步重复，不能宣称稳定性已充分验证。

## 最省成本的下一步

建议先做现有轨迹兼容性核查、本地 CPU 微基准与小规模 BC 筛选，单线程累计计算提案上限 10 分钟；本轮未执行。Mamba 现有训练清单与 TGPPO train 有 15 实例交集，另外 4 个属于 TGPPO dev，不能混入拟合。版本、cutoff、特征缩放和动作映射仍须核对。

通过离线关口后只让一个结构进入有预算的 PPO，不为分钟级小试申请 Custom。CR 保持终止；树信用分配等待 PDI 适配和未完成子树处理。

完整依据和停止条件见 [调研报告](REPORT.md)；来源与阅读程度见 [文献台账](LITERATURE.csv)。本阶段为调研交付，等待验收。
