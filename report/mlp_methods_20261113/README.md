# MLP／MLP＋BC：11月13日方法讲解材料

已整理为**12页可编辑PPT＋实际渲染检查的矢量PDF＋详细讲稿**，建议主讲前10页，后2页作为问答附录。

> **新增：参照用户提供的 Mamba-Branching 图风格重绘的训练图**：[高清 PNG](figures/06_mlp_mlpbc_training_mamba_style.png)｜[矢量 SVG](figures/06_mlp_mlpbc_training_mamba_style.svg)｜[矢量 PDF](figures/06_mlp_mlpbc_training_mamba_style.pdf)｜[独立可编辑 PPT](figures/06_mlp_mlpbc_training_mamba_style.pptx)。覆盖 MLP 与 MLP+BC 两条初始化路径、BC／PPO 梯度和实际双路 Actor。已嵌入本周精简稿与详细讲稿；独立单页 PPT 与原12页幻灯片分别保存。

- **直接汇报：[PowerPoint](MLP_MLPBC_方法讲解_20261113.pptx)**。文字、矩阵和模块均为原生可编辑对象，逐页已附演讲者备注。
- **稳妥展示／打印：[PDF](MLP_MLPBC_方法讲解_20261113.pdf)**。已实际渲染逐页查看。
- **离线浏览：[网页幻灯片](MLP_MLPBC_方法讲解_20261113.html)**。左右键翻页，F全屏，无需联网。
- **准备问答：[方法详解与讲稿](方法详解与讲稿.md)**。含每个改动的依据、精确结构、BC数据与训练、结果解释及老师可能追问的问题。
- **快速扫图：[12页缩略预览](PREVIEW_ALL.png)**。

## 六张可直接插入其他汇报的主图

| 内容 | 预览PNG | 矢量SVG | 矢量PDF |
|---|---|---|---|
| 轻量Actor详细结构 | [PNG](figures/01_mlp_actor.png) | [SVG](figures/01_mlp_actor.svg) | [PDF](figures/01_mlp_actor.pdf) |
| 原Actor与MLP结构对照 | [PNG](figures/02_actor_comparison.png) | [SVG](figures/02_actor_comparison.svg) | [PDF](figures/02_actor_comparison.pdf) |
| BC→PPO→CPU部署 | [PNG](figures/03_bc_ppo_pipeline.png) | [SVG](figures/03_bc_ppo_pipeline.svg) | [PDF](figures/03_bc_ppo_pipeline.pdf) |
| 完整66实例主结果 | [PNG](figures/04_full66_results.png) | [SVG](figures/04_full66_results.svg) | [PDF](figures/04_full66_results.pdf) |
| BC的证明率／PDI取舍 | [PNG](figures/05_bc_tradeoff.png) | [SVG](figures/05_bc_tradeoff.svg) | [PDF](figures/05_bc_tradeoff.pdf) |
| MLP／MLP+BC 论文风格训练图 | [PNG](figures/06_mlp_mlpbc_training_mamba_style.png) | [SVG](figures/06_mlp_mlpbc_training_mamba_style.svg) | [PDF](figures/06_mlp_mlpbc_training_mamba_style.pdf) |

图形采用白底、统一颜色语义、模块分区和张量维度；参考Set Transformer、Gasse等论文的表示方式，根据本项目源码重新绘制。SVG文字转矢量路径以避免缺字，PPT保留可编辑文字。字体许可见assets/OFL.txt。

## 今天的核心结论

1. MLP用树上下文条件化共享评分替换整条重Actor：参数4,738,895→24,493；相对新B0平均PDI降低19.35%，证明率提高7.58个百分点。
2. MLP＋BC是相同网络加专家初始化：相对MLP多证明7个实例，但平均PDI增加2.76%，不能说所有指标都改善。
3. Actor决策路径约快20.43倍，不能说端到端求解快20倍；当前学习方法整体仍落后relpscost。

结果限定为固定100 episode、训练seed11、solver seed0、66实例、3600秒、已知最优cutoff受控评测。本轮没有按测试结果挑权重，但66实例有历史测试暴露；尚不能作跨种子稳定收益、完全未见保留集确认或作者论文最终性能结论。

依据与哈希：[SOURCES_AND_EVIDENCE.md](SOURCES_AND_EVIDENCE.md)、[DELIVERY_AUDIT.json](DELIVERY_AUDIT.json)。原始数据副本在data，图和PPT生成脚本在scripts；本次没有训练或新增SCIP。
