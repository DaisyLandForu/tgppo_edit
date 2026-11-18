# 文献、源码和实验依据

## 阅读与图形参考

| 来源 | 本次使用范围 | 链接 |
|---|---|---|
| TGPPO，AAAI 2026／arXiv 2511.12986 | §4.1原Actor、§4.3/5.3训练；Fig.1闭环布局参考 | [论文](https://arxiv.org/html/2511.12986v1) |
| Zarpellon et al.，AAAI 2021 | 搜索树条件化已有依据；廉价特征不算本次新增 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/16512) |
| Zaheer et al.，Deep Sets，NeurIPS 2017 | 参数共享、排列结构依据；本MLP不含集合汇聚 | [论文](https://arxiv.org/abs/1703.06114) |
| Lee et al.，Set Transformer，ICML 2019 | 实际查看Fig.1模块分图、维度标注；未声称当前使用ISAB | [论文与图](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf) |
| Gasse et al.，NeurIPS 2019 | 专家动作交叉熵先例；实际查看Fig.2图形表达 | [论文与图](https://proceedings.neurips.cc/paper/2019/file/d14c2267d848abeb81fd590f371d39bd-Paper.pdf) |

所有主图为本次原创重绘，只借鉴组织和表达方式，没有把引用模型的结构当成实际实现。Gasse使用strong branching教师，本项目复用relpscost标签，必须区分。TGPPO架构事实与实现维度优先按下列本地源码核对。

## 精确实现

- [LightActor](../../research/tgppo/experiments/paper-aligned-20261106/code/project/policy/light_actor.py)：kind=mlp的双路编码、128→64→32→1头、mask/softmax。
- [原Actor](../../research/tgppo/experiments/paper-aligned-20261106/code/project/policy/actor.py)与[模块](../../research/tgppo/experiments/paper-aligned-20261106/code/project/policy/modules.py)：Transformer、BiMatching及TreeGate压缩链。
- [BC实现](../../research/tgppo/experiments/paper-aligned-20261106/code/pretrain.py)：实例均衡、Adam、交叉熵、选模里程碑、独立优化器。
- [专家数据实现](../../research/tgppo/experiments/paper-aligned-20261106/code/expert_data.py)：合法范围、标签映射、候选数限制与域差异。
- [专家审计](../../research/tgppo/runs/light-actor-20261105/offline/data_audit.json)：12/3实例与1468/288状态、原始60文件SHA。
- [BC seed11记录](../../research/tgppo/runs/paper-aligned-20261106/training/mlp_bc-s11/bc/result.json)、[seed22](../../research/tgppo/runs/paper-aligned-20261106/training/mlp_bc-s22/bc/result.json)：1000步完成、选100步、留出一致率与NLL。
- [PPO配置](../../research/tgppo/experiments/paper-aligned-20261106/configs/protocol.json)：25实例×5种子、500目标、3600秒、AdamW等。
- [部署计时源码](../../research/tgppo/experiments/paper-aligned-20261106/code/project/agent/agent.py)：ActorPolicy.act计时包括输入转换与动作构造，不能称纯forward。

## 本轮实际结果

[完整评测目录](../../research/docs/methods/tgppo_paper100_test66_20261110/README.md)，冻结模型身份见[plan.json](../../research/tgppo/experiments/paper100-test66-20261110/plan.json)。本目录data复制主表、独立效应、逐实例配对和原始结果索引；method_evidence.json记录由主表计算的百分比与计时定义。所有读取源SHA由DELIVERY_AUDIT.json记录并在生成前后核对。

这批模型均为GPU源目录的固定100导出；后续CPU/GPU续训的latest不改变本次身份。66实例有历史测试暴露；只有seed11/solver0的全66面板；BC预训练两个种子的内部留出指标不能充当完整求解的两训练种子复现。

## 历史诊断仅用于设计动机

- [末端头重置](../../research/docs/diagnostics/tgppo_20261102_head_reset/REPORT.md)：重置未恢复预定响应；未唯一定位LN2。
- [完整Actor与tail8验证](../../research/docs/diagnostics/tgppo_20261102_end2end/REPORT.md)：弱响应没有稳定求解收益。
- [轻量模型文献调研](../../research/docs/methods/tgppo_light_actor_survey_20261105/REPORT.md)：更广模型选择依据。

这些旧协议结果不与本轮3600秒绝对数值拼接，不用于给某个结构细节分配因果收益。方法模块组合并不等于“首创MLP＋BC＋PPO”。
