# TGPPO 公开评审检索与复现一致性复核

日期：2026-11-08。补充回答“评审是否提过 B0 弱响应，以及复现不确定性从哪里来”。本次只检索公开资料、读取源码及核对文件身份，没有训练、求解、更新权重或改变现有作业。

## 结论

1. **未找到公开可访问的该论文评审正文，无法确认评审是否提过候选概率近均匀／clip fraction 持续为零。** “未找到”不等于“评审未提过”。
2. **当前并非作者整个训练流程的原样复现。** B0 的 Actor、Critic、modules 与在线作者源码逐字节相同，正文主要训练设置已对齐；但训练模式、GAE／边界、更新组织、设备与随机流存在已知差异。不能由三个网络文件相同推出训练流程相同。
3. **不确定性并非全因作者未公开设置。** 有些是我们主动保留的调整，有些是正文与公开脚本不能相互对应，还有些才是未取得作者最终运行配置／正式权重造成的缺口。之前“主协议对齐”的表述不能理解为“完整论文性能已复现”。
4. 现有证据确认的是：**作者公开网络在当前本地训练实现和已有权重上，在已检查状态中出现极弱候选概率响应。** 尚不能把它断言为论文正式模型的必然缺陷，也不能宣称已排除复现影响。

## 公开评审检索

论文为 [Learning Branching Policies for MILPs with Proximal Policy Optimization](https://arxiv.org/abs/2511.12986)，发表于 AAAI-26。[AAAI-26 官方说明](https://aaai.org/conference/aaai/aaai-26/submission-instructions/)确认使用 OpenReview 投稿，但使用该平台并不自动意味着每篇论文的评审向公众开放。

检索了论文完整标题、TGPPO、作者名、arXiv 编号与 OpenReview 的组合，也访问了 [AAAI-26 OpenReview 入口](https://openreview.net/group?id=AAAI.org/2026/Conference)。没有取得该论文的公开 forum／评审文本；直接 API 查询还遇到 HTTP 429，不能把请求失败当作“零条评审”。未绕过访问限制或联系作者。

同时通过 GitHub 连接器读取作者仓库当前 main 的完整树、Releases 和仓库 issue 搜索。返回树未截断，树 SHA 为 `4f90ccddd7999779c93ff2425ca8dd056113a523`；Releases 为空，本次 issue 查询返回空。当前 main 文件树中未发现正式模型权重或最终运行参数 JSON。**这个范围内的未发现，不等于作者在任何渠道都没有提供；GitHub issue 也不是同行评审。** 公开 README 链接的数据是 T-BranT 数据集，不能把数据链接当作取得了 TGPPO 正式权重。

检索范围、在线树与源码身份核对见 [PUBLIC_SOURCE_RECHECK.json](PUBLIC_SOURCE_RECHECK.json)。本地原版的11个相关文件均与在线 Git blob 对应；本轮 B0 的三个网络文件另与这些原版文件逐字节比较通过。

## 逐项对齐：明确差异与未知事项分开

| 项目 | 公开依据与当前状态 | 应如何归类 |
|---|---|---|
| B0 网络结构 | 本轮 `actor.py`、`critic.py`、`modules.py` 与作者在线源码相同；保留五层256维八头 Transformer、BiMatching、TreeGate | **确认一致的结构文件**；不是完整训练流程一致 |
| 最终训练主要设置 | 全25实例、solver 0–4、目标500完整episode、单次3600秒、AdamW与正文最终超参数；每完整episode后3优化epoch | **按正文对齐的目标协议**；不能写成已经完成500或复现论文指标 |
| Actor 的 train／eval | 作者采样时eval，更新时train；本轮采样和可微PPO重算均eval，关闭Actor更新阶段dropout | **公开代码明确，我们主动采用的调整**；不能归为作者未公开。eval不关闭梯度 |
| GAE与episode边界 | 作者源码包含额外的下一条done遮罩、末状态bootstrap与多轨迹合并；本轮保留分episode、终止／截断和实际下一状态等修复 | **已知实现差异**。修复不计作MLP创新，也不保证与作者实际实验轨迹等价 |
| 更新组织 | 正文§5.3描述每episode优化3轮；公开`train.py`先用同一份权重收集实例×seed的多条轨迹，合并后调用一次learn；本轮逐完整episode更新 | **正文与公开入口有差异**；本轮选择正文，不等于同时复刻公开入口。合并或逐episode还会改变优势归一化范围 |
| 短轨迹与逻辑batch | 原Agent不足batch256就返回；本轮非空短episode也更新。逻辑batch256使用加权微批累积，保留每逻辑batch一步AdamW | **主动选择／公共修复与工程调整**。微批改变浮点累积和Critic dropout掩码组织，不能声称逐位等价 |
| CPU／GPU | 公开`train.py`构建Agent指定CPU，Slurm脚本申请40核CPU；本轮CPU采样、GPU更新 | **已知设备与并发组织差异**。PPO目标形式相同，不代表数值／随机轨迹相同 |
| 初始化 | 网络构造代码可读，使用PyTorch模块构造初始化；依赖文件也列了版本。本轮明确设置训练种子11/22并隔离日程、初始化、动作、更新等随机流 | **不能说“作者没提供初始化方法”**；缺的是与论文最终运行对应的完整全局seed／启动配置／RNG记录。公开入口里worker设seed，不足以重建central learner最初状态 |
| 特征与solver | 保留25/8/53维和三个getter调用；本次对比当前stage4绑定与本地T-BranT绑定，三个getter函数体一致（Node比较仅带出后续函数装饰器的位置差异） | **已有明确实现来源，没有发现这里的新特征公式差异**。未拿到作者正式运行的绑定二进制／输入快照，不能升级成作者端逐值一致；也不能把接口列成已确认根因 |
| 参数搜索与测试 | 复用正文最终超参数，未重跑完整嵌套Optuna；旧66已被历史实验查看，本轮固定500不据其选模 | **明确的研究范围与数据暴露边界**，不是“作者没公开就无法做” |
| 作者正式权重与运行日志 | 本次所查main／Releases未取得最终权重，也未取得能与论文结果一一对应的完整启动命令和训练日志 | **尚缺的作者端证据**，不可用本地B0替代它作因果断言 |

主要源码依据（固定为本次所查版本）：

- [作者 Agent](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/project/agent/agent.py)：采样eval、更新train、GAE、样本阈值与损失。
- [作者 train.py](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/train.py)：建模、CPU设备、并行轨迹合并、初始化与更新顺序。
- [作者 train.sh](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/scripts/train.sh)：示例启动参数与CPU资源。
- [作者 requirements](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/requirements.txt)：依赖版本；不能说没有公开软件设置。
- 本轮既有决策：[PROTOCOL_ALIGNMENT.md](../../../tgppo/experiments/paper-aligned-20261106/docs/PROTOCOL_ALIGNMENT.md)。这些边界已有记录，但此前对外说明不应弱化为笼统的“可能有复现影响”。

## 为什么仍需要作者最终运行配置

公开的示例 `train.sh` 当前实际设置为：`TIME_LIMIT=600`、`SEEDS="0"`、`TRAIN_ITERS=12`，而正文最终训练描述为500完整episode、solver seeds 0–4、最多3600秒。Python入口默认值又是900秒、seeds 0/1/2、12次迭代。脚本还引用本次main树中未见的 `params/best_hparams_aggregate.json`。

这些是**公开文件内容的事实**，并不证明论文实际实验只用了600或900秒；示例和默认值可以被作者实际启动参数覆盖。但目前不能仅靠公开示例把正文结果的全部执行设置还原出来，更不能声称我们选择了唯一确定的作者执行路径。

## 对B0诊断和方法结论的影响

已经排除的是：本轮复制改坏了三个网络文件、完全没调用优化器等具体解释。尚未排除的是：更新模式／GAE与采样组织改变后，长期学习会与作者正式流程不同。**在既有权重上切换train/eval做几次前向，只检验即时输出，不等价于按另一模式重新训练；此前的模式检查不能被解释为排除了训练模式影响。**

结构压缩和候选差异衰减有直接源码与逐层证据，但“哪种设置决定长期停留在弱响应区”仍是待验证问题。所有方法共用修复有助于本地公平对照，却不能排除修复与不同Actor结构的交互；相对当前B0的提升不能直接外推为相对作者正式模型的提升。

当前六个训练任务、权重、求解时限与预算均保持原样。若后续继续定位，优先在独立副本用相同既有轨迹比较作者原Agent与当前Agent的target、初始ratio、梯度；先分离更新模式与GAE的影响，不重训完整500。该局部对照只能定位差异，不能代替正式权重或证明长期收敛；本次未启动它。取得作者正式权重／对应运行记录的证据价值最高，但本次没有联系作者。
