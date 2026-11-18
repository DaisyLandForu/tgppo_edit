# 新训练采用的正文协议与实现边界

2026-11-08。用户明确选择“论文正文＋标准 GAE”，而非逐字执行公开代码中的GAE和多轨迹批更新。

依据：[TGPPO §4.3／§5.3](https://arxiv.org/html/2511.12986v1)、[作者Agent](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/project/agent/agent.py)、[作者train.py](https://github.com/benmhamed-a/tgppo-aaai-26/blob/4f90ccddd7999779c93ff2425ca8dd056113a523/code/train.py)。正文没有逐项定义train/eval和终止遮罩细节；前者参考作者代码，后者采用标准GAE，不冒称取得了作者实际最终运行配置。

| 项目 | 新训练设置 |
|---|---|
| 数据与solver | 全25训练实例；solver seeds 0–4；125组合各轮独立洗牌，共4轮 |
| 预算 | 每方法／种子500完整episode；单次SCIP3600秒；watchdog4200只处理异常 |
| 更新组织 | 完成一个episode后完整遍历3epoch；逻辑minibatch256；非空短episode也更新；空episode照计且无虚构更新 |
| Actor模式 | 采样／部署eval；PPO反向train；B0 dropout=.05；MLP／MLP＋BC模型中没有dropout，实际为0 |
| Critic模式 | 采样eval、更新train；原Critic结构；独立参数，不新增状态 |
| GAE | 每episode递推，gamma=.97、lambda=.92；只用当前transition的done遮罩；明确任务终止与采集截断 |
| 终止 | 证明或本轮3600秒任务截止视为有限任务结束，bootstrap=0；不把SCIP时限自动改成无限任务截断 |
| 采集截断 | 若将来出现采集端截断，必须保存实际下一决策状态并bootstrap；当前完整episode入口不主动截断 |
| 优势／价值 | episode内优势归一化；Critic拟合原单位return的MSE；不新增价值归一化方法 |
| 优化器 | AdamW，lr=.00024/.00012，beta=(.9,.999)，eps=1e-8，weight_decay=.01，梯度范数裁剪.5 |
| 奖励 | 原H3；3600秒原生归一化，禁止复用120秒分母 |
| 模型 | B0原Actor／Critic／modules逐字保留；MLP24493参数，B0 Actor4738895参数；不新增结构或专家持续监督 |
| BC | 仅MLP＋BC，复用合法训练专家NPZ，重新1000步BC；内部固定NLL规则选初始化；PPO重新建立AdamW，不带BC动量 |
| 选模／评估 | 固定第500episode；0/100/200/300/400仅保存；66×5solver正式评测必须另手动启动 |
| 执行设备 | 默认CPU，与公开作者入口的设备选择一致；可显式--device cuda，同一轮统一选择且不在续训时切换 |
| 资源上限 | 每条件／种子新累计48小时；断点续跑不清零，到限未完成500则保留进度，不自动追加 |

GAE采用：`delta[t]=reward[t]+gamma*(1-done[t])*V(next)-V[t]`，`A[t]=delta[t]+gamma*lambda*(1-done[t])*A[t+1]`。绝不把下一条done再乘一次，也不把不同episode接成一条轨迹。

## 仍然不能称为作者完整原样运行

- 正文每episode更新与公开train.py合并多条轨迹后更新有差异；本轮按用户选择跟正文，不复制该入口批组织。
- 标准GAE和已验收的边界处理与公开Agent不逐字一致。这不是未公开设置，也不计入MLP创新。
- 按候选长度组织加权微批，基础微批8，候选注意力单元预算1048576；每逻辑batch仅一步AdamW。不截候选、不静默减批或改学习率。dropout掩码和浮点累积不等于原大batch逐位结果。
- 11/22的初始化、日程、动作和每episode PPO种子明确配对；不同结构会消耗不同数量的dropout随机数，不宣称更新后的全部随机流逐位一致。
- 未重新做嵌套Optuna，也未取得作者正式权重。原66已有历史暴露，不称新盲测。作者公布easy/hard名单实际33/32，缺1例，完整66不删例。
- CPU型号、PyTorch／SCIP与绑定均记录；归一化按完整身份匹配。CPU与CUDA可产生不同随机／数值轨迹，不混在同一方法主比较里。

## 运行与结果保护

包、RUN、日志、检查点、恢复锁和预算均独立。新入口拒绝旧模式／旧实验的checkpoint；旧六作业及其预算不改、不取消。

可只读复用旧paper-aligned中完整身份匹配的3600秒原生缓存，另记来源SHA；不锁、不写旧目录。不匹配则在用户提交的新作业中补测。不能复用旧PPO轨迹、优化器或权重作为新训练起点。
