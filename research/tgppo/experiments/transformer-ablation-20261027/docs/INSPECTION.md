# 第一阶段检查（改动前）

本轮以真实文件和落盘 checkpoint 为准。2026-10-27；用户本轮明确授权独立 Transformer 消融，覆盖历史 Critic 研究的逐阶段限制；随后要求所有真实训练交由集群 Custom 作业，当前容器不启动训练。

## 基线与历史隔离

原版 `tgppo/repo/code`、修复版 stage2—5、服务器 `server-20261022` 均保留。Actor/Critic/modules 三文件在原版和 server 版逐字节相同。服务器 B0 是原架构，带已验证的训练正确性修复；B0-s11/s22 均有 500 episode summary、125/123 次更新。R0/Rjoint 是历史 Critic 改进，不纳入本次对照。早期 stage5 中断记录与本次完整 server6 结果不是同一个实验。现有 B0 训练约 7 小时/种子，不能在无 GPU 开发容器冒充重跑完成。本工作目录是独立复制，不覆盖原版、历史模型、日志或上传快照。完整受保护文件哈希见 source_inventory.json。

## 实际模型和张量

`project/policy/actor.py:Actor.forward`：候选 `[B,L,25]` → LayerNorm(25)+Linear(25,H) → `[B,L,H]`。node `[B,8]` 与 mip `[B,53]` 拼接，经 LayerNorm(61)+Linear(61,H) 成树 `[B,H]`。树广播后与候选拼接 `[B,L,2H]`，global_embedding 线性降至 `[B,L,H]`；这也是消融直接进入 BiMatching 的候选表示，不能删除这一步。

Transformer 输入 `[L,B,H]`，输出 `[L,B,H]`，转置为 `[B,L,H]`。`BiMatchingNet.forward(tree=[B,H], candidates=[B,L,H], padding_mask=[B,L])` 输出 `[B,L,H]`，TreeGateBranchingNet 输出 logits `[B,L]`，无效项填 -inf，softmax 返回概率 `[B,L]`。

`project/policy/critic.py:Critic.forward`：独立候选编码 `[B,L,25]→[B,L,H]→[L,B,H]`；没有 Actor 的 global_embedding。独立 Transformer 后转置、BiMatching 融合独立树编码；masked mean `[B,H]` + tree `[B,H]` 经 aggregation 和 Tree-Gated value_head 得 `[B,1]`。两网络不共享参数。本次一个 use_transformer 开关同时控制两网络；它检验整个原架构的 Transformer，并不能单独归因 Actor/Critic。

H=256，5 encoder layers，8 heads，FFN=1024，GELU，dropout=.05，batch_first=False；PyTorch 1.13.1 默认 norm_first=False，即 attention/FFN 两个残差后各 LayerNorm，eps=1e-5，无额外 encoder 末尾 norm。没有位置编码。输入输出同维，删除整个 encoder 也删除其内部归一化，不补新的 norm/MLP，以免增加变量。

BiMatching 通过 tree→candidates 和 candidates→tree 两路在候选轴 softmax，分别得到候选集合加权汇聚和树广播，再 sigmoid 融合；保留全部六个线性层和融合结构。

## 配置与训练路径

paper_hparams.json/runtime.DEFAULT_CONFIG：Actor lr=.00024，Critic=.00012，Adam，clip=.16，entropy=.003，gamma=.97，lambda=.92，batch=256，epochs=3，gradient clip=.5。H3 奖励文件保持不变。server.train 每四个完整 episode 收集后更新，4 单线程 SCIP worker，逻辑 batch=256，微批16，候选不截断，Actor rollout/PPO 重算均 eval（有效 dropout=0），Critic rollout eval、更新 train（.05）。各自 Adam；GAE/value 原单位；B0 Critic MSE，无校准/额外搜索信息。

20 train/5 dev 来自 docs/stage1/protocol_draft.json，66 test 清单来自 server-run/test66.json。120 秒/sandbox/最优 cutoff 协议，SCIP6.0.1、SoPlex4.0.1、自定义 PySCIPOpt1.4.9；不是完整论文3600秒/AdamW复现。66 test 已在历史研究使用，本次只能称冻结的既有测试集。

## 在线真实路径

server.evaluate → runtime.run_jobs → runtime._worker → load_actor → ActorPolicy → Environment.reset/run_episode → SCIP optimize → Brancher.branchexeclp → getCandsState(25,branch_count)、getMIPState(53)、getNodeState(8) → ActorPolicy.act → Actor.forward → Categorical.sample → branchVar(cands[action])。

Cython getCandsState 直接按 SCIPgetLPBranchCands 顺序同时生成候选对象和25维特征；返回 cands_pos 是原变量位置，不是策略 action 的索引。回调使用同序 cands[action]；批训练 padding mask True=无效，动作仍对应每行有效候选索引。

## 现存问题和测量边界

- 已接受修复：终止/截断 GAE、真实 gap、奖励实际时限、零动作 episode、Actor-only 部署、dropout 概率一致、微批累积与可恢复 RNG；本次两组共享。
- 原网络不接受空候选或全 padding：BiMatching softmax 全 -inf 会 NaN；在线由回调空候选 DIDNOTRUN、ActorPolicy 输入校验拒绝全 padding，memory 拒绝空候选。保持该有效状态合同并测试拒绝行为，不给一组单独补救。
- Critic mask pooling 假定 pad 为有限数，接口拒绝 NaN；不要用 NaN pad 作为有效输入。
- getNodeState 中 len(getLPBranchCands()) 是旧绑定 tuple 的长度，疑似上游特征缺陷；本次记录、两组保持一致，不改特征。
- getFairNNodes(brancher_name) 已支持，公式 Nodes+2*cutoffs+2*domain reductions；必须对已注册规则调用，避免空指针。本次补只读结果采集；不改优化。
- 既有 actor 回调时间包含张量转换及动作采样，不能标为纯网络延迟。本次额外记录 forward 时间与分位数，CUDA 前后同步；保留原计时字段。
- 原生 PDI 支持共同horizon上下界，保留超时与 SCIP infinity sentinel；cutoff infeasible 是优于最优cutoff区域的证明，不是原MILP不可行。

- 首次新增 Fair Nodes 采集探针遇到旧绑定 TypeError（expected bytes, str found）；只读调用改用 b"规则名" 后三次在线检查均通过。首次日志与无效结果保留在 online-check-first.log、online-check/transformer.json；它不是原架构缺陷，也不进入任何性能统计。

- 已核对历史 mspp16 内存故障及独立补跑：8GiB地址空间不足，24GiB成功。新两组评估共同保留此实例级资源修复：mspp16单worker、24GiB、容器32GiB、watchdog300秒，SCIP搜索时限仍120秒；其他65实例和训练维持8GiB。不会先明知失败再删除这个测试实例。
