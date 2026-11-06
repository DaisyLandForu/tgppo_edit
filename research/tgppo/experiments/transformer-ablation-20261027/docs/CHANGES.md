# 相对 B0 副本的改动

初始本地提交 `c56cfd8` 保存改动前服务器实现。所有文件位于独立 `transformer-ablation-20261027` 目录；原始目录未修改。

| 文件 | 函数/配置 | 改动及实验影响 |
|---|---|---|
| code/project/policy/actor.py | Actor.__init__、forward | 新增默认true开关；false释放encoder并旁路到原BiMatching，保留global_embedding和原头；公共层初始化/RNG不变 |
| code/project/policy/critic.py | Critic.__init__、forward | 同时旁路独立Critic encoder；保持原masked pooling、aggregation和value_head |
| code/project/runtime.py | DEFAULT_CONFIG、ARCH_KEYS、architecture、config_with_protocol、build_agent、load_actor | 严格布尔配置，保存/加载架构；缺字段的旧checkpoint默认true，保留旧哈希 |
| code/project/agent/agent.py | ActorPolicy.__init__、_act | 额外记录forward-only耗时，CUDA同步；不改变动作或PPO公式 |
| code/project/brancher/brancher.py | Brancher.__init__、branchexeclp、stats | 收集每次forward样本、候选数和mean/P50/P95；保持cands[action]及原分支路径 |
| code/project/environment/environment.py | Environment.run_episode | 只读Fair Nodes（旧绑定bytes名称），完整实例墙钟（包括free）；不改SCIP参数 |
| code/server.py | params、select、evaluate | 透传架构开关；时限标签跟随实际配置；两组共同保留mspp16内存修复，其余复用原训练/评估函数 |
| code/ablation.py | protocol/check_pair、smoke/prepare、main、freeze、benchmark、aggregate/report | 对原server薄封装；控制变量、独立输出、真实GPU桥接、冻结防篡改、保留超时并汇总 |
| configs/*.json | use_transformer、variant、run_directory | 正式两配置仅三项不同，其他训练/求解条件一致；smoke单独缩短工程预算 |
| tests/test_ablation*.py | 14项检查 | 生产模型尺寸、原前向等价、梯度、mask/置换/索引、加载和协议/汇总/资源检查；不启动MILP训练 |
| custom_job.sh、env.sh | Custom命令与环境 | 固定当前真实Python/绑定路径；每次独立日志、显式恢复，训练必须CUDA |
| plot_results.py | main | 在独立报告环境绘制真实loss/reward日志，未渲染当前不存在的训练曲线 |

`project/policy/modules.py`（BiMatching/TreeGate）、`Agent.learn`、memory、reward、特征绑定均不改。不存在新的可学习补偿层。移除encoder内部norm及模型容量下降是该消融的组成部分，结论需同时承认这两个因素。

相对B0的源码、配置和测试补丁见 `ablation_from_b0.patch`；完整文档与实测证据由本目录Git历史保存。
