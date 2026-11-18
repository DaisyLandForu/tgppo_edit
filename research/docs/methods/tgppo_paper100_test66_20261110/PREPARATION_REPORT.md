# 2026-11-10 固定100评测准备报告

已按用户选择冻结四组seed11第100episode权重，评完整66实例、solver seed0；66份新原生relpscost与264份模型项共330项。原500主协议未改，本轮不训练、不按测试集挑权重。

| 标签 | 来源 | Actor SHA256前12位 |
|---|---|---|
| b0 | tgppo/runs/paper-aligned-20261106/training/b0-s11/actor-0100.pt | 25fcd50cf5c6 |
| b0_new | tgppo/runs/paper-trainmode-20261108/training/b0-s11/actor-0100.pt | 2e745f331f25 |
| mlp | tgppo/runs/paper-aligned-20261106/training/mlp-s11/actor-0100.pt | 39b2fc7ffe9a |
| mlp_bc | tgppo/runs/paper-aligned-20261106/training/mlp_bc-s11/actor-0100.pt | 8ee6f07ec547 |

完整来源、checkpoint SHA、配置和实例清单见新包plan.json；330任务见task_manifest.csv。实际CPU加载全部4份Actor、父/子进程概率与动作逐位一致：B0两组各4,738,895参数，MLP两组各24,493参数；无Critic/优化器构造、无CUDA初始化。11项合成检查通过，涵盖完整汇总、恢复、跨环境不拼接与无部分排名。旧49/新53冻结训练包及保护源码SHA保持。

四作业各4CPU/32GiB/0GPU，任务85/85/85/75。普通4单线程worker、6GiB地址空间；mspp16单worker、24GiB，与普通worker不重叠。SCIP3600秒，watchdog4200秒；无脚本总墙时限。全触顶纯搜索排程约23小时、watchdog窗口约27小时另加准备开销；提前证明会更快，排队另计。

每项完成即在共享盘原子保存，不依赖/tmp留存。显式--resume补缺项；完整实例五方法面板保留，跨环境不完整面板重新同环境配齐。结果齐全后统一比较新旧B0、MLP结构、BC与原生relpscost的PDI/证明率/节点/PAR-2/部署开销，并列共同调用/无Actor/混合面板。

本轮尚未启动新增求解。用户需手动提交四个CPU Custom；这不是500episode或跨种子稳定性结论，历史测试暴露保留。
