# 四个CPU Custom：冻结方法的16实例测试子集评估

> **已被用户批准的全66方案替代。** 四个旧作业尚未提交；旧入口现转入新包。请使用[全66的四作业说明](../../frozen-test66-20261105/docs/CUSTOM_JOBS.md)提交完整0–3，下面16实例正文仅保留历史记录，不再按此范围启动。

本轮是**原66测试目录中的固定16实例扩展**，不是开发集，也不是全66或完全未查看过的保留测试。使用此前冻结的B0／MLP×无／有BC×训练种子11／22，共8份100episode PPO权重；不使用正在训练的持续监督分支。

每个Custom申请 **4个专用CPU核、32GiB内存、0 GPU**，CPU型号不限，平台时限建议1小时。每个作业程序累计上限50分钟，四作业同时运行预计约40–50分钟，不含排队；到限保存缺项，不保证全部完成。四条命令分别提交到四个作业，不能在一个4核作业里同时运行。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh run --shard 0
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh run --shard 1
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh run --shard 2
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh run --shard 3
```

共享盘路径和已有tgppo环境须与之前Custom相同；不安装新依赖。脚本自动进入正确目录、选择既有Python、隐藏GPU并限制每个worker为单线程。可与现有GPU训练并行，不修改其源码、配置或输出。

| 分片 | 实例 | 数量 |
|---|---|---:|
| 0 | neos11、bab5、neos-820879、map16715-04 | 64模型＋8原生 |
| 1 | neos-801834、aflow40b、opm2-z7-s2、ns1208400 | 64模型＋8原生 |
| 2 | neos-662469、seymour1、rout、neos-829552 | 64模型＋8原生 |
| 3 | neos-612162、mik-250-20-75-4、neos-686190、ns1830653 | 64模型＋8原生 |

总计288项，每实例solver种子0／1。同实例全部方法和本轮新relpscost在同一作业内配对。输入及权重先核对SHA，8个实际Actor的CPU父子进程概率/动作一致性通过后才求解；不训练、不选模、不追加实例或重试。

最后一个作业结束后自动更新总报告：

`/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_test16_20261105/REPORT.md`

同目录的`summary.csv`、`effects.csv`、`paired.csv`分别给出主指标、结构与BC独立增量、逐实例正反例；`cohorts.csv`记录CPU和成本，`missing.csv`记录缺项。原始持久JSON、tar和日志在`research/tgppo/runs/light-actor-test16-20261105/shard-N/`。四组未齐时报告明确显示未完整，不按部分矩阵排名。

以下命令只读状态或已有结果，不新增求解；不需要为其另申请作业：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh status
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test16-20261105/custom_job.sh summarize
```

同一分片有锁与一次attempt保护；不要删除状态文件重新启动。节点强制杀死可能打断末尾保存；应先检查持久文件，不能把平台“结束”视为72项有效完成。完整协议见[PROTOCOL.md](PROTOCOL.md)。
