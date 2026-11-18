# 固定100权重：完整66实例首轮评测

本轮仅评测训练seed11的B0、B0_new、MLP、MLP＋BC，使用各自固定`actor-0100.pt`的独立只读副本。用户已选择完整66测试实例、solver seed0；新增66份同机原生relpscost，共330项。单次求解3600秒、watchdog4200秒，部署只加载Actor，不加载Critic。原500episode正式评测入口及正在进行的训练均保持原状。

## 提交4个CPU Custom

每个作业申请 **4 CPU、32GiB内存、0 GPU**，CPU型号不限；四个合计16 CPU、128GiB。每个作业只执行下面一条命令，四条可同时提交。工作目录不限，共享目录须可见，沿用现有tgppo环境。无需申请V100或A100。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh run --shard 0
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh run --shard 1
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh run --shard 2
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh run --shard 3
```

| 分片 | 实例数 | 求解项数 | 执行安排 |
|---|---:|---:|---|
| 0 | 17 | 85 | 4个单线程worker |
| 1 | 17 | 85 | 4个单线程worker |
| 2 | 17 | 85 | 4个单线程worker |
| 3 | 15 | 75 | 普通14例4worker；结束后mspp16的5项单worker串行 |

普通worker地址空间上限6GiB；mspp16为24GiB，且不会与该分片普通worker重叠。沿用论文对齐包的评测内存策略；不是将训练的64GiB门槛套到推理。

## 耗时与停止条件

每项SCIP求解仍3600秒，不额外缩短。若所有任务都触顶，含mspp16串行安排，四作业同时运行的纯求解排程约23小时；按4200秒watchdog窗口约27小时，另留少量准备、读写开销。这是触顶估计，易例提前证明会缩短实际时间，排队另计。若平台必须填总时限，建议至少30小时；本脚本没有累计48小时/两小时等总时限。

每分片完成固定清单即退出。无自动重试、扩实例、加种子、训练或按测试结果选模。失败项保留，只有显式`--resume`才补跑；不把watchdog失败当合法超时结果。全部330项形成有效配对后才生成总排名。

## 中断续跑与查看进度

例如分片2中断，在新CPU Custom中执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh run --shard 2 --resume
```

其他分片替换数字即可。所有原始worker JSON完成即原子写入共享盘，缓存仅用于读模型和实例，`/tmp`丢失不会丢已落盘结果。同实例五方法必须来自同一CPU/软件/主机/核心配额环境：同环境补缺项；换环境保留已完整的实例面板，对尚不完整的实例在新环境重新配齐五方法。保留旧attempt，不删除锁、原始记录或预算账目。重复提交同一分片会被执行锁拒绝。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh status
```

四分片均结束后自动汇总，也可手动执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper100-test66-20261110/custom_job.sh summarize
```

进度：`research/docs/methods/tgppo_paper100_test66_20261110/progress.json`。

最终：同目录`REPORT.md`、`summary.csv`、`paired.csv`、`independent_effects.csv`、`results.json`。原始数据：`research/tgppo/runs/paper100-test66-20261110/shard-*/attempts/*/raw/*.json`。

## 本轮能回答什么

- B0_new对B0：Actor更新阶段启用train/dropout后，第100episode求解响应是否改变。两次训练环境/随机演化有差异，不能将单次差异视为纯粹的模式因果估计。
- MLP对B0与B0_new：轻量结构在这份预算下的求解效果与CPU部署成本；MLP无Dropout/BatchNorm，旧/新更新模式的有限切片等价证据见既有核验。
- MLP＋BC对MLP：专家BC初始化的独立增量。
- 四模型分别对原生relpscost：PDI、证明率、节点数、PAR-2、求解墙时与Actor推理时间。

主表使用全部66，另外报告共同调用Actor、全无Actor、混合参与面板。未调用Actor的变化不能归因于网络。PDI上下界为测量界，不是统计置信区间。当前只含一个训练种子和一个solver种子、100episode，不是最终500episode×solver0—4的论文完整结果；这些测试实例已有历史暴露，不称全新盲测，也不用其结果挑checkpoint。完成后提交分析供验收。
