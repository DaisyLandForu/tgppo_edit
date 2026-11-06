# 完整66实例测试：四个CPU Custom

用户最终指定4个并行作业，平台剩余160GB内存。本说明替代未执行的16实例方案与准备中的九分片方案。

**每个Custom申请4个专用CPU核、32GiB内存、0GPU，CPU型号不限。** 四作业合计16CPU／128GiB（约137.4GB）／0GPU，低于剩余160GB。平台时限设4小时15分钟，程序累计上限4小时，包含全部准备、求解与保存汇总。四作业同时运行预计约2–3小时，不含排队，不保证故障时全项齐全。

| 作业 | 实例数 | 任务 | 执行方式 |
|---|---:|---:|---|
| shard0 | 18 | 324 | 4个单线程worker |
| shard1 | 17 | 306 | 同上 |
| shard2 | 17 | 306 | 同上 |
| shard3 | 14 | 252 | 13普通实例先4worker，全部退出后mspp16单worker |

**下面每条命令各提交一个Custom，共四个。** 不要把四条命令放在一个4核容器里同时启动。共享盘挂载及既有tgppo Python/SCIP环境与此前Custom相同，无新依赖。

作业0：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh run --shard 0
```

作业1：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh run --shard 1
```

作业2：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh run --shard 2
```

作业3：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh run --shard 3
```

全部66实例新测：B0／MLP×无／有BC×训练种子11／22的8份冻结100episode权重，solver0／1，1056模型＋132新relpscost＝1188项。同一实例所有方法和原生在同一作业主机配对；不填旧8实例结果、不训练、不选模，不读取正在训练的持续监督分支权重。可与其GPU训练并行。

普通worker地址空间8GiB、watchdog180秒。mspp16的18项在shard3普通池全部退出后才开始，单worker/24GiB/300秒watchdog；不会同时留下其他普通worker占用内存，且全部模型和原生采用相同例外。SCIP搜索时限始终120秒。每worker都是单线程，mspp16阶段只用一个核。

脚本先校验冻结输入、源权重与实例SHA，实际8个Actor的CPU fork概率/动作一致性通过后才优化。原权重、GPU代码、原基线保持只读。不同分片允许不同CPU，自动记录硬件；不把混合硬件绝对延迟当统一微基准。

最后一个作业结束自动更新：

`/home/duweiyue25/tgppo-transfer/research/docs/methods/tgppo_light_actor_test66_20261105/REPORT.md`

summary.csv、effects.csv、paired.csv、mean_decomposition.csv、cohorts.csv分别是主表、结构与BC增量、逐实例正反例、无Actor贡献分解、硬件与成本。四分片1188项有效才有全量主表；未齐明确列missing.csv，不用部分子集排名。负/混合结果正常保留，不自动调参或继续训练。

持久原始记录、tar和日志在`research/tgppo/runs/light-actor-test66-20261105/shard-N/`。无需为汇总另申请作业；以下只读操作不求解：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh status
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/frozen-test66-20261105/custom_job.sh summarize
```

共享锁与一次attempt保护防重复，不删除状态重置预算，无自动重试。节点强杀/丢失可能打断末尾保存，应先核对持久数据，不以平台“结束”认定全项有效。旧test16入口已转发本包，原四条命令也对应当前四分片；建议直接复制本页新命令。

完整66覆盖含mspp16，历史测试暴露边界仍须披露。严格协议见[PROTOCOL.md](PROTOCOL.md)，固定资源与实例分配见[shards.csv](shards.csv)。
