# 200 episode：四个并行CPU Custom

每个作业申请**4CPU／32GiB／0GPU**，CPU型号不限，四个总128GiB。四条命令分别填入四个作业，允许同时提交：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh run --shard 0
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh run --shard 1
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh run --shard 2
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh run --shard 3
```

| 分片 | 实例数 | 求解项 | 调度 |
|---|---:|---:|---|
| 0 | 17 | 85 | 4单线程worker |
| 1 | 17 | 85 | 4单线程worker |
| 2 | 17 | 85 | 4单线程worker |
| 3 | 15 | 75 | 普通14例4worker，mspp16五项单worker串行 |

每项3600秒搜索／4200秒watchdog。原100轮在现有4CPU服务器按四分片顺序用42.90小时，分片各约9.5—11.6小时；本轮模型与机器不同，不保证同样耗时。四作业并行全触顶的纯搜索排程约23小时，按watchdog窗口约27小时，另计准备与排队；平台要填写总时限时建议至少30小时。

权重固定为新B0-s22、MLP-s11、MLP＋BC-s11、MLP＋BC-s22的200；额外66份原生在各分片同环境重新测。没有旧B0-200，也不自动纳入之后新到200的模型。solver固定seed0，不将一次solver重复当作多种子确认。

显式续跑（示例分片2，替换数字即可）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh run --shard 2 --resume
```

共享盘逐项原子落盘。重复提交同一分片会被锁拒绝；换CPU／主机不拼接同一实例的不同方法，保留旧attempt。失败／watchdog不算合法超时，不自动改短时限或重试。

查看／汇总：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh status
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper200-test66-20261113/custom_job.sh summarize
```

未齐只输出progress；全齐输出REPORT.md、summary.csv、paired.csv、independent_effects.csv、raw_rows.csv、results.json，并另外输出paper_subset_metrics.csv。后者按作者固定easy33、hard32、未分组1例报告SGM；不按本轮输赢重新划分。

用户可以手动停止现有六训练作业再提交评测；本入口只读已冻结200权重、不停止作业。正在采样或更新的未提交工作可能需续训重做，但固定200不受影响。若以后补齐另两份200，应另冻结完整矩阵，不修改本轮已执行的plan。

相同训练种子下：BC_s11＝MLP＋BC_s11−MLP_s11；combined_s22＝MLP＋BC_s22−新B0_s22。缺少纯结构配对，不用MLP_s11对新B0_s22冒充独立结构消融。所有四权重对relpscost逐实例配对。
