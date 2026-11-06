# 今晚只提交这一个CPU评估作业

申请 **4个专用CPU核、32 GiB内存、0 GPU、平台时限1小时**。CPU型号不限，不指定dell-gpu；使用原共享目录与tgppo依赖，不重装。提交命令：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/dev-curve-20261105/custom_job.sh run
```

只需一个Custom，不运行旧light-actor训练入口。预计开始执行后20–30分钟，程序累计上限40分钟，包含导入/验证/求解/保存/汇总；排队时间另计。平台1小时不是放宽内部预算。关闭IDE、退出Codex不影响由集群托管的作业。

自动评估现有MLP＋BC的BC初始化、PPO20、PPO100，训练种子11/22，原5开发实例×solver0/1，共60模型＋10同机新原生＝70项。没有新增训练、GPU或66测试集任务；不需要分开申请三个时间点，以确保同主机比较。

完成后自动生成：

- `research/docs/methods/tgppo_light_actor_20261105/dev_curve_20261105/REPORT.md`：学习曲线、证明率和部署成本，完成或失败都会报告。
- 同目录`summary.csv`、`instances.csv`、`paired.csv`、`panel_effects.csv`、`selection.csv`、`tasks.csv`、`missing.csv`及`analysis.json`。
- `research/tgppo/runs/light-actor-20261105/development-curve-70/`：`runner.log`、`job_state.json`、身份清单、冻结权重副本、持久`evaluation_rows.json`与`raw_results.tar`。节点/tmp消失后仍能汇总。

可选查看（不求解、不加载Torch）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/dev-curve-20261105/custom_job.sh status
```

正常结束无需手动summarize。如只需重新生成报告：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/dev-curve-20261105/custom_job.sh summarize
```

重复提交会被共享锁拦截；已完成不重复执行，失败/中断也不会自行重试或重置预算。失败看`runner.log`和报告，保留原目录再交给Codex定位；不要删除状态后再跑。集群强制SIGKILL/节点断电可能使自动报告来不及完成，已有共享盘记录保留，重新汇总不会新增求解。

本交付仅完成入口和离线校验，尚未在Custom实际执行70项；最终以`job_state.json`和`REPORT.md`为准。方案见[PLAN.md](../../../docs/methods/tgppo_light_actor_20261105/dev_curve_20261105/PLAN.md)。
