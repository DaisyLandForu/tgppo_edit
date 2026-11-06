# Custom 界面填写和提交顺序

本轮为4个离线评分头作业。沿用项目已使用的 Custom 配置方式：镜像选择现有 **scip**，Shell **bash**，持久目录挂载 **`/home/duweiyue25`**，工作目录可填 `/home/duweiyue25/tgppo-transfer`。命令使用绝对路径。复用已有 tgppo Conda 环境，不运行 pip/conda 安装。

每个真实对照作业：**1 CPU / 8 GiB RAM / 0 GPU，平台时限15分钟**。程序硬限制每作业600秒，并且最多20个评分头 Adam 步。四作业可以并行；不要申请V100来执行本包，也不要让两个Custom使用相同job ID。

## 1. 一次准备和预检

准备只读已有 checkpoint/轨迹并冻结缓存，无参数更新、无SCIP。可在当前环境执行；若本地准备已完成，重执行会核验后跳过，不重新生成。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh prepare
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh preflight
```

`preflight` 必须退出0再提交训练。若共享目录或代码不同，不要跳过校验或删manifest；用新的独立实验目录/预算明确迁移。

## 2. 四个独立Custom命令

下面每段对应一个独立Custom，不是要在同一个作业中按顺序粘贴四条。

**Custom 1：original-s11**

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh run original-s11
```

**Custom 2：linear-s11**

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh run linear-s11
```

**Custom 3：original-s22**

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh run original-s22
```

**Custom 4：linear-s22**

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh run linear-s22
```

资源和完整命令的结构化清单：[../configs/custom-jobs.json](../configs/custom-jobs.json)。该文件只供填写界面，脚本不会调用平台API或自动提交作业。

## 3. 查看状态和汇总

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh status
```

输出同时给出记录状态、实际进程锁持有情况、已提交步数和封存情况。旧 running 标志不作为活进程证据。

四个Custom均退出0后：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh summarize
```

结果在 `research/tgppo/runs/offline-head-probe-20261102/summary/REPORT.md`，以及 `paired_results.csv`、`curves.csv`、`summary.json`。缺任何一组或数据不一致时，汇总会非零退出并写completeness.json，不伪造完整比较。

## 4. 日志、终止和重提

各作业独占 `jobs/JOB_ID`，launcher-logs 下每次启动独立日志；每个更新步保存 latest.pt、steps.json，保留 history.json 和 state_metrics.csv。原源目录和其它作业目录不回写。

成功作业再次提交相同命令，只核验并跳过。失败、超时、平台SIGTERM或未提交inflight状态会原样保留；脚本不提供自动清零重训。请保留目录交回审阅，勿删除budget、latest或inflight后重跑。这样可以避免重复更新冒充续接，也不会留下被父进程丢弃的后台子进程。

自测入口（仅合成小数据）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/offline-head-probe-20261102/custom_job.sh tests
```

本轮不含在线采样、SCIP基线、开发集选模、测试集求解或正式GPU训练。四组完成后交付验收，不自动续跑其它阶段。
