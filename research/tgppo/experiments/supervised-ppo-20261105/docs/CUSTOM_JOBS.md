# 现在提交：持续专家监督的四个训练作业

仅轻量MLP＋现存BC初始化，比较原PPO与持续专家监督PPO。每组仍100episode，不扩大旧预算；不运行旧light-actor／CR命令。

## 1. 四条命令分别提交四个GPU Custom

每个申请 **1GPU、至少4个专用CPU核、32GiB内存，平台时限3小时**。CPU／GPU型号不限，不指定dell-gpu；使用既有共享盘与tgppo环境，不重装。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_job.sh run --variant control --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_job.sh run --variant expert --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_job.sh run --variant control --seed 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_job.sh run --variant expert --seed 22
```

`control`专家系数0，`expert`系数0.01；其余保持同一协议。每个逻辑PPO小批次补16个训练专家状态，直接复用原BC权重，不重新BC。使用新副本修复后的CUDA随机流，两种子各自做实际配对检查。

每作业预计执行1–2小时；内部累计上限140分钟，含导入、验证、必要原生归一化、GPU合成桥接、训练与导出。排队另计，平台3小时不放宽内部预算。匹配CPU/软件的原生归一化缓存缺失时才补75项，4worker理论时限约37.5分钟，计入当前140分钟；同CPU四作业共享缓存与锁。

训练完成自动导出0／20／100 Actor，保存训练与监督次数、初始化配对、目标GPU检查结果，随后退出释放GPU。**GPU作业不会接着跑CPU评估，也不会自动训练更多episode。**

## 2. 四个训练完成后提交一个CPU Custom

申请 **4CPU核、32GiB、0GPU、平台时限1小时**，执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_cpu.sh run
```

无需等Codex回来汇总。此作业自动完成110项开发评估：BC0两份＋control/expert的20/100各两份，共100模型项＋10本轮同机新原生。两个条件共享相同BC0，不重复测。预计30–40分钟，累计上限50分钟，无GPU，无66测试。

提前提交时，入口会说明尚缺哪些训练并立即退出，**不等待占核、不创建评估attempt、不消耗50分钟预算**；四训练完成后再运行同一CPU命令即可。

正常结束自动生成：

- `research/docs/methods/tgppo_supervised_ppo_20261105/results/REPORT.md`：BC-only、固定100episode与开发选模结果、两种子独立增量、证明率、成本、预定筛选判断。
- 同目录`summary.csv`、`instances.csv`、`selection.csv`、`fixed_budget_effects.csv`、`selected_effects.csv`、`training.csv`、`tasks.csv`、`missing.csv`及`analysis.json`。
- `research/tgppo/runs/supervised-ppo-20261105/development-110/`：持久JSON/tar、权重SHA、主机身份、runner.log、job_state.json。节点/tmp消失后仍可汇总。

四训练同时排上且缓存就绪，预计从开始执行到开发报告约2–3小时，不含排队；不是保证获得正收益。若训练到限或失败，不把不完整矩阵写成“完成”，不自动扩大预算或进入测试。

## 查看与恢复

训练状态（可在CPU环境查看）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_job.sh status
```

CPU评估状态／已有结果重新汇总：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_cpu.sh status
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/supervised-ppo-20261105/custom_cpu.sh summarize
```

GPU失败先看RUN下`launcher-logs/`及`jobs/`，确认可恢复后在原训练命令末尾加`--resume`；沿用剩余140分钟，已完成不重训。CPU已有attempt不会重复执行或自动重试，故障时保留目录交给Codex定位；不要删除状态或清零预算。强制SIGKILL/节点丢失可能使末尾自动报告来不及完成，不能凭平台绿灯假定110项齐全，以持久状态和有效计数为准。

本地仅完成合成更新／恢复与入口验证，未启动真实训练或新增求解。目标CUDA检查会在用户Custom上实际执行，通过后才训练；当前不宣称目标GPU已验证。严格规则见[PROTOCOL.md](PROTOCOL.md)。
