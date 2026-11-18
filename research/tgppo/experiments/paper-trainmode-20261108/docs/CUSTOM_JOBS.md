# 新独立训练作业：按正文与标准GAE

2026-11-08。用户已选择正文协议，入口已准备；真实训练由用户手动提交Custom，不从旧PPO权重续训，不取消旧六作业。

## 推荐CPU入口

每条件／种子一个作业，4 CPU、64 GiB内存、0 GPU、48小时，CPU型号不限，尽量使六组CPU型号一致。原作者公开入口也在CPU上更新；平台无法豁免低GPU利用率回收，CPU训练可避免因此中断。该选择保持训练目标，仍可能改变数值和随机轨迹，不保证CPU更新比CUDA快。

容器沿用已运行的SCIP6.0.1、定制绑定、Python3.8与torch1.13.1+cu117；即便CPU训练也不自动升级或更换PyTorch。共享挂载`/home/duweiyue25/tgppo-transfer`。默认Python路径沿用既有环境，可用`TGPPO_PYTHON`指定同版本环境。

每行单独提交一个Custom：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant b0 --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant mlp --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant mlp_bc --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant b0 --seed 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant mlp --seed 22
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant mlp_bc --seed 22
```

B0／MLP／MLP＋BC×种子11/22，共6组。新累计上限每组48小时，不清零旧作业预算，也不把旧已训练episode算入新500。最多288新作业小时属于六组资源上限，提交前按实际可用资源决定并行数；脚本不会替用户申请任何作业。

阶段顺序：检查冻结源码／数据 → 目标CPU合成更新与恢复检查 → 新初始化（MLP＋BC额外做BC）→ 按需只读复用匹配3600秒原生缓存或补测 → 完整episode采样与PPO更新。原生归一化或采样可能持续一小时，不将其误判为GPU工作。

## 显式CUDA选项

如果仍希望GPU更新，申请1 GPU＋4 CPU＋64 GiB，六组统一改用：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 11 --device cuda
```

其余条件替换variant／seed。此时必须通过目标GPU检查；本地只验证CPU，不能声称A100已验证。正常的CPU求解阶段仍可能触发平台低GPU利用率回收，脚本没有伪造负载或改变3600秒求解上限。

## 支持正常断点续训：恢复与状态

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh status
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_cpu.sh run --variant b0 --seed 11 --resume
```

“拒绝旧模式检查点”不是论文要求，也不是禁止断点续训：仅禁止把历史eval更新的PPO进度混入新train更新实验。同一新实验因释放、抢占或时间到限而中断，可以正常恢复。

GPU条件恢复用原CUDA入口加`--resume`。不要在续跑时切设备或换variant／seed／协议；新包拒绝旧实验或eval更新模式的checkpoint。pending含更新前轨迹／优化器／RNG，latest是已提交状态，中断后只重放未提交的更新，不重复计episode。在相同设备／软件与配置下，本地合成检查验证了B0训练dropout开启时，中断在Actor／Critic步骤之间仍可从pending重放，恢复后的模型、AdamW状态和RNG与不中断路径逐位相同。这不是额外训练或重复更新。

若中断发生在SCIP求解或归一化途中，未完成的那次求解从头运行；不能恢复SCIP内存中的搜索树。已提交episode保留，不回滚此前训练。换硬件后的浮点／随机实现及按新CPU补测的归一化可能改变后续轨迹，不承诺跨硬件逐位一致。

48小时为累计预算，到限需先审阅进度，不自动扩额。正常SIGTERM会记账；若平台强杀／节点失联而缺少最终耗时，保留原预算记录，需按平台实际运行时长核对，不靠删文件清零预算。

输出：`research/tgppo/runs/paper-trainmode-20261108/training/{b0,mlp,mlp_bc}-s{11,22}`。核对`state.json`、`updates/`、`budget.json`、`latest.pt`；只有`summary.json`明确500完整episode才算训练完成。CPU入口日志也会持久到新RUN的launcher-logs。

## 学习监控与停止边界

Actor更新train，B0 dropout=.05；所有采样eval。`initial_dropout_clip_fraction`可能在任何优化步前非零，不能当作学会；同时看`deployment_tv_before_after_mean`及距均匀TV等小面板监控。

固定保存0/100/200/300/400/500；全部25都用于训练，不使用原5开发选模，不用66调参。500仍是目标，不能保证48小时完成，不能临时缩短求解以凑满。

训练结束直接退出，不自动评测。正式CPU评估仍需六组500全部完成，并另手动调用`custom_cpu.sh eval --shard N`；完整66×5×7＝2310项，原17分片入口已保留，但本阶段不申请或执行这些评测。先审阅训练结果和成本再安排。

[完整协议](PROTOCOL_ALIGNMENT.md)、[差异影响与日志解释](TRAINING_DIFFERENCES.md)。
