# 旧 paper-aligned 作业换 GPU／CPU 续训

用于已经有 `latest.pt` 的旧实验，默认显式恢复，不能启动全新训练。
建议每作业 1 张 V100（优先 32GB）、4 CPU、64GiB，CPU 型号不限；沿用原环境和共享磁盘。

seed11 三个作业分别执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resume-20261108/custom_resume.sh run --variant b0 --seed 11 --resume
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resume-20261108/custom_resume.sh run --variant mlp --seed 11 --resume
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resume-20261108/custom_resume.sh run --variant mlp_bc --seed 11 --resume
```

seed22 同样支持，把 `--seed 11` 改为 `--seed 22`。同一条件只启动一个作业。

入口调用冻结的旧训练程序，保留模型、两套 AdamW、RNG、进度及累计 48 小时预算。
Actor 更新模式、GAE、每完整 episode 后 3 轮更新、500 episode 和 3600 秒求解上限均不变。
这仍是旧实验续接，不是新的 `paper-trainmode-20261108` 实验。

当前 seed11 三组已提交 91 episode；第 92 次采样未完成且没有 pending 轨迹。
更换 CPU 时，旧记录保留，新采样写入该 episode 的 `resume-cohorts/<身份哈希>`；
只允许 CPU 型号变化，checkpoint、任务、动作种子、软件／绑定／架构必须一致。
新 CPU 的原生归一化由原程序按新身份补测，不复用异机分母。
已有 pending 的作业仍优先恢复完整轨迹更新，不重新求解该轨迹。

本地仅做合成缓存路由检查和冻结文件校验，没有训练或新增 SCIP 求解。
V100 的真实 CUDA 检查由作业启动时原程序执行；换硬件不保证逐位一致，
3600 秒墙时限制下不同 CPU 也可能改变采样轨迹和归一化结果。
V100 作业仍需遵守平台 GPU 利用率回收规则，换卡本身不能保证不会再次被释放。

只读入口检查：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resume-20261108/custom_resume.sh check
```
