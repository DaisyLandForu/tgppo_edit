# 现有作业保留、旧seed22换卡恢复与新实验启动

2026-11-08，北京时间；进度为读取时快照，不替代平台进程存活检查。用户明确保持seed11三组现有A100作业不动，当前优先使最新训练模式开始运行。

本次只读日志／检查点并补充交接文档；没有取消、重提或启动任何训练／SCIP任务，没有修改任何训练包。

## 核验结果

- 旧`paper-aligned-20261106`的49个冻结文件全部与release SHA一致，新`paper-trainmode-20261108`的53个文件也全部一致。此前没有改动原训练脚本。
- 旧seed11 B0／MLP／MLP＋BC的完整latest均为episode89、phase ready；状态文件记录进入下一次原生归一化。模型、两个优化器、随机状态和memory字段均可CPU读取。B0 Actor优化器step4638，MLP4761，MLP＋BC3705。
- 旧MLP-s22已提交47个episode，pending保留第48个episode更新前状态及轨迹；Actor优化器step2280。旧MLP＋BC-s22已提交52个，pending保留第53个，step1902。latest与pending均可正常读取，均保存单张CUDA设备的RNG。
- 这7份检查点均为旧实验、有效Actor训练dropout=0，不能标成新train模式训练结果。新实验六组当前没有训练状态／latest文件。

## 旧seed22可以在V100恢复

现有旧入口没有A100型号硬锁；保持单张可见CUDA设备、原Python3.8／torch1.13.1+cu117及依赖、共享目录和配置，用原入口加resume：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp --seed 22 --resume
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp_bc --seed 22 --resume
```

每行一个V100作业。这里沿用旧入口，默认CUDA，不添加只在新入口支持的device参数。启动时先做实际V100合成更新／恢复检查；本次本地没有V100，不能写成跨卡续训已经验证成功。恢复权重与AdamW进度，不保证后续计算与持续使用A100逐位相同。

旧任务累计预算继续记账，MLP-s22约剩34小时45分钟，MLP＋BC-s22约剩35小时15分钟；不因为换卡清零。若新主机CPU／软件身份不同，后续需要按规则补匹配的3600秒原生归一化；pending更新本身已持有前次完整轨迹，不需要重新采该episode。

## 优先最新方法时使用新目录

最新方案是正文逐episode更新＋标准GAE＋Actor更新train。使用[V100新训练命令](V100_CUSTOM_JOBS.md)，例如：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-trainmode-20261108/custom_job.sh run --variant b0 --seed 11 --device cuda
```

首次从新初始化开始；以后同一新条件中断再加resume。旧seed22的恢复任务是补齐旧实验，不替代新模式实验。资源优先级按用户当前要求先考虑新实验，旧中断结果可保留，不自行决定取消旧seed11作业。

旧训练继续写`research/tgppo/runs/paper-aligned-20261106/`；新训练写`research/tgppo/runs/paper-trainmode-20261108/`。两个实验有独立模型目录、日志、锁和预算。新包只读访问匹配的旧原生缓存，不写旧权重或其保存位置；相同variant／seed标签不会导致覆盖。
