# 新一轮Custom作业：论文主协议对齐

先提交下面**6个训练作业**。每行单独一个Custom，推荐A100×1、4专用CPU、64GiB内存、平台时限48小时。CPU型号不限，GPU型号记录但不在脚本中硬锁，避免增加调度限制；尽量为三方法申请相同A100规格与CPU型号。

容器继续使用此前可运行TGPPO环境，共享挂载`/home/duweiyue25/tgppo-transfer`、原定制绑定与SCIP动态库。默认Python：`/home/duweiyue25/conda/envs/tgppo/bin/python`（3.8／torch1.13.1+cu117）；可用环境变量`TGPPO_PYTHON`指定等价环境。脚本不自动安装／升级依赖，不自行提交平台作业。

## 现在提交

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant b0 --seed 11
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp --seed 11
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp_bc --seed 11
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant b0 --seed 22
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp --seed 22
```

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant mlp_bc --seed 22
```

正式设置是全部25实例×solver0–4、500完整episode、3600秒、每episode后3epoch AdamW；已独立对齐Critic和随机流。不沿用旧100／120协议，不从旧PPO权重续训。只有MLP＋BC额外先重做合法专家BC。

每个目标GPU作业先执行实际CPU→CUDA概率／PPO更新／精确恢复／Actor-only检查，失败立即退出；本地检查通过不代替A100检查。随后按需要补3600秒原生归一化，CPU型号／软件匹配时共享。阶段`normalization`可能持续一小时；不是入口卡死。GPU低占用通常意味着CPU正在求解。

## 查看与恢复

不需要GPU，秒级状态读取可直接本地执行：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_cpu.sh status
```

输出目录：`research/tgppo/runs/paper-aligned-20261106/training/{b0,mlp,mlp_bc}-s{11,22}`。重点看`state.json`、`budget.json`、`updates/`及`latest.pt`；平台“结束”不代表完成500，必须有`summary.json: training_complete=true`。每次入口日志自动保存在同一RUN的`launcher-logs`。

中断后使用同一条件、同一种子，加`--resume`：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_job.sh run --variant b0 --seed 11 --resume
```

其他5组同理。已有作业仍活跃时重复提交会被互斥锁拒绝；已完成组再次调用立即返回。不删除旧状态、不重新计episode或预算。当前48小时为每个条件／种子的累计执行上限，续跑消费余额；到限未达500时先汇报进度再确定是否追加。SIGKILL／节点失联缺记账时保守保留分配成本，不承诺无损自动续跑。

## 训练完成后再评估

六组500全部完成后才允许正式CPU测试；提前调用会立即退出，不占GPU、不创建评测attempt。没有自动在GPU作业末尾运行测试。

完整66×5solver×（6模型＋1原生）＝2310项。固定分17个CPU分片：0–15各4／5普通实例，16只处理mspp16。普通每分片4CPU／32GiB／0GPU、4个单线程worker；16建议4CPU／32GiB／0GPU，实际单worker24GiB串行，以免mspp16与其他任务叠内存。单分片累计48小时；先根据训练实际速度估成本，再提交所需CPU分片，可分批提交，**现在不用申请这17项**。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_cpu.sh eval --shard 0
```

替换`--shard`为1至16，每个独立作业。普通最大175任务，理论搜索上限43.75小时（4worker）；mspp16为35项串行，35搜索小时。watchdog、导入、I/O和过冲还会增加时间，不能保证48小时填满分片，支持显式恢复。各实例的所有模型与原生留在同一CPU主机配对。恢复评测要求原主机／软件保持；换机不静默拼接未完成面板，先审阅已有cohort再安排重测。

全部完成后本地汇总，无需另申请Custom：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-aligned-20261106/custom_cpu.sh summarize
```

报告：`research/docs/methods/tgppo_paper_aligned_20261106/results/REPORT.md`，底表`summary.csv`、`paired.csv`、`author_subsets.csv`。结果未齐只写进度，不按部分矩阵排总体名次。

## 耗时和交付预期

六个GPU并行时，等待主要由最慢一组决定，不是把六组时间相加；排队不计执行预算。3600是求解上限，不能拿旧120秒耗时直接乘30，也不能用A100峰值算力估CPU求解速度。48小时完成500要求平均每episode含准备与更新<5分46秒；目前没有真实长时限样本，不能保证明早拿到完整500及全66评测。

明早先核对实际完成episode／125组合覆盖、有效更新／AdamW步数、正在归一化还是采样、预算余额及目标GPU检查。未齐时给进度与剩余时间估算，正式结论等待固定500及完整2310项，不用短预算结果顶替。
