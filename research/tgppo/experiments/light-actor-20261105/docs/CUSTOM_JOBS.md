# 三种轻量 Actor：现在可提交的 Custom 训练

> **2026-11-05 15:20北京时间更新：16/16组（种子11、22）均已完成100 episode，最后一个作业正在自动统一CPU开发评估。无需再次申请或重复运行下面的训练/汇总命令。** 新增审阅与实时进度见[结果报告](../../../../docs/methods/tgppo_light_actor_20261105/results_review/REPORT.md)；当前不能把训练完成当作测试集或PDI收益已验证。以下为此前冻结的提交说明，保留作追溯。

2026-11-05（北京时间）。入口准备完成；**Codex 没有提交 GPU 作业，也没有本地新增 SCIP 求解。** 本轮已经完成合法数据复用、三个 Actor 实现、CPU BC/延迟和合成 PPO 恢复检查。BC 的可学性有信号，留出 NLL 门槛未通过，三个模型进入用户指定的探索性 PPO 对照；不能称为泛化或求解收益已验证。

## 申请方式

**每条命令一个 Custom：1 张 GPU、至少 4 个专用 CPU 核、32 GiB RAM，外层时限建议 4 小时 15 分。** V100/A100 均可，不限制 CPU 型号，不硬锁 GPU 型号；实际 CUDA 桥接不通过就停止，不偷偷回退 CPU。使用现有共享盘和 `tgppo` 环境（Python3.8、Torch1.13.1+cu117、SCIP6.0.1 定制绑定），无需重新安装。

首先提交下面 **8 个种子11作业**。它们已经包含原 Actor 控制及三个轻量结构的独立 BC 消融，不是8种新模型：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor b0 --pretrain none --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor b0 --pretrain bc --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor mlp --pretrain none --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor mlp --pretrain bc --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor set --pretrain none --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor set --pretrain bc --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor isab --pretrain none --seed 11
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh run --actor isab --pretrain bc --seed 11
```

资源充足时，建议把这8条各复制一次，将末尾 `--seed 11` 改成 **`--seed 22`**，共16作业；无需等待种子11结束。全部16条也在 [commands.txt](commands.txt)。尽量将同种子的各组放在相近硬件，但不以型号限制阻塞排队；环境、原生归一化及训练硬件差异均会记录。

不要运行旧 CR 入口。本轮路径明确是 `light-actor-20261105`。

## 自动完成什么

1. 核对冻结代码、协议、合法专家数据和软件；同 CPU/软件指纹共享原生归一化，优先复用已有75项原生记录，只有目标指纹缺失时才补测。等待和补测计入当前作业预算。
2. `--pretrain bc` 使用同一数据、实例均衡采样、1000 次 BC 更新、batch16、Adam lr=.001；按固定内部留出集的归一化 NLL，在0/100/200/400/600/800/1000步中选初始化。所有组跑满相同 BC 更新预算，严格并列取早。这个选择规则在提交 GPU 前冻结。
3. 以 BC 实际初始化或纯 PPO 初始化做目标 GPU 的合成 CPU→CUDA 更新/恢复检查，**不另加真实求解 pilot**。Actor-only 部署、原 Critic/H3/PPO 保持一致；BC 优化器不带入 PPO。
4. 真实 PPO：100个完整episode、4个CPU采样worker、每任务120秒/watchdog180秒、原20训练实例和solver seeds0/1/2，保存20/100。原网络控制也重新训练，不把旧异构作业直接当匹配控制。
5. 全部16作业结束后，最后一个作业自动在**同一CPU**上评估16模型×5开发实例×2求解器种子=160项，复用/核对同机原生参照，生成CSV和报告。不会启动66测试实例、CL、历史或树信用分配。

## 等多久、何时停止

- 上轮同规模100episode的6次训练实测64.9–68.4分钟。这轮**训练预计约1–2小时/作业**，BC/桥接通常较短；Actor变快也可能产生更多训练样本，因此尚不能承诺新模型更快训完。
- CPU指纹已有原生缓存时准备较短；缺失时75项120秒原生测量在4worker下理论时限合计约37.5分钟，另有启动/导入开销。**排队时间不包含在估计中**。
- 每GPU入口累计上限**140分钟**，包括导入、归一化等待/补测、BC、桥接和PPO；不是可以训练4小时15分。恢复使用剩余预算，已杀死但无最终记账的attempt按保守分配上限计费。
- 全部模型统一评估另有**累计90分钟**上限；因此完整训练和评估从开始执行起预留约2–3小时较合理，最坏受预算停止，不能保证全矩阵齐全。平台4小时15分为最后完成者的统一评估预留空间。
- 16作业训练上限合计37小时20分；加90分钟评估和10分钟本地准备，上限共**39作业小时**，这是并行耗时相加，不是需要用户等39小时。若先跑单种子8项，相应训练上限为18小时40分。
- 北京时间**11月5日22:00**为本轮截止；到限保留原始失败/成本/checkpoint，不自动扩到500episode。若100不齐，仅在全部所选组均有20时统一退到20；连共同20都没有则报告矩阵不完整。

## 状态、恢复与只跑8项的情况

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh status
```

失败后先查看 `research/tgppo/runs/light-actor-20261105/launcher-logs/`，确认原因后在**同一条原命令末尾**加 `--resume`。不删除 jobs、attempts 或 checkpoint，不重新计满140分钟。已完成命令不会重复训练。

自动汇总等待完整16项。如果资源只够先做种子11，8项全部结束后在现有**CPU环境**执行下面一条即可，不需另申GPU；输出单种子结果并保留为探索记录，之后仍可增加种子22，共享评估预算不重置：

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh finalize --seeds 11
```

完整16项的自动评估若被打断，使用同一入口的 `finalize`（不带`--seeds`）补齐未完成项；不会重训或重复匹配有效评估。

完整主结果：`research/tgppo/runs/light-actor-20261105/results/REPORT.md`，同目录 `tasks.csv`、`summary.csv`、`paired.csv`。仅种子11结果在 `results/seeds-11/`。分开提供同预训练条件的结构差、同结构的BC差，并标记没有Actor调用的配对，避免把纯计时波动当成方法收益。

本地实测和局限见[准备报告](../../../../docs/methods/tgppo_light_actor_20261105/REPORT.md)。实际目标GPU桥接和正式PPO仍须在上述作业中执行，当前没有GPU成功或求解改进声明。
