# 统一开发评估的实际中断与续接

**后续授权与执行更新：用户已明确选择“本地CPU追加90分钟统一重测”。** 新目录为`research/tgppo/runs/light-actor-20261105/cpu-recovery-20261105`，旧预算和结果封存；最新状态见[progress.json](progress.json)。不再等待这项预算确认。

第一次本地尝试中，环境导入等待645.560秒，随后4个原生worker在约180秒watchdog退出，各CPU仅约1.4秒；检查时子进程处于Ceph元数据等待。该尝试已主动停止，实际记账共1067.476秒，**余下4332.524秒沿用，不清零**。旧Gold6148的两个超时没有当时的内核等待记录，不能断言也由同一原因造成。

独立恢复入口改为：从本地工作目录预载依赖、禁写字节码，数据/Actor按SHA等值复制到本地缓存；单线程CPU父进程、CUDA未初始化时才验证fork子进程。实际四种Actor概率及采样动作必须与父进程逐位相同，SCIP仅做Model创建/版本检查、不做额外求解；通过后才用原worker执行交错的170项任务。原冻结源码、网络、权重、SCIP设置、120/180秒上限和规范动作RNG不改。验证结果在新RUN的`cpu_fork_verification.json`，没有该文件不能称验证通过。

选择该方式参考[Python3.8进程方式说明](https://docs.python.org/3.8/library/multiprocessing.html)及[PyTorch1.13.1多进程注意事项](https://raw.githubusercontent.com/pytorch/pytorch/v1.13.1/docs/source/notes/multiprocessing.rst)：fork涉及线程/锁及CUDA限制，因此这里明确限定单线程CPU，并做真实权重概率/动作核查，不把它用于GPU训练。

新批次的逐任务输出先写本地缓存，避免Ceph元数据操作落入单任务watchdog；完整原始JSON及tar归档保存在共享RUN。所有失败保留，按尝试累计预算。下文保留中断时的两种选项作历史记录。

2026-11-05北京时间15:55，原最后一个Custom的finalize因SET+BC-s11两个`roi5alpha10n8`任务触发180秒watchdog退出。不是聊天中断导致训练丢失；16份100episode权重均完整。

- 已提交10个完整模型批次100项；SET+BC-s11另有8项有效原始结果、2项watchdog失败。共108项有效模型结果，缺52项；10项缓存原生参照仍完整。
- 原评估累计预算5400秒，已计费2822.625秒，剩余2577.375秒（42分57.4秒）。记录保留，不清零、不把异常当作正常完成。
- 原评估为Gold6148；当前本地为Silver4214。直接补52项并与Gold6148混合绝对PDI/墙时会引入硬件混杂。不能将原来的“同一CPU评估”条件悄悄改成混合CPU。
- 已向用户提交两种具体续接选择，等待回复期间仅整理已有记录，不新增求解或训练。

## 选项一：本地CPU统一重测（建议，待追加预算授权）

新增一次累计90分钟的纯CPU评估预算，当前本地4单线程worker、CUDA隐藏，160项模型+10项新原生，共170项。保持同一冻结checkpoint、SCIP设置、120秒时限、solver种子与配对动作RNG；原Gold6148结果/失败全部封存，新Silver4214为独立完整cohort，不混合旧绝对时间。没有训练，无需GPU或Custom。

纯评估只需要5开发实例×2solver的10项原生参照，**不重测训练归一化的75项**。预计约70–85分钟，累计90分钟或北京时间22:00到限停止并保留缺失。此选项超出原90分钟评估上限，必须获得用户明确追加预算；不得通过新目录偷偷重置原预算。

## 选项二：Gold6148的CPU Custom续接（原剩余预算内）

用户申请至少4核、32GiB的**仅CPU作业**，CPU型号须匹配旧cohort的Gold6148，仅为复用本次计时cohort，不恢复训练CPU型号限制。不得给同机以外的新作业过强的物理同机保证：它是同型号/同软件续接，应记录新的host及负载边界。

保持CUDA隐藏，先确认CPU型号，再运行已经冻结的现有入口：

```bash
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
from pathlib import Path
cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
assert cpu=='Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz', cpu
print(cpu)
PY
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/light-actor-20261105/custom_job.sh finalize
```

仅在前置CPU核查成功后执行最后一行。原入口按身份复用108项有效结果，仅补52项；原有2个失败记录不删除。预计执行25–35分钟，不含排队，仍受原剩余42分57.4秒和22:00截止约束。此命令不运行PPO/GPU训练；CPU不匹配就不应启动原finalize，以免在剩余预算内重做整个矩阵。

两种方案均只完成现有开发评估。跨实例确认仍须通过PLAN.md的候选门槛；不因本次异常追加训练、修改权重或扩大watchdog。
