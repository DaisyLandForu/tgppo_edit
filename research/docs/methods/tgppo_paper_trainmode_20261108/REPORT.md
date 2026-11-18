# TGPPO 正文协议训练副本与影响说明

2026-11-08，北京时间。状态：独立训练入口和本地检查完成，等待用户手动提交Custom；没有启动真实训练、GPU或SCIP求解。旧六作业、源码和权重未改。

## 采用的方案

用户明确选择“论文正文＋标准GAE”：全25训练实例×solver0–4，500完整episode、单次3600秒、每episode后3epoch AdamW。Actor采样eval、更新train；GAE分episode并正确处理任务终止／采集截断。公开代码多轨迹合并和额外done遮罩不复制，这一点不能称为作者最终流程逐字复刻。

独立包：[paper-trainmode-20261108](../../../tgppo/experiments/paper-trainmode-20261108/README.md)。[六条启动命令](../../../tgppo/experiments/paper-trainmode-20261108/docs/commands.txt)和[配置](../../../tgppo/experiments/paper-trainmode-20261108/configs/protocol.json)可直接审阅。

## 哪些调整影响学习

| 项目 | 相对正在运行的paper-aligned | 对效果的判断 |
|---|---|---|
| Actor更新模式 | eval改train；B0有效dropout恢复.05 | 会改变梯度和正则，影响可能实质性；大小与方向未量化，不能承诺修复弱响应 |
| GAE／episode边界 | 保留已有标准GAE和正确边界，本轮不改回公开代码额外遮罩 | 直接决定优势与价值目标；对当前旧版本而言不是又换了一套GAE |
| 更新组织 | 保留逐完整episode后3epoch；不恢复多轨迹批更新 | 已与正文对齐；不同组织会改变策略刷新频率、步数和归一化范围 |
| 训练数据、时限、优化器、奖励 | 继续500／3600／25×5／AdamW／H3 | 不降预算、不换任务；120秒历史实验与本轮仍不可直接等同比较 |
| B0和MLP网络 | Actor／Critic／modules／light_actor四文件与旧版本逐字节相同 | 公共训练变化不算MLP结构创新；MLP本来没有dropout，实际仍0 |
| 设备 | 默认CPU，与作者公开入口一致；提供显式CUDA选项 | CPU／CUDA可能有不同数值与随机轨迹；同轮方法统一设备，不能把旧CUDA对新CPU的所有差别归因于模式 |
| 微批与随机流 | 保留加权微批、一逻辑batch一步、配对初始化和种子 | 不等于原大batch逐位执行；B0新增dropout会改变随机数消耗，边界明确记录 |

完整逐项说明见[TRAINING_DIFFERENCES.md](../../../tgppo/experiments/paper-trainmode-20261108/docs/TRAINING_DIFFERENCES.md)。TGPPO正文给dropout超参数但没有完整train/eval细则，模式关系参考公开Agent；标准GAE采用[GAE原始方法](https://arxiv.org/abs/1506.02438)。

## 新日志避免误读

旧eval模式仍可以反向和更新。新train模式下，dropout可在第一次optimizer.step之前造成非零clip；不能把clip从0变非0当作学习恢复。

新入口分别记录：eval采样概率一致性、初始train比值误差和dropout截断比例、整个更新的clip，以及最多8个第一微批训练状态上的eval分布更新前后TV／距均匀TV／argmax变化。这个小面板不是独立开发集，也不是PDI收益。

## 正常断点续训保留

“拒绝旧模式检查点”不是论文规定，只防止把历史eval更新进度混入新train实验。**同一新实验被释放后允许--resume**；模型、AdamW状态、RNG、轨迹及episode游标均恢复。PPO中断从pending重做未提交更新；提交完成后从latest进入下一episode。SCIP未完成求解需要重跑，不能恢复其进程内搜索树。

专项CPU检查模拟B0在Actor.step之后、Critic.step之前中断，从pending恢复后与不中断路径的模型／优化器／RNG逐位相同。跨硬件不承诺逐位等价；累计48小时和强杀缺记账的核对规则保留，不删除旧账自动扩额。

[新旧实验恢复命令与行为](../../../tgppo/experiments/paper-trainmode-20261108/docs/RESUME.md)。新实验必须从新初始化启动；旧实验可以在原模式继续恢复，没有自动取消任何作业。

## 核验与资源

20/20 CPU检查通过：实际五层B0与MLP合成PPO更新／导出、train/dropout和eval一致性检查分离、异常旧概率失败、GAE终止／截断手算、分episode不串联、短／空episode、AdamW／初始化配对、pending只提交一次、dropout中断精确恢复、只读复用归一化、2310覆盖及不完整矩阵拒绝排名。

测试检查累计91.940秒，含导入和首次失败；连同封存与入口核验，准备核验累计100.234/600秒。最终套件20项均通过。首次3处错误来自测试夹具未设置预算属性／重复device参数，修正后保留失败日志与记账，不覆盖成无失败历史。[CPU_CHECKS.json](CPU_CHECKS.json)、[PREPARATION_BUDGET.json](PREPARATION_BUDGET.json)。未执行真实episode、SCIP优化或CUDA，目标Custom仍会做实际设备检查。

默认推荐4 CPU／64GiB／0 GPU、每条件累计48小时；可显式统一CUDA。6组新资源上限288作业小时，只有用户提交后才执行。48小时不能保证完成500，不缩短3600秒凑数。

仅只读复用旧3600秒原生缓存且核对完整CPU／软件／实例／solver／cutoff身份；另存来源SHA，不向旧RUN写锁或结果，不复用旧PPO权重和轨迹。

训练完成后退出，不自动正式测试或扩预算。本阶段交付脚本和说明后等待验收／用户提交作业。

最终封存与入口核验通过：53个包文件、97个外部输入、166个历史保护文件不变；六个新条件均not_started。shell语法、冻结check和status均通过，证据见[FINAL_VERIFICATION.json](FINAL_VERIFICATION.json)。
