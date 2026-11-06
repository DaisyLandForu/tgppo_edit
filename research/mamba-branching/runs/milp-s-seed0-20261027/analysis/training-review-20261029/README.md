# seed0 正式训练结果核查

核查日期：2026-10-29（北京时间）。训练与验证集选模完成；测试求解尚未开始，不能据此认定复现论文求解性能。

CL 50/50 epoch，458,400 次更新；IL 50/50 epoch，4,900 次更新。原作业在CL更新102,875受控停止，新A100作业从102,876无间断接续。两阶段各50次验证记录齐全；没有failed/failed_batch事件、非有限更新指标或漏计更新。源码、执行快照、配置、数据清单、selected/latest checkpoint哈希均通过核对；检查的模型及优化器张量均有限。全部475条轨迹在本次复核时也成功预加载。

| 指标 | 结果 |
| --- | --- |
| CL最佳验证轮次 | 44/50 |
| CL最佳验证loss | -1.360250318 |
| IL最佳验证轮次 | 50/50 |
| IL选模loss（窗口内NLL求和，再跨窗口平均） | 84.730569595 |
| IL验证每状态NLL | 0.952592203 |
| IL验证专家动作准确率 | 73.701775% |
| 验证规模 | solver seed4，95条原始轨迹，475个100状态上限窗口，42,250个有效状态 |
| 冻结模型 | ../../training/selected.json |

IL第1轮到第50轮，验证loss从189.733降至84.731、动作准确率从45.89%升至73.70%。中间有波动（例如第35轮），随后恢复改善；最后一轮同时取得最低NLL及最高准确率，没有看到持续的验证退化。训练loss最终65.021，仍低于验证loss84.731，存在泛化差距。CL验证有波动，按照冻结规则保留第44轮。CL和IL损失定义不同，不作跨阶段数值比较。

当前V100-SXM2-32GB使用同一独立Conda环境重新加载最终embedding/Mamba，复算完整seed4验证集；loss、每状态NLL和动作准确率与A100训练记录完全一致。仅执行推理，无训练更新，无测试实例。A100的真实Mamba微批8对微批1探针已通过，最大张量尺度误差5.96e-8；正式IL峰值分配显存约904.7 MiB，合成容量探针峰值917.4 MiB。这些是PyTorch allocated指标，不等于整卡进程显存。

加速作业于北京时间2026-10-28 20:07:05启动，23:51:54冻结选模，共3小时44分49秒，包含剩余CL和完整IL。CL从首到末更新约3小时33分35秒，IL约10分36秒（阶段边界计时略有不同）。CL平均吞吐从旧作业3.13升到新作业27.74更新/秒，观察到约8.86倍；这是缓存/预取/日志/检查点优化及V100→A100、CPU主机变化等共同效果，且比较不同训练区间，不能当作A100单独加速倍数或严格对照基准。

原训练/执行配置与checkpoint未修改。`report.json`包含全部逐轮验证、日志事件汇总、哈希检查与本次GPU复核；`validation.csv`可直接绘图。`verify_trained_model.py`保留本次只读GPU复核脚本，输出路径为/tmp，未纳入训练冻结源码。

下一步按既定协议并行提交4个评估Custom：每个4 CPU、16 GiB内存、1 V100、48小时上限，当前SCIP镜像及相同持久挂载。每个作业只执行一个SHARD，分别为0、1、2、3。

```bash
ROOT=/home/duweiyue25/tgppo-transfer/research/mamba-branching
RUN=$ROOT/runs/milp-s-seed0-20261027
SHARD=0
bash "$ROOT/custom_job.sh" evaluate --run-dir "$RUN" --shard-index "$SHARD" --num-shards 4 --workers 1
```

全部4个作业成功结束后汇总：

```bash
bash "$ROOT/custom_job.sh" summarize --run-dir "$RUN"
```

共8个测试实例×5个solver seed=40对、80次求解；同一对relpscost/Mamba顺序运行于同一CPU主机。每次求解上限3600秒；冻结模型不再依据测试结果调整。作业配置见`configs/custom-jobs-milp-s-seed0.json`的mamba-evaluate-0..3。尚无测试节点、时间或PDI结果，73.70%的模仿准确率不能直接推导求解加速。
