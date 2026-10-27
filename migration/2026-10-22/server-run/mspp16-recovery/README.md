# 只补跑 mspp16；原实验入口不变

用户于 2026-10-23 明确授权本资源修复。七路原评测各已完成 198 次任务尝试，其中 **195 条有效，只有 mspp16 的 solver seeds 0/1/2 失败**。已训练的 500 episode、开发选模和其他 65 个测试实例均无需重跑。原始失败状态、日志、结果与训练/评估代码保持原样。

新增资源例外**只用于 mspp16**：单 worker，单进程地址空间上限 **24 GiB**，容器内存至少 **32 GiB**。六份冻结 Actor 和原生 relpscost 全部按相同例外补跑，各三次，共 **21 次**。SCIP 搜索时限仍为 **120 秒**；进程 watchdog 为 300 秒，额外覆盖大实例读入、导入库和释放内存，不扩大搜索预算。其余 65 实例沿用原来的 8 GiB worker 结果。

隔离工程诊断已验证：8 GiB 上限下虚拟地址达到 7.999 GiB 后报内存错误；24 GiB 下同一原生任务有效完成，虚拟地址峰值约 11.36 GiB、实际驻留约 9.20 GiB。该诊断运行于 E5 开发容器，只用于定位故障，不能当作 Gold 节点的正式补跑结果。正式补跑仍需实际执行下列作业。

## GPU 和 CPU 要求

**这七个补跑作业不使用 GPU。** SCIP 与 Actor 推理均在 CPU，GPU 利用率为 0 是预期结果。此前六个正式训练的 PPO 更新才实际使用 CUDA。

要求实际 CPU 仍为 **Intel Xeon Gold 6148**，与已完成的 65 实例一致，求解器/绑定/软件 SHA 也必须一致。如果平台可以指定这类 CPU，请申请 **8 CPU / 32 GiB / 0 GPU / 1 小时**。若 CPU-only 无法指定节点，只能沿用 **8 CPU / 32 GiB / 1 V100 / 1 小时** 取得相同 CPU 的平台调度方式；这时显卡会空闲，并非算法计算需求。随机分配到 Silver/E5 时仍会拒绝，以免把硬件性能差异混进 120 秒 PDI 对照。

镜像沿用 `scip (user-duweiyue25/envd:11260740-c4da)`，Shell 选 bash，工作目录 `/home/duweiyue25/tgppo-transfer/research`，挂载 `/home/duweiyue25`。

## 启动七个独立 Custom 作业

准备命令已由 Codex 在开发容器执行；它只检查并冻结原 1365 条有效记录、21 条失败记录、模型 SHA 和资源修复清单，不运行求解。必要时可以再次执行，不覆盖旧清单：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh prepare
```

以下**每行对应一个 Custom 作业，共七个，可以并行**。每个作业内的三个种子串行执行，防止多个大内存 worker 争抢容器内存。

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry B0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry B0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry R0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry R0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry Rjoint 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry Rjoint 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh retry native
```

每作业完成后出现 `MSPP16_RECOVERED ... 3/3 valid`。若节点中断或某一任务失败，平台确认旧作业已停止后，用同一条新命令续跑；只重试缺失/失败的 mspp16 任务，保留所有尝试。不要用原 `post_job.sh evaluate` 重试本故障，因为原入口仍会应用 8 GiB 上限。不要删除旧缓存、锁或日志。

## 查询和合并结果

开发容器可执行，不运行求解：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh status
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/mspp16-recovery/run.sh report
```

新输出根目录：`tgppo/runs/server6-20261022/test/mspp16-recovery-20261023/`。

- `policy.json`：原 195×7 有效记录、21 条失败记录的来源及 SHA，冻结模型及新资源例外；不可变。
- `tasks/METHOD-sSEED/`（原生为 `native/`）：三条补跑结果、独立尝试、预检、完成记录。
- `reports/时间戳/`：`comparison.json`、`comparison.md`、`per-task.csv`，每次报告保留独立快照。
- 启动日志：`tgppo/runs/server6-20261022/custom-logs/mspp16-*.log`。

尚未补齐七路时，报告只比较所有方法共同有效的 **65 实例**，明确标注 `partial65`，不把它冒充完整 66 实例。最后一个补跑作业完成后会自动生成 **完整 66 实例、带 mspp16 内存例外**的报告，也可手动再次执行 `report`。即使部分模型已补齐，阶段性报告也不会单独把 mspp16 加进它们的均值。

原 `official66-best/` 不写入成功标记，平台旧失败作业不会被改成成功。新报告通过来源映射复用其 195 条有效结果；原失败和补跑成本另记。汇总采用实例等权，两个训练种子均报告，B0 为本轮 120 秒/Adam 参照，不能当作论文 3600 秒/AdamW 全规模复现。阶段 5 尚待实际补跑结果，阶段 6 未启动。
