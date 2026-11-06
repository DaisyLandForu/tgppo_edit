# Crater Custom 作业操作单（2026-10-22）

用户已确认：当前无 GPU 容器用于开发；正式训练另行申请 Crater Custom 作业。作业容器共享 `/home/duweiyue25`，可访问本项目与 Conda。当前镜像为 `scip (user-duweiyue25/envd:11260740-c4da)`，Custom 作业也选择此镜像。GPU 选择策略和作业最长时限仍需在创建作业时核对。这里没有 Slurm 客户端，不填写虚构的分区、账户或作业 ID。

固定项目根：`/home/duweiyue25/tgppo-transfer/research`。
Python：`/home/duweiyue25/conda/envs/tgppo/bin/python`。
所有新输出：`tgppo/runs/server6-20261022/`。

从空环境重建时先安装 `tgppo/setup/requirements-core.txt` 和 `torch==1.13.1+cu117`（使用 PyTorch cu117 索引），再以 `TGPPO_PYTHON` 指定上述 Python 执行 `bash migration/2026-10-22/server-run/build_local.sh`。构建脚本保存每次尝试，用用户目录中的 Ubuntu bliss 依赖，不需 sudo；Cython 在源码副本中生成 C，仅将新 `.so` 放到 stage4 绑定的 `src`。求解器已成功、只需重建绑定时可传 `--binding-only`。共享环境准备完成后，六个 GPU 作业直接使用它，不各自重复安装。

本轮仍是 B0/R0/Rjoint × 11/22，各 500 个完整 episode，每 100 保存，4 个单线程 SCIP worker，CUDA FP32 PPO、Adam、120 秒任务。`protocol.json` 的旧授权文字记录的是打包时状态；当前用户已明确授权此前置检查通过后的六作业，不改冻结数值。阶段 5 未完成，阶段 6 未启动。

界面统一填写：Shell 为 bash、工作目录为项目根、用户空间挂载点为 `/home/duweiyue25`，保留普通用户。镜像选择 `scip (user-duweiyue25/envd:11260740-c4da)`，已知该开发容器为 Ubuntu 22.04 x86_64。Conda、重建的 solver 和代码都在共享用户目录，无需先保存新镜像；若保存镜像，新作业仍使用这些共享路径。GPU 作业需使用平台支持 GPU 驱动注入的运行时。Torch 使用环境内 CUDA 11.7 运行库，不使用源机 `.so`；2026-10-22 的目标准备作业已在 Tesla V100-SXM2-32GB 上通过实际 CUDA 更新、恢复及三组吞吐检查。截图中的 Python 3.11/3.10/3.9 公共镜像不作为此次启动首选，冻结入口使用绝对路径的 Python 3.8 环境。

| 用途 | CPU | 内存 GiB | GPU | 数量 | 建议名称 |
|---|---:|---:|---:|---:|---|
| 目标基线、三组桥接、20 实例吞吐 | 8 | 32 | 1 V100 | 1 | tgppo-prepare |
| 正式训练 | 8 | 32 | 1 V100 | 6 | tgppo-B0-s11 等 |
| 完成后的开发集选模 | 4 | 16 | 0 | 6 | tgppo-dev-B0-s11 等 |

每个训练作业按原模板申请最长可用时限（模板 24 小时），核对 Crater 的实际限制。不要因为资源可申请而扩大本轮 worker 数、样本数或模型。CPU 绑核可按平台资源策略开启，预检同时检查 cgroup 配额；基线、训练、评估须同类 CPU、同一求解器和绑定哈希，否则入口拒绝复用。截图仅显示 GPU 数量，需在资源选择处确认 V100；预检也会拒绝其他型号。

先创建 **一个**准备作业，启动命令填：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh prepare
```

该作业在前台依次运行 CUDA/资源预检 → 小型工程检查（含目标 CUDA 检查）→ 25×3 原生 relpscost 基线 → B0/R0/Rjoint 完整模型 GPU 桥接 → 每组覆盖全部 20 个训练实例的独立成本试跑。试跑不计入正式六作业的 500 episode，不参加选模或测试。桥接包含真实中断后重做完整更新、Actor/Critic/两套优化器/CUDA RNG 与不中断参考结果的比对；采样记录必须证明 worker 未初始化 CUDA。任何一项失败都会停止，保留日志，不自动降低协议。

观察 `custom-logs/prepare-*.log`，以及：

- `normalization/baselines.json`：75 条有效原生任务和目标软硬件指纹。
- `gpu-bridge/verification.json`：`passed=true`，完整 CUDA 更新与恢复证据。
- `throughput/verification.json`：`passed=true`，三组 20 实例覆盖、更新耗时、显存分配峰值、worker RSS、100/500 episode 墙钟估算。估算不含未来排队，初始策略之后决策数可能变化；实际峰值还应结合 Crater GPU/容器监控。

2026-10-23 00:04 核查：上述三项证据均通过，`check_preparation.py` 返回 `ready=true`；三组各 20/20 个 episode、5 轮完整更新、最终 checkpoint 哈希均已验证。首次准备作业在所有计算结束后，因运行期间原地修改共享启动脚本，Bash 从旧偏移续读新文件，出现 `line 44: e: command not found` 并返回 127。原失败状态及日志保留，不改成正常退出；计算证据仍有效，**无需重跑准备，可直接按下文创建六个正式训练作业**。修复入口为开始工作前一次解析完整代码块，并在块内明确退出；新增覆盖运行期间修改脚本和真实阶段失败的检查，10/10 通过。诊断、复现及哈希台账见 `tgppo/runs/server6-20261022/repairs/launcher-tail-20261022T160236Z/REPORT.md`。此修复没有改动训练运行时代码或协议。

准备作业中断后，用同一命令创建新的 Custom 作业。原生任务按不可变身份复用有效结果；桥接每次保留独立目录；吞吐试跑通过 `latest.pt` 显式恢复。若 CPU 型号、二进制、协议发生变化，先检查错误和证据，不删除旧目录解锁。

三项前置证据通过后，创建六个独立 Custom 作业，每个启动命令分别填以下一行。每个作业均为 8 CPU / 32 GiB / 1 V100：

可先在开发容器执行 `/home/duweiyue25/conda/envs/tgppo/bin/python migration/2026-10-22/server-run/check_preparation.py` 查看三个阶段：只有 `ready=true` 才提交训练。当前采用先完成准备、再申请训练的顺序，不让训练作业占着 GPU 等待。提前启动时入口会明确列出 `pending_files` 并以 75 退出，不再抛出缺少 baselines.json 的文件堆栈；失败或指纹不一致仍会拒绝。该检查只读共享证据，真正训练入口仍会核对当前协议、目标 CPU/软件和 V100，不能靠占位文件绕过验证。更换 CPU 型号或软件栈会触发基线不匹配，不能直接复用旧机器的结果。

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train B0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train B0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train R0 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train R0 22
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train Rjoint 11
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh train Rjoint 22
```

这六行应分别放入六个作业，不能全部放进一个启动命令框。记录 Crater 分配的六个作业 ID 与资源/实际节点类别；当前开发容器没有提交 API 或这些 ID。训练入口使用 `exec` 托管前台进程，日志在共享盘；Codex 对话退出不会充当进程管理器。

中断/到时后，在平台确认旧作业已终止，再用对应相同命令创建续跑作业。命令显式传 `--resume`；不删 `latest.pt`、`attempts/`、`collection/` 或心跳。PPO 半轮中断后使用更新前完整轨迹和随机状态重做，保留失败成本。双启动由内核文件锁拒绝；不要手工删除锁文件。SIGKILL 可能使末段成本没有落盘，成本须再结合平台实际计费/资源记录。

开发容器随时可查看已提交检查点进度：

```bash
/home/duweiyue25/conda/envs/tgppo/bin/python migration/2026-10-22/server-run/status.py
```

状态程序读取 `latest.pt` 和稳定里程碑。只有完整 500、五个里程碑和最终哈希匹配的 `summary.json` 才报告完成；是否正在运行仍应查 Crater。心跳与旧 running 标志不能证明完成。

六作业完成后，创建 CPU Custom 作业逐种子选模，命令例如（其他五个仅替换组名/种子）：

```bash
bash /home/duweiyue25/tgppo-transfer/research/migration/2026-10-22/server-run/custom_job.sh select B0 11
```

同一 5 开发实例×3 求解器种子，在 100/200/300/400/500 中按实例等权 120 秒 PDI 最低选择，相同分数取较早者。保留六份 `selection.json`、`actor_best.pt`、`actor_final.pt`；两个训练种子全部报告。锁定这六份选模结果后，按主 README 的 Actor-only 评测入口运行 66 个保留实例及有可信 cutoff 的独立兼容集，并同硬件/设置运行 relpscost。当前没有额外兼容集清单；不能凭空报告这部分结果。

完整成本包含准备、失败/重试、六个训练、450 次开发求解、1188 次 best-Actor 测试求解和 198 次原生测试求解。若另评 final，另外记录其 1188 次，不能隐入训练成本。结果汇报注明 B0 为本轮 120 秒/Adam/四任务更新参照；缺少 Rprod，不能单独宣称联合关联收益。完成阶段 5 后等待明确验收。
