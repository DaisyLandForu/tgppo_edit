# 脚本、配置与输入准备交付

日期：2026-11-02。用户已验收上一阶段单步 PPO 重放，并明确选择“离线评分头对照”。本次交付四个可并行 Custom 作业的入口、配置、固定输入和汇总程序。**四个真实评分头作业均未启动，也未代用户提交 Custom。**

## 已确认的准备结果

| 核验项 | 结果 |
|---|---|
| 合成小数据自测 | 13 项通过；包含缓存重放、非均匀策略 mask/排列、梯度累计、步数、重复提交、超时与平台中断 |
| Bash 入口语法 | `bash -n custom_job.sh` 通过 |
| 真实输入缓存 | 每种子 256 个更新状态 + 164 个监测状态；两种子均完成 |
| 缓存与原 Actor | 840 次原 Actor 前向捕获均通过原评分头回放 `torch.equal` 检查 |
| 来源行为概率重放 | 种子 11/22 的最大 ratio 偏差分别为 4.77e-7 / 2.38e-7，低于预定 1e-4 |
| 配对初始化 | 四组初始分布与均匀分布 TV 均为 0；初始行为 ratio 最大偏差 4.77e-7 |
| 原输入保护 | 66 项源码、checkpoint、轨迹及已验收材料引用的 SHA 均匹配 |
| 执行版本 | Python 环境中的 Torch 1.13.1+cu117、NumPy 1.24.4；执行设备 CPU，CUDA 未初始化 |
| 只读准备预算 | 预算账本记录 34.7218 秒，限额 600 秒，正常退出 0 |
| 参数与求解更新 | 真实模型准备阶段 optimizer 步数 0；新增 SCIP 任务 0 |
| 完整预检 | `preflight` 退出 0；四组配置、缓存、源数据和代码指纹核验通过 |
| 准备命令重复执行 | 核验后跳过，输出 `Already completed and verified; no new computation.` |
| 真实实验状态 | 四组均 `not_started`，提交步数 0，无活跃作业锁、无结果封存 |

自测记录见 [unit-tests.log](unit-tests.log)，数值检查边界见 [NUMERICS.md](NUMERICS.md)。自测只在临时目录使用合成小数据更新评分头；不能计作真实四组实验结果。

真实准备证据：[prepared/manifest.json](../../../runs/offline-head-probe-20261102/prepared/manifest.json)、[准备预算](../../../runs/offline-head-probe-20261102/preparation/budget.json)。每个缓存约 43.7 / 25.8 MiB；四作业共享只读缓存，不共享优化器或可写作业目录。

| 缓存 | SHA256 |
|---|---|
| seed-11.pt | `8a650a3e7050e6f4ed9d0af6e77fe2a14210cd031429a13a37d4c0e09f426c04` |
| seed-22.pt | `9c7cd03c9c716937416b792c6ec20bb51a74b38951892f6a5e08a71e206d27fd` |

本次独立复核的机器记录见 [verification.json](verification.json)，交付文件 SHA 清单见 [PACKAGE_MANIFEST.json](PACKAGE_MANIFEST.json)。运行时另以 prepared manifest 固定 12 个代码/配置文件的指纹；不要修改这些文件后混用旧缓存。

## 现在可以提交什么

按 [CUSTOM_JOBS.md](CUSTOM_JOBS.md) 各提交一条命令，组成 `original/linear × s11/s22` 四个独立作业。每作业 1 CPU、8 GiB 内存、0 GPU，平台时限建议 15 分钟；程序硬限制 600 秒和 20 个评分头 Adam 步。真实 Custom 容器执行和实际完成耗时尚未验证；本机缓存准备、预检和合成自测不替代目标环境验证。

这四组仅比较冻结表示上的条件学习响应。两组都将最终标量读出归零并使用 fresh Adam；原头中间层保留 ep100 权重，因此不是原 B0 的原样续训。原头 241,315 参数、线性头 257 参数；头的容量、门控及深度不同，不能把差异唯一归因于 LN2。固定 batch 拟合或 TV 变大，均不能直接证明候选排序正确或求解性能提高。

四作业全部正常退出 0 后执行 `summarize`。提前触发 KL 边界、超时、异常或材料不完整时，保留输出交回审阅；程序拒绝自动清零重跑和不完整汇总。提交结果后停止，等待本阶段验收。
