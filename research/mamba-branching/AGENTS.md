# Mamba-Branching 复现约定

- 用户于 2026-10-27 明确要求独立 Conda 环境，先按原论文完成 MILP-S 闭环。
- Python 环境为 `/home/duweiyue25/conda/envs/mamba-branching`；启动统一通过 `scripts/runtime.sh`。不修改或借用 TGPPO 环境里的二进制绑定。
- 用户已接入 GPU；2026-10-27 实测开发容器 1 V100-PCIE-32GB、约 8 CPU、32 GiB RAM；用户 Custom 节点为 V100-SXM2-32GB、同样约 8 CPU / 32 GiB RAM。两节点合成预检均通过。大规模采样放到 CPU Custom 作业；当前只做有限诊断，不自动启动正式矩阵。
- 起步资源：调试 1 V100 / 8 CPU / 64 GiB；采样合计 16 CPU / 64 GiB；先完成单种子闭环再并行。
- 原文和源码固定在 `sources/`、`upstream/`。工作副本 `code/`，修改保存补丁及来源哈希；历史 TGPPO 目录只读。
- 论文设置见 `configs/paper-milp-s.json`，实例清单见 `configs/milp-s-instances.json`。训练专家是 relpscost；采样不设 cutoff；验证轨迹使用 solver seed 4；8 个测试实例不用于选模。
- SCIP 必须是 8.0.4；特征 25/8/53；正式训练 100 个状态的序列，评估 24 步历史加当前状态。上游 `max_seq_len` 计数交替的候选组和动作组：对应参数为 49（2×24+1），不能误传 24。
- 论文使用 Mamba-1；不要因仓库导入 Mamba2 就改成 Mamba-2。V100 的真实 CUDA 前向/反向及完整批显存均需实际核实。
- `gpu-preflight` 只是合成核函数/参数更新/权重恢复检查，不能称为完整 MILP 闭环、正式续跑验证或论文复现成功。
- 上游脚本存在 padding、静默跳过样本和异常处理问题，见 `docs/SOURCE_AUDIT.md`。正式训练前必须明确并验证处理方案，不能把未修的默认脚本当成已验证训练入口。
- `code/paper_training.py` 是已验证的损失适配层：CL 仅用有效候选；IL 在 Mamba 前打包真实候选和历史动作。IL 按论文每轨迹求和、轨迹间均值，上游按状态均值的差异须保留说明。`gpu-training-pilot` 仍只是诊断；正式入口为 `scripts/formal_*.py`，指南 `docs/CUSTOM_JOBS.md`。
- CUDA 固定数据恢复的 RNG、数据游标及下一步损失一致；自定义核更新后有约 1e-8 参数差异，不能声称逐位确定性。
- 记录真实进程、退出状态、配置、代码与环境哈希；保留失败结果。没有 GPU 证据就报告待验证，不捏造运行结果。
- 用户已要求通过并行 Custom 完成正式单种子闭环。正式 RUN 已初始化在 `runs/milp-s-seed0-20261027`；本会话未替用户提交正式作业。8 个 CPU 采样分片→完整性校验→1 GPU 顺序 CL/IL→4 GPU 评估分片；不要把同一种子的两阶段训练并发写到同一 RUN。
- 训练逻辑 batch32，IL 微批1累积后统一按 max_norm=1 裁剪。50/50 epoch 来自上游默认值；只用 seed4 验证 loss 选模，严格并列取早，冻结 `training/selected.json` 后才允许 8 测试实例求解。全部任务缺一不可，失败不静默删除。
- `runs/entry-check-20261027-final/verification.json` 验证了诊断数据上的跨进程 CL/IL 恢复、连续运行数值比较、梯度累积、选模、求解及失败边界；它不是正式 475 条/50+50 epoch 已完成的证据。
- 2026-10-28 用户授权直接做RAM缓存/批量组装/有序预取/IL微批加速。新入口`train_fast.sh`、说明`docs/FAST_TRAINING.md`；不改原冻结源码或RUN/config.json。实际执行覆盖及新增源码哈希另存`training/fast/execution.json`并绑定checkpoint；IL微批由Custom启动时验证后选择8/4/2/1，逻辑batch32不变。旧CL已受控停于更新102875；预加载全475条、数据与CPU迁移恢复验证已通过，CPU恢复使用GRU替代Mamba，不能声称真实Mamba CUDA门禁已跑过。迁入后用同一快入口续跑。

- 2026-10-29核查：正式CL50/IL50已完成（458400/4900更新），CL最佳44、IL最佳50，selected已冻结；A100微批8探针通过。当前V100完整复算seed4验证loss/准确率与A100记录完全一致（0.952592每状态NLL、73.70%准确率）。结果见`runs/milp-s-seed0-20261027/analysis/training-review-20261029/`。正式测试尚未启动（0/40对），下一步用户提交4个评估Custom，不因训练完成就声称论文性能复现成功。
- 2026-10-29正式评测完成：40对/80次，4分片passed，120结果文件SHA与403项检查通过。报告`runs/milp-s-seed0-20261027/analysis/evaluation-review-20261029/`。Mamba/relpscost几何均值比：普通节点3.030、公平节点1.921、耗时1.831、原生PDI1.261；8实例均平均更慢，7实例公平节点更多。80次均在已知最优cutoff下无incumbent且搜索结束，不能说原MILP无解或实际找到了最优解。单种子主方法/relpscost闭环已完成；神经基线排名和多训练种子未验证。冻结模型不因测试结果更换；没有启动追加实验。
