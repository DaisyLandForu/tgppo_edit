# TGPPO 项目工作约定

## 2026-11-18 B0原因与复现归因补充

- 用户追问B0弱响应是否论文或复现问题。只读重核论文v1及作者固定commit4f90，在线Actor/Critic/modules blob与当前trainmode三文件完全相同（11f696/fa682c/1d26fe）。当前新B0已train更新，不能继续把旧eval调整当作新330弱响应的充分原因。主超参数/25×5/3600/目标500按正文对齐，但标准分episode GAE、短轨迹更新、微批/随机流/设备是明确调整；公开train.py多轨迹合并，与正文逐episode不同。不能把“论文对齐”简称等同完整作者流程/性能复现，不能把主动改动归为作者未公开或用户操作失误。
- 多阶段候选差异衰减有历史逐层/局部干预/梯度证据，BiMatching个体标量通路、TreeGate窄压缩/LN2机制继承公开源码；末层Linear没有LN1。历史LN2/8→1对照未恢复稳定收益，仍非单层唯一根因。当前330探针不是重做旧逐层定位；模型参数改变/clip0均不能单独判定学习成功失败。尚未拿作者正式权重/最终运行记录，不能宣布论文必然错误或以本地改进直接等同胜过作者正式模型。
- 新澄清docs/methods/tgppo_seed11_training_20261118/CAUSE_AND_REPRODUCTION.md及AUTHOR_SOURCE_RECHECK.json；原曲线/报告不覆盖。建议仅为未执行提案：既有轨迹上作者原Agent与当前Agent局部对照、新固定300层级复查，零新SCIP；不能把旧修正版Agent PPO重放当作者原Agent核验。未训练/求解/更新权重/联系作者/改变作业，现有训练继续。

## 2026-11-18 seed11三模型训练曲线核查

- 用户问seed11曲线收敛和训练结果；只读检查新device-resume/gpu三个实际latest，13:24北京时间快照新B0 330、MLP458、BC462，锁活动、仍采样，V100目标检查passed。模型有限/内存空，318/440/443有效PPO episode与版本计数及完整曲线吻合；12/18/19为零决策episode，非更新失败。不要盲用11/16旧停止状态或旧GPU父分支。
- 按祖先checkpoint已提交游标接续：MLP旧GPU1—113→CPU114—284→GPU285—458，BC旧GPU1—137→CPU138—284→GPU285—462，新B0旧1—164→新165—330。原始日志逐项哈希/任务顺序/有限数核对，曲线与CSV/ANALYSIS在docs/methods/tgppo_seed11_training_20261118。用独立rl4scip环境Matplotlib生成并查看PNG，另PDF/SVG，未改TGPPO依赖/冻结代码/权重，无新训练/求解/GPU/作业变更。
- MLP/BC最近50episode非空更新clip均值1.23%/1.20%，Critic约20.94/21.17仍任务波动；有策略更新，不能宣布实际求解收益已经收敛。B0318/318 clip=0；318次部署探针TV到均匀最大2.44e-7、更新前后最大1.22e-7，全<1e-6。探针每次最多第一微批8训练状态且按候选长度排序，不能推广全状态或断言作者网络必然缺陷；曲线平稳不等于Actor学好。
- 最新权重尚无独立评测。已有200/66/solver0 seed11 MLP PDI29459.59/证明33，BC26222.10/41（−10.99%、净+8），B011当时缺200；B011已完整66评测只到100。别拿B022冒充同种子或把异硬件100→200当学习曲线。保留500固定终点，不临时调参/提前停或测试选模。近50轮外推轻量约7/6h、新B0约40h仅估计。交付报告后等用户下一步安排，原训练持续。

## 2026-11-16 六训练最新分支迁移及CPU更新线程加速

- 用户要求恢复新B0/MLP/BC六作业，seed1=11各V100，seed2=22纯CPU，问4/8核与worker/锁并行。只准备入口和离线检查，Custom由用户手动提交；未启动真实训练/GPU/SCIP求解。当前六源均已停止，新B0 164/218、CPU MLP284/172、CPU BC284/271；MLP11/BC11有285 pending1557/849条。不要用旧GPU轻量父分支或旧CPU B022的12进度。
- 新独立包tgppo/experiments/paper-device-resume-20261116，六新副本在同名RUN/{gpu,cpu}/training；commands.txt一行一个作业。源/预算与完整payload逐字段/张量校验、pending清理隔离、互斥锁、重复导入不覆盖通过。两协议CPU4线程合成桥接/恢复及成功异常线程恢复通过；原49/53/6来源校验不变，权重/100/200评测不改。未来新进度在新RUN，不再盲用旧recovery目录统计。
- 实际本地4.01核E5-2650，用现存pending大候选731/1039/1145做Actor+重Critic前后向微基准，约60.75秒、0优化/求解；1→4线程3.39/2.67/3.47倍，只是网络计算部分，不代表整轮/整场训练或8核收益。CPU更新wrap Agent.learn设置线程，避开seed_all每次重设1；采样仍单线程、逐episode后3epoch不变。CPU建议4核4线程/GPU4核1线程，8线程仅可选待测，不自动更改workers或并行采episode。
- 旧模式MLP无Dropout/BN，保留其aligned算法配置，新B0保留trainmode；设备/线程迁移不保证bitwise。保留AdamW/游标/已有RNG/账本，no-total-wall-limit，3600/4200/500不变。V100目标桥接待Custom执行，仍有2h闲置回收/ECC风险，8核/锁不能解除，不伪造GPU利用率。报告docs/methods/tgppo_device_resume_20261116/REPORT.md，交付后等待用户提交作业反馈。

## 2026-11-16 固定200完整66结果复核交付

- 用户告知跑完，已只读核对330/330有效0失败、66完整五条件同cohort配对，4作业于11/14 01:20北京时间完成，各11.59／11.82／9.68／9.76小时。原始JSON重算全部汇总一致，四模型／完整200／原始CSV报告SHA前后不变，冻结包verify通过；无新增训练／SCIP／GPU或作业变更。六训练实际均stopped且锁空闲，新B0 164／218、MLP284／172、BC284／271，仍缺新B011／MLP22的200，无500。
- 主表PDI／证明数：原生14345.315／60，新B02233878.757／25，MLP1129459.592／33，BC1126222.103／41，BC2226469.147／40。BC_seed11 PDI−10.99%、证明净＋8（增9失1）、optimize−13.50%、44胜22负；combined_seed22 PDI−21.87%、证明＋15且无丢失、optimize−17.82%、中点50胜16负／测量界分开48胜16负2重叠。Actor55.79→3.06ms观测18.24倍是输入转换／网络／采样决策路径，不是纯forward／求解倍数。组合全66节点SGM＋10.95%，不要藏此负向；作者easy33节点／hard32 PDI SGM分别−5.40%／−34.19%。
- BC最大贡献app1-2占净收益52.12%，去掉仍PDI−6.17%；组合最大ns1830653占24.37%，去掉仍−17.75%。探索性未预注册实例级bootstrap10000（20261116）相对PDI95%：BC[−23.12,+3.50]%、组合[−38.35,−5.88]%；不是测量界，也不代替新增随机种子，BC均值统计稳健性尚不足。BC两种子端点相近不能证明增量跨种子，因为MLP22缺失；纯结构也缺配对。学习组均整体落后原生，原生PDI相比100轮本身＋31.09%，跨轮不同主机／CPU不当学习曲线。
- 新交付docs/methods/tgppo_paper200_test66_20261113/RESULTS_REVIEW、更新README、review/analyze_results.py及REVIEW.json／instance_contributions.csv／TRAINING_STATUS_20261116.json，自动原始REPORT/CSV不改。report/TGPPO_200episode评测摘要_20261116.md及索引新增。保持200冻结，先固定8例×solver1/2×五条件80项CPU确认为后续提案，缺36／28episode补200消融优先于直接六组扩500；没有实际启动下一阶段，交付后等待指示。

## 2026-11-13 用户确认固定200现有四份评测

- 用户询问200进度／论文66范围，表示将手动停六训练并申请并行Custom；随后明确选“先评已有4份200权重＋relpscost”。12:24快照新B0两seed163／206，CPU续训MLP284／171，BC284／266，六锁均活动；仅新B022、MLP11、BC11／22有200。旧B0仍137／57没有200，relpscost不训练，不拿100冒充200。未停止／提交任何作业。
- 独立包tgppo/experiments/paper200-test66-20261113与RUN、docs/methods/tgppo_paper200_test66_20261113已准备。四份actor-0200及完整episode-0200实际读取，逐张量相等、配置／进度／seed／BC与SHA核验后只读冻结。MLP来自recovery/cpu-aligned，B0来自trainmode CUDA，迁移来源写plan，不偷偷用旧GPU父分支。新B011与MLP22缺200，只有BC_seed11和combined_seed22可同种子归因，禁止MLP11对B022冒充纯结构消融／齐全跨种子结论。
- 新330＝全66×四模型＋新relpscost、solver0、3600／4200，4CPU／32GiB／0GPU四作业85／85／85／75；CPU型号不限，同实例五方法同cohort。mspp16单worker24GiB，其余四worker各6GiB；显式resume，不自动追加／重试。原生在新作业同机重测，不复用旧100绝对时间。13项合成覆盖／恢复／种子流／统计检查和实际四Actor CPU/fork逐位检查通过，无CUDA／Critic／SCIP optimize。第一次prepare在冻结完成后末尾verify触本地180秒外限exit143；独立无Torch全输入verify随后11.54秒通过，记录于PREPARATION_REPORT，不影响评测3600。原49／53／6／6及100包12冻结文件未变。
- 论文v1 §5.1表2确为25训练＋66完整test，非完整MIPLIB；最终500、每episode3轮AdamW、solver0—4，测试每策略名义66×5＝330；我们本轮330是四权重＋原生各66×solver0，覆盖不可混淆。论文easy节点SGM shift100、hard PDI SGM shift0；40核CPU、SCIP6、关primal heuristics＋已知最优cutoff。正文33/33而公开XLSX33/32缺opm2-z7-s2，附录个别bell4/n2seq24与正文不符，不当成独立新数据集。新汇总另固定作者33/32/未分组1，保留全66主表。PAPER_PROTOCOL_AND_SCOPE含原文链接；100均值不能直接同新硬件200或作者hard SGM比倍率。最终固定500门槛不变，无测试选模。

## 2026-11-13 MLP方法讲解与模型图交付

- 用户下午需要详细解释MLP／MLP＋BC相对原TGPPO的改进、依据与收益机制，并要求参考高质量论文重画更美观网络图。已读取实际light_actor、Actor/BiMatching/TreeGate、pretrain、expert_data、PPO配置与完整66结果；浏览并实际查看TGPPO Fig.1、Set Transformer Fig.1、Gasse2019 Fig.2，依据本地源码原创重绘。
- 新材料report/mlp_methods_20261113：12页原生可编辑PPT（演讲者备注）、同版矢量PDF、离线HTML、约1万字符方法详解与问答、五张独立主图各SVG／PDF／PNG、数据与来源哈希。双路LN25/61→64→64、共享头128→64→32→1，总24,493；旧4,738,895。MLP删除整重Actor，保留树条件化／候选等变；它不是完整DeepSets，也无CL、历史模块或持续专家损失。BC是已有relpscost动作，不是strong-branching；12实例1468拟合／3实例288内部留出、1000步完成选100，Actor初始化后fresh PPO AdamW，Critic保留。
- 主结果优先新B0参照：MLP PDI−19.35%、证明＋7.58pp；组合PDI−17.12%、证明＋18.18pp。BC对MLP证明净＋7（增8失1），PDI＋2.76%。时间56.64→2.77ms比值20.43是Actor决策路径（输入／检查／网络／采样log_prob），不是纯forward或端到端求解倍数；本轮MLP optimize只−2.99%、节点几何均值更高。已修正当前结果README措辞，旧原始CSV不改。
- 明确3条件顺序对照不是完整结构×BC 2×2；BC额外预训练预算、历史专家成本不能当零；没有同协议BC-only，PPO独立增益未确认。只固定100／训练11／solver0／66受控cutoff，历史测试暴露，不冒充原作者权重／500／稳定泛化。结构内部细节因果未分离，归因解释标为假设；无虚构gap曲线。原生仍整体更强。
- 报告环境独立/tmp venv，不改TGPPO Conda；字体OFL与子集随材料。12页PDF实际渲染并查看总预览／网络／训练主图；PPT重开、ZIP、shape界限、字体覆盖、notes、链接、17源SHA检查通过；未在Office实际渲染，审计诚实保留此边界。无Torch导入／训练／GPU／SCIP／作业提交；原训练代码权重与结果保留。report/README新增入口；新阶段交付后等待用户反馈。

## 2026-11-13 新B0-seed11 ECC再次复发与设备留证入口

- 用户引用原resource seed11命令（参数正确），新日志20261113T005208-KTPVCI：Pod08f04／Gold6148／V100-SXM2，08:52:08→08:53:21启动target_checks时ECC，未生产latest加载。前一771d5 attempt从152推进162后08:50:03 SIGTERM，距162提交约119.76分钟，时间吻合2h闲置但未确认信号发送者；不可把两事件混为ECC或只凭型号认定同坏卡。
- CPU实际只读核验latest162，下次163、pending无、锁空闲，有限Actor/Critic/AdamW、配置SHA／四随机流／提交摘要通过，源SHA ec6cab2d0dd0e559e37b772c31790c14985202260d1476bb98c3e117832e880c不变；不能退回旧152或旧CPU副本。原49／53／6／6源码哈希未变。新报告／EVIDENCE／CHECKPOINT_AUDIT／CPU_CHECKS／RESUME_COMMAND位于docs/diagnostics/tgppo_20261113_b0_ecc_recurrence。
- 独立入口tgppo/experiments/paper-gpu-health-resume-20261113/custom_job.sh已完成：记录GPU UUID／PCI／驱动／ECC、独立标量CUDA探针后原参数exec原resource；失败不训练，历史ECC累计非零不自动拒绝。CPU/status旁路，原目标bridge保留。20秒query／180秒探针只约束启动检查，不改SCIP3600／500／GAE／train／AdamW，不修复硬件、不解除平台闲置回收。6替身控制流检查通过，无真实本地GPU／SCIP／训练／作业变更；真实GPU待用户Custom运行。建议平台映射Pod、隔离修复故障卡后从163继续，不关闭ECC／重置共享卡／忙等伪造利用率。

## 2026-11-12 新B0两种子启动ECC故障

- 用户提供seed11原resource续训命令和130647-6pyKbR异常。共享盘另找到seed22的130737-NGtgWm，均在target_checks临时Agent向CUDA搬运时返回uncorrectable ECC error，尚未生产latest加载／本次真实PPO。北京时间21:07:32.869／21:07:41.521停止，均Gold6148／V100-SXM2-32GB，Pod ac9dc／35cd0；没有GPU UUID／物理节点名，不推断同一坏卡或永久损坏。RAM门槛通过，OOM0，非此前SIGTERM，也不以64GiB或缩batch修ECC。
- CPU只读加载两latest，已提交152／156，下次153／157，无pending，配置SHA／有限模型与AdamW／四随机流／Actor提交摘要均通过，源SHA不变、mtime早于失败attempt、锁空闲。最新SHA a83a8393b9c5ab3031d480121d216196c6d50ad34c4851571be35cab2912a1a0／e93edecf8d7eec51f7559cf65392fd7867806a22eaa74f78c33a99ac13726a20。旧49／新53／recovery6／resource6冻结文件校验通过，无训练／SCIP／CUDA／权重改写。
- 新交付docs/diagnostics/tgppo_20261112_b0_ecc下REPORT、EVIDENCE、CHECKPOINT_AUDIT、RESUME_COMMANDS。建议换健康GPU节点或管理员恢复原卡后，原4CPU／48GiB／1GPU resource命令继续两seed，不清零训练、不需改算法。已核实NVIDIA官方ECC214定义及Volta Xid48重置／重启机制，但没有本次内核Xid，不能声称实测48／断电根因；不执行共享GPU reset、不关闭ECC、不伪造利用率，不给CUDA_LAUNCH_BLOCKING当修复。

## 2026-11-12 固定100／全66评测完成

- 用户询问进度。只读核实既有终端序列已于北京时间14:59:57完成；四分片85／85／85／75均complete、锁空闲，无残留评测worker，330有效／0失败、66完整五方法配对、同一cohort。总经过墙时42.903小时；重新校验冻结包／模型／实例SHA，并从原始记录复算平均PDI及证明数，与自动汇总一致。新核验快照`docs/methods/tgppo_paper100_test66_20261110/STATUS_SNAPSHOT_20261112.json`，README已更新最终主表，旧11/11进度快照保留。
- 平均PDI／证明数：relpscost10942.995／60，旧B027395.366／31，新B027399.939／31，MLP22098.596／36，MLP＋BC22709.286／43；每方法66。MLP对旧B0 PDI−19.33%、证明＋7.58pp，Actor56.58→2.77ms约20.4倍（不是求解倍数）；BC对MLP证明＋10.61pp但算术平均PDI＋2.76%，不能概称所有指标改善。新旧B0整体接近，不能以此判未学习／唯一网络根因；学习组均整体落后原生。
- 仍为训练seed11／solver0／固定100／3600中途评测，单种子与历史测试暴露边界保留，不用于挑检查点，不冒充最终500或稳定泛化。所有66四模型均调用Actor。没有新增SCIP／训练／Custom、没有改冻结脚本或权重；完成交付后等待后续指示。

## 2026-11-11 B0-s11 Job YAML逐项核对与Crater源码线索

- 用户附件为旧失败Job sg-duweiyue25-261111-2574a（crater-workspace），匹配10:15:47→11:53:28 attempt；容器command与task-template注解及既有命令逐字一致。4CPU/48GiB/1V100资源正确，49153Mi limit与内存监测字节完全一致，峰13.09GiB/OOM0；无activeDeadlineSeconds/启动或存活探针。11:17:21才进入122采样，2166.55秒后主进程SIGTERM，未到3600/4200；latest仍121。maxRetry为次数、TTL604800为终态后7天清理，不能当运行时限。
- 当前可读/crater-start.sh无timeout，只做用户准备并sudo执行；未取得旧Custom ConfigMap历史，不能说逐字节就是旧容器脚本。可读启动文件无timeout，部分sudoers PermissionDenied，未提权/重试读取集群token。Job YAML无status，无法从中取Pod termination reason。
- 已用GitHub插件查Crater官方固定提交367718b388e0193d8e633f3710622810a9267ab3，cleaner/low_gpu_usage.go确有低GPU自动删除VCJob机制，删除窗口timeRange+waitTime；公开values默认90+30分钟，AlertEnabled仅邮件。monitor/query.go用DCGM窗口最大值与offset指标存在性。部署版本/实际参数/命中审计未知，公开默认120与本次97.7分钟不匹配，必须保留为源码机制/候选，不能宣判本次就是GPU闲置，更不能统一解释CPU故障。
- 新Job abd82已在13:44:16启动且执行锁活动，读取时normalization、next121；保持运行，不把新旧attempt混淆或重复提交。新证据B0_S11_JOB_2574a.yaml、B0_S11_JOB_CONFIG_AUDIT.json、B0_S11_CONFIG_REVIEW.md均在docs/diagnostics/tgppo_20261111_restart_audit。无新增训练/SCIP/GPU/冻结源码修改；用户已说平台无终止原因字段，不再重复索要前端同样字段。

## 2026-11-11 上午重提后四任务再次外部SIGTERM

- 用户截图大量新失败：新B0-s11/s22及CPU MLP-s22、BC-s22均已经启动，并非MLP-s11的LXCFS启动前故障。最新日志均主进程外部SIGTERM15：北京时间11:53:28／11:48:29／11:20:31／11:19:38；运行约98/92/64/62分钟，四监测OOM/oom_kill均0、采样容器峰值10.88—14.11GiB。用户明确平台无退出码／原因，要求从日志查；不要再次索要同样前端字段或编造唯一平台原因。
- 已追源码：实际unlimited_budget无计时器，4200秒watchdog仅杀子进程；失败采样距启动2166.55/3597.08/96.65秒均未达4200，MLP22在纯PPO pending重放79/81仍被终止。没有依据为这批故障缩短SCIP3600、加内存或统一归因2hGPU闲置。共享日志未记信号发送者；无集群连接器/kubectl/kubeconfig，mounted service-account token读取PermissionError，未发API请求／提权；平台后台事件仍是根因缺口。
- 新B022推进121→126，实际latest只读核验通过（参数有限、configSHA/Actor摘要/AdamW/四随机流），另三失败latest SHA与今早已核验值相同；MLP22仍已提交138、pending139。新B011仍121，BC22仍142。MLP11无新attempt、仍163，等待挂载恢复；BC11持锁更新182、14/18，已提交181，不重复提交或停止。终端评测13:04快照127有效、125项/25实例完整配对，四worker继续，不发表部分总体排名。
- 新说明`docs/diagnostics/tgppo_20261111_restart_audit/BATCH_RESTART_FAILURES.md`及BATCH_FAILURE_EVIDENCE.json、B0_S22_EP126_CHECK.json；四失败条件原续训命令另存BATCH_RESUME_COMMANDS.txt（不包含运行中BC11）。建议由平台后台按精确Pod ID／终止时间定位外部SIGTERM并换健康节点后再恢复；没有改冻结脚本、参数、算法、预算历史，没有实际提交／取消／新增训练或SCIP。

## 2026-11-11 六训练SIGTERM与100评测进度核查

- 后续用户截图MLP11作业sg-duweiyue25-261111-300b7卡“启动中”，kubelet FailedMount：宿主机/var/lib/lxc/lxcfs/proc/{cpuinfo,stat,uptime,swaps}不是File。与训练脚本无关，可能断电后LXCFS/FUSE未恢复，唯一根因需管理员节点检查。只读确认CPU MLP11仍163、latest未变、执行锁空闲、无该作业新训练日志。建议取消该卡启动作业后换健康CPU节点或请平台修复，原4CPU/64GiB/0GPU命令不改；不重提/取消其他作业，不改模型/内存阈值或手造空/proc文件。证据说明`docs/diagnostics/tgppo_20261111_restart_audit/MLP11_FAILED_MOUNT.md`。

- 用户称凌晨集群断电、六训练失败并要求续训命令和100评测检查。六最新匹配日志均外部SIGTERM15：新B0两seed已提交121/121；CPU recovery MLP11/22为163/138；BC11/22为181/142。停止北京时间11/10 23:22:59—11/11 00:46:57，两BC相差0.07秒；只能确认外部终止，与集群关停相容，不能仅程序日志证实物理断电。GPU22刚提交有效PPO更新17.43秒即停止、GPU11距更新约58m51s，加上CPU同期停止，不支持统一归因两小时GPU闲置；四有监测作业OOM/oom_kill均0，另两MLP无同样监测。
- 六latest＋MLP22 pending139（6795状态）＋BC11 pending182（1402）实际只读加载通过；模型／AdamW有限、configSHA与已提交Actor摘要匹配，pending的网络／AdamW／版本与latest相同，源SHA不变，无Agent构造／optimizer.step／SCIP／CUDA。pending需从更新前重放整轮，不能叠加原内存73/81或6/18部分步。六执行锁空闲、预算attempt均已结算。
- 续训命令统一既有paper-resource-resume入口，`run --experiment trainmode --variant b0 --seed11|22 --device cuda --ram-gib48 --resume --no-total-wall-limit`，4CPU/48GiB/1V100；另四`--experiment aligned --variant mlp|mlp_bc --device cpu --ram-gib64`，4CPU/64GiB/0GPU，分别恢复CPU recovery现有目录，不能误回旧GPU父分支。500/3600/GAE/三epoch AdamW不变，无需改冻结脚本、删锁或清空预算；未实际提交作业。
- 固定100全66评测仍在终端服务器，flock3786／shell3792／分片1主3995和四worker活动；分片0已85/85，分片1已21/85，分片2/3等待，09:59北京时间快照共106有效0失败、20实例100项完整五方法配对。不是全66完成，不重复启动或发表局部总体排名。用户只有4CPU/32GiB，四分片顺序运行；后台SSH断开不影响进程。本轮刷新progress，保持冻结100权重及现有进程。
- 新报告`docs/diagnostics/tgppo_20261111_restart_audit/REPORT.md`、RESUME_COMMANDS.txt、EVIDENCE.json、CHECKPOINT_AUDIT.json与EVAL_PROGRESS_SNAPSHOT.json。旧49／新53／recovery6／resource6及评测冻结输入验证通过。本次只读核查和交付，无新增训练／求解／取消作业，后续六Custom由用户手动提交。

## 2026-11-10 四模型固定100／完整66首轮评测授权

- 用户要求评seed11旧B0、新B0、MLP、MLP＋BC，随后明确选择完整66测试实例、solver seed0；本轮330项＝264模型＋66新relpscost。这只覆盖原固定500门槛之外的独立100中途评测，不改变原500正式入口，不用测试挑checkpoint，不改训练或暂停正在运行作业。
- 独立包`tgppo/experiments/paper100-test66-20261110`，四CPU Custom入口`custom_job.sh run --shard 0|1|2|3`；每个4CPU／32GiB／0GPU、CPU型号不限，任务85／85／85／75。3600秒求解／4200watchdog；普通4单线程worker地址6GiB，mspp16独立串行24GiB、不与普通池重叠。无脚本累计总墙时限；全触顶纯搜索排程约23小时、watchdog窗口约27小时另加准备开销，提前证明会缩短，排队另计。
- 四源均为原GPU目录固定actor-0100.pt及episode-0100.pt，新B0已实际保存100。已复制RUN/frozen只读，SHA与完整checkpoint provenance核实；不读latest，不用CPU续训分支的后续权重。plan.json包含完整配置／进度／SHA与66分片清单，task_manifest.csv列330项。
- 逐项JSON在共享盘原子落盘，显式--resume，同环境复用有效项；完整五方法实例面板跨环境保留，不完整面板不能跨CPU/软件/主机/核心环境拼接，换环境重新配齐。无自动重试/扩种子/训练。四作业完自动汇总，未齐仅progress、不全局排名；新旧B0、结构、BC、vs原生的PDI/证明/部署成本分列，共同/无Actor/混合另报。单训练／solver种子、100中途预算、历史测试暴露边界保留。
- 本地11项合成检查通过；实际四Actor CPU加载与父/子fork概率动作逐位一致，无Critic构造／CUDA初始化／SCIP optimize／训练；旧49／新53冻结训练包与保护源码SHA未变。RUN`tgppo/runs/paper100-test66-20261110`，报告`docs/methods/tgppo_paper100_test66_20261110`。当前为prepared_not_submitted，用户需手动提交四CPU作业；不能称330已跑完或已有新收益。说明与命令在新包docs/CUSTOM_JOBS.md／commands.txt，结果齐全后提交分析并停止验收。

## 2026-11-09 MLP＋BC-seed11释放后CPU续训

- 用户报告旧paper-resume的MLP＋BC-seed11被释放，要求CPU命令。只读确认原paper-aligned/mlp_bc-s11为SIGTERM stopped、next137、latest存在、pending无、执行锁空闲，最后attempt已结算22567.654秒；CPU副本尚不存在。不得再称该BC-seed11仍运行。
- 可用48GiB资源入口`paper-resource-resume-20261109/custom_job.sh run --experiment aligned --variant mlp_bc --seed 11 --device cpu --ram-gib 48 --resume --no-total-wall-limit`，4CPU／48GiB／0GPU、CPU型号不限，恢复137后采第138，原3600上限与500目标保留。CPU输出仍是recovery/cpu-aligned/training/mlp_bc-s11，首次独立复制模型／AdamW／游标，此后重复同命令恢复自身latest；未启动实际作业／求解或改权重。

## 2026-11-09 新B0 GPU恢复与48GiB容器内存

- 用户先称新B0-seed22 GPU失败，补充的实际命令却是trainmode原custom_job.sh `--seed 11 --device cuda --resume`。必须区分作业名与argv；已对上北京时间13:47的seed11 episode12缓存身份失败，CPU Gold6148→E5-2680v4，V100-PCIE32GB。latest SHA d6e44d7...与旧episode12策略身份相同，仍已提交11；seed22原GPU仍12，未找到另一份可归因于本次seed22 GPU的新日志。报告`docs/diagnostics/tgppo_20261109_resource_resume/REPORT.md`及FAILURE_EVIDENCE.json。
- 用户要求b0_new_seed_22可申请48GB。64GiB是原GPU preflight硬门槛，不是网络数学要求；新B0-s22现有12份采样worker峰值RSS最大2.506GiB，s11最大2.444GiB，旧B0-s11原路径136记录最大3.516GiB，均不是整容器峰值，不能保证500所有阶段。支持本轮先申请4CPU／48GiB／1GPU，并监测总容器峰值；RAM与显存分开。
- 独立包`tgppo/experiments/paper-resource-resume-20261109/custom_job.sh run --experiment trainmode --variant b0 --seed 22 --device cuda --ram-gib 48 --resume --no-total-wall-limit`（实际失败seed11则改seed11）。复用原recovery GPU恢复与CPU cohort，原GPU目录/模型/GAE/AdamW/微批/500/3600/worker32GiB均不改；仅preflight用配置副本覆盖RAM门槛，资源覆盖写environment。支持gpu别名规范为cuda。每30秒cgroup current/peak/OOM与RSS写独立RUN memory，保留实际variant/seed；不保证解除2hGPU闲置回收。
- 原49／新53／recovery6冻结文件均未变，未改正在运行脚本/权重、未提交/取消或真实训练/SCIP。6项合成资源检查执行真实冻结preflight函数AST与假硬件（无Torch/solver导入），测试48接受、32拒绝、64选项、CPU/软件守卫、gpu别名/恢复转发、内存采样。初次依赖导入90秒到限记录单存，不是训练故障；最终检查结果在CPU_CHECKS.json。实际GPU/48GiB整轮未本地验证，不能写GPU实测通过。新包README/commands.txt供手动提交。

## 2026-11-09 100episode进度与曲线答疑

- 用户问新B0-seed22的CPU续训命令，以及seed11是否到100／可否评测／曲线。统一入口参数`--experiment trainmode --variant b0 --seed 22 --device cpu --resume --no-total-wall-limit`，4CPU／64GiB／0GPU；其CPU副本已建立、SIGTERM stopped、next12，重复恢复自身latest。最新13:48快照新B0-seed11父CUDA又因episode12缓存身份失败，CPU分支也SIGTERM stopped、next11，不能再称新B0 CPU仍运行。
- 旧11/06 paper-aligned seed11 B0／MLP／BC为137／113／137，固定actor-0100齐全，有效更新132／108／132；新11/08 trainmode B0 seed11／22只有11／12、有效10／12，无100。CPU分支继承游标但updates为空时，不可写成从零训练；图保留父记录，CPU新增分开列，不拼旧B0与新B0。
- 只读新绘图产物`docs/methods/tgppo_paper_aligned_20261106/training_curves_20261109`，PNG/PDF/SVG、metrics.csv、SNAPSHOT、README与plot脚本；原11/8图不覆盖，无模型导入、优化步或新增SCIP。旧seed11 B0 clip0/132，MLP非零39/108、BC62/132；loss波动无持续收敛证据，不以loss/clip推出PDI收益。新B0探针更新后距均匀TV中位5.41e-9/7.66e-8，仅至多8训练状态，尚未见强响应，不凭早期11/12宣判长期失败。
- 旧三组0／100的6固定Actor SHA已读核实，MONITOR100_PROPOSAL.json仍为未执行提案：3训练实例dcmulti/l152lav/comp21-2idx×6模型＋3同机原生＝21，solver0／3600、4CPU纯搜索若全触顶约6h另留开销，不需GPU。这仅训练内监控，不是独立开发或正式测试，新B0到100后再同预算加入。用户这轮未授权明确新增求解预算，未启动该面板或66；正式固定500门槛不变，不把旧120秒测试结果借作本轮3600结果。

## 2026-11-09 作业失败／两小时GPU闲置回收修复

- 用户要求排查今早失败／释放并修复，追问CPU时限与断点；随后澄清平台无总运行时限。26分钟是旧脚本累计172800秒限额的余量，不是平台余额，不得混淆。独立新包`tgppo/experiments/paper-recovery-20261109`已准备；原49／新53冻结文件均无变更。手动入口`custom_recover.sh run --experiment aligned|trainmode --variant ... --seed 11|22 --device cpu|cuda [--no-total-wall-limit]`，自动显式resume，目标仍500／单次3600，未提交／取消作业、无真实训练或SCIP。
- 截图对应新B0两次今早启动均漏--resume，日志`Prior execution retained`，权重游标仍11／12。旧MLP-s22第71在旧入口触发CPU缓存身份错误，已提交70；旧MLP-s11已提交113、pending第114完整，3600.515秒归一化＋3601.035秒采样，pending后1.144秒SIGTERM，与用户2小时回收规则吻合，不是本次OOM/NaN。旧B0-s11当前137，因累计预算剩1594.454秒不足5100预留而主动停；另有08:39执行锁冲突日志但该旧launcher未打印variant，不能百分百绑定截图单项。BC-s11／22快照137／78且仍推进，保持不动，不并发迁移。
- GPU新入口在原RUN恢复、处理仅CPU型号变更的collect cohort，修复启动/跨节点错误，不保证免于再次闲置回收。新入口直接调用原train，异常状态仅在当前执行锁内写，修复跨容器相同PID可能覆盖活动状态的风险；不能删除锁绕过。
- CPU模式只在实际手动run时向`runs/paper-recovery-20261109/cpu-{aligned,trainmode}/training/...`导入独立副本：权重、2 AdamW、RNG、进度、有效pending／源预算保留，只改learner_device／配置SHA和pending清理目录，源轨迹不删。父任务有活动锁或未结算计时记录时拒绝迁移；重复命令恢复自身latest。新B0仍train，旧MLP模式与标准GAE／逐episode3轮AdamW均保留；CPU/GPU浮点及dropout流不逐位等价，未来评测必须记录迁移来源，现冻结GPU评测不自动接纳CPU分支。
- `--no-total-wall-limit`是显式执行资源覆盖，关闭脚本累计48小时停止、保留所有历史elapsed与新计时，仍500终止，单次3600／watchdog4200不变。不带该项仍48小时。不是删除账目或更改原冻结配置；用户手动提交含此项的命令才执行。CPU要申请4CPU／64GiB／0GPU；在GPU作业内隐藏GPU仍会回收。CPU更新可能更慢，尤其共享重Critic，不承诺500耗时。保持CUDA流需另做CPU采样→GPU更新阶段调度，目前无平台自动提交接口，未实现。
- 已核实11/8 `cpu_timing.py`55秒／外层60秒仅保护一次诊断，不是训练防回收脚本。缩短SCIP3600会改变任务与分母，旧权重只能按新协议热启动；仅缩watchdog会杀任务，不保证有合法轨迹。不要以“时限修复”静默改变论文单次求解预算。
- 16项合成CPU检查最终通过，含原全模型桥接、AdamW/RNG/游标保持、pending中断精确重放只提交一次、活动父任务拒绝、跨容器重复启动不覆盖状态、显式无累计墙时限制。先前90秒诊断外限到时另存记录，最终套件122.65秒通过；另5份现有checkpoint只读加载完整、无优化步/CUDA。报告`docs/diagnostics/tgppo_20261109_job_recovery/REPORT.md`，脚本说明/四个推荐CPU作业命令在新包README／commands.txt。优先新B0两seed＋旧MLP两seed；旧B0继续可用同入口但不替代新B0。本阶段交付停止，实际训练仍由用户提交。

## 2026-11-08 新正文训练副本与断点续训

- B0-s11此前新增A100续跑现已确认失败，原launcher `20261108T103529-V5Yzje.log`／state均为episode92 `Cached task identity changed`。环境差异仅CPU8358→Silver4216；latest SHA与旧episode92身份一致，仍已提交91、pending无，不是此次GPU回收／CUDA错误。新attempt已结算3673.458秒，累计123949.463、余额48850.537秒约13h34m。改用独立paper-resume入口恢复，不再称该B0正在normalization，勿删除旧缓存/权重/预算。证据`docs/diagnostics/tgppo_20261108_resume_hardware/B0_FAILURE_FOLLOWUP.json`；本轮仅只读核查和记录，无训练/求解/提交。
- 用户问旧seed11能否换V100。独立恢复适配器`tgppo/experiments/paper-resume-20261108/custom_resume.sh run --variant b0|mlp|mlp_bc --seed 11|22 --resume`已准备，8项合成缓存测试及完整只读校验通过，原49／新53冻结源码均无变更。原paper主程序仍负责CUDA检查、模型/AdamW/RNG/pending/锁/累计48小时，不改模式/GAE/500/3600。仅当collect身份除CPU型号外完全相同时，重跑采集到原episode下resume-cohorts/身份哈希，保留旧identity/attempt；新CPU原生分母走原身份隔离。V100实机未本地验证、跨硬件不保证逐位一致，未训练/求解/提交。报告`docs/diagnostics/tgppo_20261108_resume_hardware/REPORT.md`。最终只读发现B0-s11已有新A100/Silver4216续跑attempt，state normalization，不能建议重复提交或仅因本机无进程判停止；该未结束attempt保守预留余额，不能把预留额当实际耗尽。MLP／BC剩余已结算约17h59m／20h50m。
- 用户随后报告旧seed11三组也被释放，要求原入口续训。本次只读核验：三组state均SIGTERM stopped、next_episode91；latest均存在、pending均无、summary无。旧49冻结文件仍无变更。当前剩余累计预算B0约14h35m、MLP17h59m、MLP＋BC20h50m；沿用paper-aligned原入口加--resume，不清零进度或预算。三组episodes/92均已有identity，CPU为Platinum8358，无有效attempt结果；同CPU／软件可恢复重采第92，若换CPU会触发旧cached_job身份保护，需要保留旧记录另处理，不能建议直接删除目录。本次未提交作业／改旧脚本；不能再称旧seed11三组仍在正常运行。
- 最新MLP重训答疑：不能仅因Actor eval→train就要求四个MLP作业清零。当前11/06与11/08的GAE／targets／seal函数AST相同、Memory相同、500／3600／25×5／AdamW／逐episode3epoch一致；共享MLP无Dropout／BatchNorm及training条件分支。用旧seed22两个pending真实权重／AdamW、各4个transition的CPU局部切片分别执行旧／新3逻辑PPO步，Actor／Critic／优化器／RNG／targets逐位相同，参数最大差0；无SCIP／GPU或生产权重写入。证据`docs/methods/tgppo_paper_trainmode_20261108/MLP_RETRAIN_DECISION.md`与`MLP_UPDATE_EQUIVALENCE.json`。建议优先新B0两种子，旧MLP继续各自500；全六重训仅为统一硬件等额外控制的选项，不自动取消任何作业。不把有限CPU核验说成跨GPU等价或PDI收益。若复用旧MLP做最终比较，要固定500及显式来源清单；现新冻结评测仍要求六份新条件，不能静默改旧checkpoint标签或绕过门槛，后续需单独准备混合来源评测入口。
- 用户最新澄清：seed11三组旧paper-aligned正在A100正常运行，保持不换卡、不停作业、不改旧脚本；考虑用V100恢复旧seed22的MLP／MLP＋BC，同时优先启动最新trainmode实验。旧续跑与新实验必须明确分开，不能将旧模式续跑称为最新模式。只读核验旧49／新53冻结文件均无变更；seed11三组latest均89，旧MLP-s22 latest47／pending第48、BC-s22 latest52／pending第53，7份读取快照均含完整模型、AdamW、RNG和轨迹字段。旧脚本仅要求恢复时可见CUDA数量一致（这些检查点为1），没有A100型号硬锁；真实V100检查未本地执行，跨GPU不承诺逐位相同。旧任务V100恢复仍用paper-aligned原入口加--resume；新trainmode从新目录初始化，输出／预算独立。
- 用户随后明确选择V100执行新训练。当前交接以`custom_job.sh run --variant ... --seed ... --device cuda`六作业为准，CPU入口保留备用；默认CPU不改，必须显式带CUDA参数。新V100六条命令在`docs/methods/tgppo_paper_trainmode_20261108/V100_CUSTOM_JOBS.md`和`V100_commands.txt`，未改变已冻结训练包；源码／外部输入／保护文件校验通过，真实V100检查在用户实际作业中执行。
- 用户要求独立副本恢复Actor训练模式、核对GAE／更新组织，并明确选择“论文正文＋标准GAE”；后续追问平台释放时必须支持断点续训。新包`tgppo/experiments/paper-trainmode-20261108`，旧paper-aligned六作业／源码／权重不动，不自行提交或取消Custom。
- B0／MLP／MLP＋BC×11／22；500完整episode、3600秒、每episode后3epoch AdamW；Actor采样eval／更新train，B0有效dropout=.05，MLP无dropout仍0。继续已有标准分episodeGAE，不复制公开代码额外done遮罩／多轨迹批更新；不得宣称作者完整实际流程已取得或原样复刻。
- 默认CPU训练入口`custom_cpu.sh run --variant ... --seed ...`，4CPU／64GiB／0GPU，避免平台低GPU占用回收；CUDA显式`custom_job.sh ... --device cuda`，同轮设备一致，不在resume中切换。新条件累计48小时、手动提交；不能保证48小时到500，不缩短求解凑数。按完整身份只读复用旧3600原生缓存，另记SHA；不写旧缓存或复用旧PPO权重。
- “拒绝旧模式checkpoint”是防止跨实验混用，不是论文规定，也不是禁止断点。同一新／旧实验各自已有--resume，恢复模型、AdamW、RNG和游标；PPO中断从pending重放，已提交不重复；SCIP中断仅重跑未完成episode。B0开启dropout、Actor／Critic步间中断的CPU合成恢复逐位通过；不承诺跨硬件逐位等价，不删预算绕过累计上限。
- 新日志把初次更新前的train/dropout比值扰动与eval分布实际变化分开；clip非零可来自dropout，不视为学会。至多8状态训练监控非开发／测试，不替代PDI、证明率和部署成本。
- 准备报告`docs/methods/tgppo_paper_trainmode_20261108/REPORT.md`。20项CPU检查通过（含原B0／MLP更新、GAE、异常概率、恢复、只读缓存和完整评估覆盖）；未启动真实训练／SCIP／CUDA。新包交付后等待用户手动提交；现有实验结果不因恢复train模式预先改写为有效或无效。

## 2026-11-08 B0学习响应重新核验

- 用户要求追查历史与本轮B0 clip_fraction持续为0并判断曲线是否正常。新报告`docs/diagnostics/tgppo_20261108_b0_learning_audit/REPORT.md`，只读日志＋冻结CPU前向/score反向，无optimizer.step／SCIP／CUDA，不改原基线或正在运行的500协议，不自动停止或重提GPU。
- 历史两轮B0的100episode实际均120秒，尚未识别用户提及的360秒运行；两种子各25有效轮clip全0。旧server500为125／123有效轮clip也全0。本轮87／42episode、82／40轮、3966／2130 AdamW步仍全0。
- 22固定状态（17训练实例）上，当前B0 s11／s22距均匀TV中位1.81e-8／8.43e-9；MLP／MLP＋BC同输入0.02798／0.09463，只说明输出响应不坍缩，不证明PDI收益。B0权重实际变化，但输出主要共同logit平移；候选评分链弱响应已复现，不能再仅称“值得关注”或等待100/500求解才确认此项。
- 机制仍为Transformer／BiMatching个体通路压缩＋窄TreeGate链（含LayerNorm2）多阶段作用，未唯一归因单层，不能重开已失败LN2／尾部扫描。最终层Linear2→1无LN1。3anchor的train/dropout与float64仍近均匀，有限模式核对未恢复强偏好。
- 当前／旧Actor和modules与在线作者公开Git blob逐字节相同，排除本轮复制把这两文件改坏；不证明作者正式权重或完整运行配置同样退化。曲线“无NaN／优化器在动”与“Actor有效学习”必须分开，当前B0不宜称健康强基线；继续500不能承诺修复。下一步先审阅基线可学性和作者正式权重／实际设置，不自动改模型、扩大预算或宣判论文失败。

## 2026-11-06 论文主协议对齐：新训练准备与手动Custom交接

- 用户要求正式对标TGPPO优化器、更新、数据／solver／RNG、episode与单次求解时限，重训B0／MLP／MLP+BC；明确确认全25训练、固定500主对照，不再用原5开发选模；单作业48小时并可断点续跑。覆盖旧“不扩大当前100episode”的阶段限制；不修改旧基线／旧权重，也不重新启用CR、持续CE、CL或树信用分配。
- 新独立包`tgppo/experiments/paper-aligned-20261106`，`custom_job.sh run --variant b0|mlp|mlp_bc --seed 11|22`共6个用户手动Custom，建议A100×1／4CPU／64GiB，CPU型号不限；脚本不自行提交GPU。25×solver0–4的125组合各周期独立打乱4轮，500完整episode、SCIP3600／watchdog4200、逐完整episode后3epoch AdamW，保留原H3／Critic与已验收正确性修复。短非空episode直接更新、空episode照计不虚构步骤。
- 新3600原生归一化按实际CPU／软件和完整求解身份懒加载共享，不能复用旧120秒分母；不同CPU重新测对应分母。Actor/Critic及modules与作者代码相同，MLP仍24493参数；BC复用合法NPZ，重做1000步内部固定规则选初始化，不从旧PPO权重续训。动作／日程／Actor初始化／PPO更新／BC流隔离，目标GPU实际检查和跨作业初始化指纹一致才采样。
- 本地10项单线程CPU检查通过（完整B0/MLP合成更新／导出、短episode、模型/AdamW/RNG精确恢复、pending更新中断只提交一次、预算不清零、旧120归一化拒绝、专家合法性、2310完整覆盖／Actor参与分解）。最终套件含导入约10.23秒；未启动任何真实PPO、SCIP求解或GPU。证据`docs/methods/tgppo_paper_aligned_20261106/{REPORT.md,CPU_CHECKS.json,cpu_checks.log}`；A100兼容性必须由实际Custom入口重检，不能称本地已验证GPU。
- 当前保守冻结为每条件／种子累计172800秒，含准备、原生、等待与失败；显式resume消耗余额，缺最终记账按分配上限保守计，不自动清零。0/100/200/300/400/500保留，pending/latest原子区分；到限未满500先交实际进度，另确认追加，不能保证一夜或48小时完成。6组上限288作业小时，不挪用旧预算。
- 原66历史已暴露，不称新盲测。目标评估固定500，66×5×（6模型+原生）＝2310项CPU，17固定分片（16普通＋mspp16独立串行）；只准备入口，GPU训练后不自动测试。六组500完整才允许CPU入口创建attempt，全2310齐全才排名；共同Actor／无Actor／混合分解，保留PDI测量界、证明／PAR-2／成本。现在只请用户提交6训练，CPU评测先等实际进度与成本审阅。
- 新核实作者XLSX为33easy＋32hard，缺opm2-z7-s2，与正文33/33不符；名单与源SHA在包docs/author_subset_audit.json。完整66不删，作者固定33/32附表，不补猜成员／不按新结果重划。仍未重跑嵌套Optuna；保留Actor eval概率重算、独立Actor/Critic、CPU-GAE与微批工程边界，不宣称全部论文执行模式／绝对性能已完整复现。

## 2026-11-06 持续专家监督110开发评估完成并交付

- 用户运行supervised-ppo-20261105/custom_cpu.sh并报告完成，要求分析。实际110/110有效、100模型+10同机新原生、Silver4214、1842.379/3000秒；tar与JSON逐条相等，10源Actor/完整checkpoint/冻结副本SHA、62包/83外部/41保护源码与实际CPU fork检查通过。新增results/{RESULTS_REVIEW.md,review_results.py,REVIEW_DATA.json,verified_summary.csv,mean_decomposition.csv,FINAL_AUDIT.json}，原自动报告/CSV/源码/权重不改，无新增求解或训练。
- 全5开发两种子合并：BC0 3038.224；control20 3026.187、100 3035.890；expert20 3039.780、100 3044.066。学习组证明率均60%，新原生70%。固定100 expert对control全体+0.269%、共同Actor4+0.166%；seed11为−0.371/−1.040%，seed22为+0.915/+1.388%，没有跨种子稳定收益。
- 预定0/20/100全5选模：control两seed均20，expert s11选100、s22选0。expert selected PDI3031.100对control3026.187（全体+0.162%）；共同Actor901.017对901.618（−0.067%，种子方向相反）。相对control选模s11全体+0.062/共同−0.735%；s22+0.263/+0.614%；无跨种子同时胜control与BC的共同实例，门槛失败。s22选0不等于没训练；不事后放宽门槛或改变选模面板。
- roi5alpha10n8全部模型无Actor，约占PDI数值76%；固定100净差+8.1757中ROI贡献+6.9710（85.3%），共同Actor贡献+1.2047。选模净差+4.9138由ROI+5.3944与共同−0.4806组成。不能归因无Actor差异，但共同Actor仍跨种子不稳，结论不因剥离ROI反转为稳定收益。
- expert两seed实际537/597专家batch、8592/9552样本，末加权loss约.006539/.006220，非未启用；control0。初始化配对相同，control24有效PPO轮与498/558步、expert25轮537/597步，固定episode不等同步数。s11训练硬件Silver4110/V100-PCIE vs Gold6148/V100-SXM2；s22同Gold6148/SXM2仍退步，记录因果边界，不自动为挽救增量重训。
- 本轮训练+开发闭环结束，建议停止当前固定CE增量的追加GPU、λ扫描与66测试，不补跑即可交付。保留无BC轻量Actor成本/部分求解证据、BC尾部风险和CE混合负结果；当前非论文性能复现。下一阶段如继续，先对齐强TGPPO基线协议，再提公平预算；本轮不启动500/3600或其他结构/CL/信用分配，报告后停止等验收。

## 2026-11-06 B0与论文复现边界澄清

- 用户质疑“当前预算下”与是否完整复现，要求核对论文训练episode及性能。新说明`docs/methods/tgppo_light_actor_test66_20261105/BASELINE_PAPER_ALIGNMENT_20261106.md`；只读原文/源码/记录，无新增训练或求解。
- 原文arxiv2511.12986v1 §4.3/5.3明确总500完整episode、每次最多3600秒、25实例×solver0–4、AdamW、文字描述每episode后3优化epoch；§5.4测试66×5solver、3600秒，easy节点/hard PDI与SGM/实例胜率。当前本轮B0为100episode、120秒、20训练＋5开发、训练solver0–2/测试0–1、Adam、4episode收集批/256样本阈值/每次3epoch，全66算术均值。五层256维8头原Actor结构保留，不能说架构被缩小，也不能说论文完整性能已复现。Adam差异不是正确性修复或创新；名义dropout与Actor有效eval及shared front-end等仍有对齐边界。
- 历史server6两份B0确实训练各500（125/123有效批更新），开发均选100，但也是120秒/Adam/20+5协议；本轮独立重训100，不能说项目从没500，也不能混旧500批次与新100表格绝对值。旧论文表3胜率90.62%对TBRANT PDI、46.87%对relpscost均是实例胜率，非平均PDI降幅或证明率。论文亦未全面胜原生，当前输原生不自动证明代码错误。
- 本轮B0 PDI2708.509/证明17.803%，同批原生2653.381/31.818%；没有匹配3600秒等协议，无法定量回答达到论文性能比例。当前方法只可称相对此受控预算B0的初步增量，不能直接声称超过完整论文TGPPO。正式对齐/500或3600矩阵仅为后续建议，不因本轮问差异而自动启动或改已有基线。

## 2026-11-06 全66评估完成：结果复核交付并停止

- 用户报告四CPU作业结束，要求分析与建议。核实1188/1188有效：1056模型＋132本轮新原生，全部66含mspp16；四作业324/306/306/252项，均Silver4214不同主机，143.08/120.42/128.27/176.45分钟，并行约2h57m、合计9.4703作业小时，均在240分钟限内。
- 新增只读`docs/methods/tgppo_light_actor_test66_20261105/{RESULTS_REVIEW,MEETING_UPDATE_20261106}.md`、`review_results.py`、REVIEW_DATA.json与FINAL_AUDIT.json。独立复算主表、四tar共1188记录语义等值、源/冻结Actor SHA、包15/外部9/原57/保护41与实际fork核验通过；原自动REPORT/CSV/日志/冻结包不改，无新增求解/训练/重试。
- 全66 PDI：原生2653.38，B02708.51，B0+BC2729.88，MLP2679.22，MLP+BC2768.45；证明率31.82/17.80/17.80/21.59/21.59%。MLP−B0中点−1.08%、共同Actor57−3.10%，Actor58.19→4.17ms（−92.83%）、墙时−2.83%；全66积分差额界跨0，不能称确定全量PDI收益。对原生MLP仍PDI+0.97%、证明低10.23pp、墙时+5.79%。
- BC对MLP全66+3.33%（s11+2.24/s22+4.45）、共同Actor+5.36%；完整方案对B0中点+2.21%、共同Actor+2.10%。ns1208400单例BC净差贡献+108.49，超过全体净差+89.23，MLP四次证明、BC两次；去掉它BC−0.74%仅是事后敏感性，主表不删。完整方案35实例两种子测量界均改善、8均退步；不能写BC普遍无效，也不能继续用旧8−6.25%称全66稳定收益，不能事后按66重选主模型。
- 5个mspp16记录SCIP实际240.94–261.35秒（配置120），120秒PDI重建区间很宽；全66中点排序须连同区间报告。共同Actor57按既定规则，5全无Actor、4混合单列；不将无Actor变化归因神经决策。保留测试历史暴露/结构CUDA流未完全对齐/同型号不同主机与访问状态差异，积分测量界不是统计CI。
- 同步只读发现持续监督control/expert×11/22均training_complete且100episode、24/25轮PPO更新、无待更新transition；development-110尚无attempt。下一步建议用户另提交既有`supervised-ppo-20261105/custom_cpu.sh run`，一个4CPU/32GiB/0GPU，110项开发累计50分钟，不追加GPU。该CPU阶段以前已批准，当前仅给建议/启动命令，不在本轮本地执行。新监督仍只回答相同BC初始化下的增量，不替代无BC强对照。
- ns1208400新增solver2/3、MLP/MLP+BC×两训练种子，8模型+2原生=10项/4CPU/累计15分钟，仅为后续已见尾部复测提案，未授权/未执行；不据66调λ或宣称新泛化，不自动长训/CL/树信用分配或改基线。本阶段报告交付后停止等待验收。

## 2026-11-05 全66最终资源调整：仅四个并行CPU作业

- 用户明确纠正为4个并行作业，平台仅剩160GB内存，允许提高单作业时限。覆盖下面尚未提交的九分片准备状态；所有66仍完整评估，不能再让用户申请9作业。当前包/入口路径不变，run仅接受--shard0／1／2／3。
- 每作业4CPU／32GiB／0GPU，合计16CPU／128GiB（约137.4GB），CPU型号不限；每作业累计14400秒含全部准备/求解/保存/汇总，平台4小时15分，预计同时执行2–3小时不含排队，无自动重试/预算重置。4作业累计最多16作业小时。
- 分片0／1／2／3为18／17／17／14实例，324／306／306／252任务，总1188。普通实例按固定哈希分18／17／17／13，mspp16进3；分片3必须先等4-worker普通池所有子进程退出，再单worker/24GiB/watchdog300串行运行mspp16，不与普通worker重叠占内存。普通仍4单线程worker/8GiB/watchdog180，搜索均120秒，模型与原生同例外。
- 新全66脚本/文档/清单据此重封存，旧16入口同样转发当前四分片，不存在新9或16真实attempt。结果仍在docs/methods/tgppo_light_actor_test66_20261105；四分片全部1188项有效才出全量主表。正在运行的GPU监督包与旧方法权重不改。

## 2026-11-05 完整66测试授权，替代未提交16实例方案

- 用户明确说明四CPU作业尚未提交，要求改脚本直接跑66。授权完整66固定旧权重CPU评估与8普通＋1特殊分片，不授权本地真实求解、训练、改模型或新持续监督包。仅准备入口，由用户手动提交。
- 当前包`tgppo/experiments/frozen-test66-20261105`，入口`custom_job.sh run --shard 0..8`共9作业；0为9实例162项、1–7各8实例144项、8为mspp16的18项。总1056模型＋132本轮新原生＝1188项，所有66本轮新测，不拿旧8结果填入。原8份100episode Actor、solver0/1、SCIP120秒及指标保持。
- 普通作业4CPU／32GiB／0GPU、4单线程worker/8GiB/watchdog180；mspp16作业1CPU／32GiB／0GPU、单worker/24GiB/watchdog300，搜索仍120秒，模型与原生同例外。CPU型号不限，每作业累计7200秒含准备/执行/持久化/汇总，平台建议2小时15分，九作业累计最多18作业小时；估计同时执行1–2小时，不含排队，无自动重试或预算重置。
- 旧16包custom_job.sh兼容转发新入口、旧指南加替代提示，原16 release和审计保留为历史状态，SHA变化由本次替代明确授权；其余旧16源码不改、原RUN无attempt。请给用户新九条命令，不能只提交旧四条。
- RUN`tgppo/runs/light-actor-test66-20261105`，报告`docs/methods/tgppo_light_actor_test66_20261105/REPORT.md`最后作业自动汇总。九分片1188项未全有效不称全66完成、不按部分矩阵排名。保留历史测试暴露、结构CUDA流与异构CPU限制，新持续监督仍独立110开发评估。本阶段只准备检查；实际结果齐后交报告等验收，不自动扩训。

## 2026-11-05 固定16测试实例的四CPU作业授权

- 用户明确同意拆成4个CPU作业并要求启动脚本，覆盖下面CPU_EVAL_PROPOSAL“未授权”的历史状态；仅准备独立入口，由用户手动提交，不在本地启动新增SCIP。原提案、已交付报告和正在运行的持续监督GPU包保持不变。
- 新包`tgppo/experiments/frozen-test16-20261105`，入口`custom_job.sh run --shard 0|1|2|3`；每个4CPU／32GiB／0GPU、72项、累计3000秒含准备/求解/保存/汇总，无CPU型号限制。全部256模型＋32新原生＝288项，不训练、不选模、不扩实例/自动重试或重置预算。
- 固定旧B0／MLP×无／有BC×seed11/22的100episode权重；16实例与先前提案逐字一致，每实例全部方法和solver0/1、原生在同一分片配对。是原66测试目录的子集，不是开发集，不称全66或完全未见保留测试。新持续监督分支不进此矩阵，其110项开发评估仍独立。
- 交付说明在新包docs/CUSTOM_JOBS.md。RUN`tgppo/runs/light-actor-test16-20261105`；结果`docs/methods/tgppo_light_actor_test16_20261105/REPORT.md`由末个作业自动汇总。四分片未完整时只报缺项，不排名；完整后报告结构/BC独立增量、原生PDI/证明/PAR-2/成本、共同/无Actor与硬件边界，阶段交付后停止等待验收。

## 2026-11-05 23时 GPU排队期间的交付与方法澄清

- 用户询问GPU四作业未调度时还能提前做什么，并担心“只验证持续监督”意味着旧MLP＋BC失败/贡献被撤销。明确：当前方法仍有Actor结构替换、BC初始化两项已完成对照的因素，完整方案对B0的8探索实例PDI−6.25%；PPO相对BC-only混合、新持续监督只是第三项可选增强。持续监督不是明天交付的前提，不以其失败否定旧整体方案，不改既有队列、冻结源码、λ或预算。
- 新增只读交付`docs/methods/tgppo_light_actor_20261105/meeting_ready_20261106/`：六页MEETING_READY.md（实际结构Mermaid、主表、独立增量、部署成本和PPO消融）、main_results.csv、independent_effects.csv、QUEUE_AND_DELIVERY.md。由build_pack.py复用已落盘数据生成CSV；无GPU/新训练/SCIP求解，旧报告与当前GPU包不改。
- 提供可审阅但未授权/未执行的CPU扩展提案：旧B0/MLP×无/有BC×seed11/22八份固定100episode权重，16个当前8之外的训练外实例（固定名称哈希选，已知mspp16内存例外预先排除），solver0/1，256模型＋32新原生＝288项；4CPU Custom各4核32GiB0GPU、4实例72项，建议每作业累计50分钟，共200作业分钟。CPU_EVAL_PROPOSAL.json与proposed_cpu_tasks.csv仅清单，不是可提交入口；不能因用户本轮问可行性而自动启动新增求解、挪用原110评估或本地准备余额。旧数据历史可能看过，不称新保留集/全66。
- 共享盘状态动态：起初四组均无jobs启动记录，23:13快照出现control-s11 running，后读phase为normalization且已有实际环境记录，另三组暂无启动记录。未调用平台队列API；接续需看新RUN/jobs、实际phase和checkpoint，不从孤立running标志推断已训练/已完成。本轮不取消/重提GPU任务或改CPU训练执行协议。

## 2026-11-05 持续专家监督单变量实验授权与GPU交接

- 用户已明确授权按上一阶段建议继续：保留轻量MLP和BC，停止扩大100episode，下一轮只验证持续专家监督；开发跨种子收益成立后才考虑扩训与冻结评测。覆盖此前“本轮未授权/未实现”的提案状态。当前实现与本地检查完成，等待用户手动提交GPU，不自行申请Custom或本地真实训练。
- 独立副本`tgppo/experiments/supervised-ppo-20261105`：control λ=0、expert λ=.01 × CE/log(K)，每个逻辑PPO minibatch16专家状态、一次辅助反向、共享原裁剪/Adam；仅12训练实例1468状态，288留出不进梯度。直接复用原MLP两种子实际BC100步初始化，不重做BC、不改Actor结构/特征/Critic/H3/GAE，不混入CL/奖励/树信用分配。当前旧整体方案正向证据保留；单项监督负结果不能写成整体方法失败。
- 四训练Custom：control/expert×seed11/22，各100episode、0/20/100保存、1GPU+4CPU+32GiB、CPU/GPU型号不限、每作业累计8400秒含准备/必要归一化/桥接/训练/导出，平台建议3h；不沿用旧22:00截止。重复/恢复共享剩余预算，已死而无elapsed的attempt保守计分配上限。新副本修复CUDA流；目标实际RNG/监督恢复检查通过才训练，同种子实际初始化指纹配对，不用旧PPO控制替代匹配新控制。
- 训练入口`tgppo/experiments/supervised-ppo-20261105/custom_job.sh run --variant control|expert --seed 11|22`；不在GPU作业中持有GPU做CPU评估。全部4训练完成后用户另提交`custom_cpu.sh run`，4CPU/32GiB/0GPU，110项=100模型+10新原生，同机原5开发×solver0/1，累计3000秒，无自动重试。提前提交立即退出，不创建评估attempt。CPU侧自动持久化/汇总，实际十模型CPU fork验证后才求解。
- 本地14核心、9类汇总、2入口复用检查通过，无真实PPO episode/新增SCIP/GPU；核心实耗113.4秒，额外独立进程含导入按上限保守记账，含最后CLI/stdlib保留累计374.684/600秒。lambda0与原PPO状态/优化器/RNG等价，监督不改固定轨迹Critic或全局RNG，恢复精确；实际目标CUDA尚待用户Custom。冻结62包文件/83外部输入，旧41保护源码保持。
- 严格方案`tgppo/experiments/supervised-ppo-20261105/docs/PROTOCOL.md`，提交说明同目录CUSTOM_JOBS.md；准备报告`docs/methods/tgppo_supervised_ppo_20261105/REPORT.md`。RUN`tgppo/runs/supervised-ppo-20261105`，最终CPU结果在`docs/methods/tgppo_supervised_ppo_20261105/results/REPORT.md`，当前尚无真实新收益。
- 每条件/种子公平按全5开发PDI选0/20/100，严格并列取早；同时报告固定预算差、证明率、共同Actor与无Actor。新监督相对匹配PPO和BC在两个种子、全体/共同Actor均改善且证明不降、至少2共同实例跨种子同胜才建议晋级；测量界非统计CI，训练硬件异构需注明。通过也只提交报告供验收，不自动300/500/66测试/改λ。

## 2026-11-05 21:28 开发集三时间点完成、结果交付并停止

- 用户报告Custom完成，已独立核对70/70有效（60模型＋10同机新relpscost）、持久JSON与tar逐条一致、输入32项／原冻结57文件／外部14项／保护41源码及先前38项报告证据SHA。累计1371.782/2400秒，Silver4214、4单线程CPU；没有本轮新增训练/GPU/66测试。
- MLP＋BC0／PPO20／PPO100全5开发PDI为3030.041／3037.678／3035.018，证明率均60%；原生3042.257、70%。PPO100对BC：全体+0.16%，共同Actor四实例−0.45%；共同Actor种子11+0.57%、种子22−1.47%，实例改善/退步0/4与3/1。不能宣称稳定PPO增量或已收敛。
- roi5alpha10n8全部模型无Actor，占总体PDI数值约76%；0→100全体均值+4.976由共同Actor贡献−3.311与无Actor贡献+8.287构成，不把ROI变化归因于模型。按预写全开发PDI规则，种子11描述性最低BC0、种子22PPO100；没有覆写/导出原最佳权重。相对原生PPO100 PDI仅−0.24%、证明率低10pp、optimize墙时+5.37%，不能说胜过relpscost。
- 分析交付`docs/methods/tgppo_light_actor_20261105/dev_curve_20261105/{RESULTS_REVIEW,NEXT_STAGE_PLAN}.md`，复核脚本`review_results.py`、`REVIEW_DATA.json`、`mean_decomposition.csv`、`FINAL_AUDIT.json`；原自动REPORT与所有旧冻结输出不改。
- 下一轮仅为建议：MLP固定合法BC初始化，匹配原PPO／PPO＋持续小权重专家监督×种子11/22＝4个新GPU Custom，先100episode；新副本核验CUDA流、旧基线只读，不以“已证实遗忘”为理由。0/20/100原开发统一比较预计110项CPU，建议独立50分钟；训练每作业建议新140分钟上限。只有跨种子PDI/证明/共同Actor实例收益成立才考虑扩训与冻结测试。尚未实现、授权或启动，不自动跑4作业／300／500／CL／全66；本阶段分析完成后停止等待验收。

## 2026-11-05 开发集三时间点Custom授权与交接

- 用户同意复用已有BC初始化/PPO20/PPO100，并明确要求改为集群Custom运行，准备下班。覆盖此前“70项未授权”的状态；本地不再执行求解。新独立入口`tgppo/experiments/dev-curve-20261105/custom_job.sh run`，由用户手动提交一个4CPU／32GiB／0GPU作业，CPU型号不限。
- 仅MLP＋BC、训练种子11/22、原5开发实例×solver0/1；60模型＋10同机新原生＝70项。沿用dev动作种子语义；BC是实际第100步初始化而非1000步末尾，20/100核对原完整checkpoint与Actor导出的张量。
- 新阶段独立累计执行上限2400秒，最多4个单线程worker、每项120秒/watchdog180；含导入/校验/求解/保存/汇总，无自动重试/预算重置。排队不计执行，平台1小时时限不扩展内部预算，不复用旧训练22:00日历截止。既有代码/权重/旧报告只读，不训练/GPU/全66，不启动后续阶段。
- RUN`tgppo/runs/light-actor-20261105/development-curve-70`；结果报告`docs/methods/tgppo_light_actor_20261105/dev_curve_20261105/REPORT.md`。自动持久JSON/tar并汇总；会话/IDE退出不影响Custom托管，强制杀死/节点丢失仍需核对实际落盘记录，不假定完成。
- 本次交付状态是“入口与离线检查完成，等待用户提交”，不是70项已完成。核查`job_state.json`、持久结果与有效任务计数后才能改写状态；重复提交锁和终态保护，不自动启动第二次或覆盖旧记录。后续只对本轮结果分析、交报告并等待验收。

## 2026-11-05 19:36 BC-only小消融完成并停止

- 用户授权的20/20项均有效，累计701.262/1200秒，16模型＋4同机新原生，4单线程CPU；无训练/GPU/重试，BC包装指纹等于原PPO初始化、张量等值，4源Actor与冻结代码不改。RUN`tgppo/runs/light-actor-20261105/bc-only-ablation-20`，最终报告`docs/methods/tgppo_light_actor_20261105/bc_only_ablation_20261105/REPORT.md`。
- PPO100对BC-only：neos-1215259在训练种子11/22为−4.68%/+12.09%，neos-512201为+4.38%/+3.66%；1改善、3退步，合并PDI+3.53%，学习组证明率0、原生100%。这是两个已挑选对象的机制消融，不是全局泛化结论；不能据此断言遗忘、H3错误或PPO全局无效。
- 因此不建议直接把8组盲目拉至300/500或先扩大66；优先复用现有BC/episode20/episode100在5开发实例补曲线。该提案为60模型＋10新原生=70项，4CPU预估20–30分钟、建议累计上限40分钟，无GPU；本轮没有授权或执行。用户关于长训/正式评测的问题仅作说明与成本估计。
- 100episode是20训练实例约五次覆盖的真实训练预算，当前PPO固定episode100，不代表已收敛；若新阶段值得长训，另建新副本修复CUDA流，主2×2×两种子共8作业，从头最多300、预定100/200/300开发选模，之后冻结评测。每组300估计3–4小时，500估计5–7小时，均非质量保证；并行开发15–30分钟，全66主2×2为1188项CPU、8普通＋1特殊内存分片估计1–2小时，不含排队。新预算/截止未授权，不能改旧140分钟/22:00配置直接续跑。
- 本阶段已停止等待验收，不自动启动70项曲线、GPU训练或全66。旧阶段报告/审计保持，新增周五补充在新阶段`MEETING_UPDATE_20261106.md`。

## 2026-11-05 19:24 BC-only小消融新授权

- 用户明确要求“你接下来现在本地完成20项小消融吧”。覆盖前条尚未授权的状态，只授权此前20项CPU方案：neos-1215259/neos-512201，现存MLP BC初始化与其PPO100episode权重，训练种子11/22、solver0/1，16模型＋4本轮新原生。累计1200秒含导入/验证/求解/清理；4单线程worker、120秒/watchdog180，无重试/训练/GPU。
- 独立产物`docs/methods/tgppo_light_actor_20261105/bc_only_ablation_20261105`、RUN`tgppo/runs/light-actor-20261105/bc-only-ablation-20`。19:24:32北京时间开始计时，timeout PID11883、runner11884；接续应核对实际进程/原始文件和预算，不能重复启动。BC张量包装为部署格式，经指纹/张量等值、实际概率/动作CPU fork核验后才开始20项求解，旧源码/权重不改。
- 用户同时询问100episode用途、是否延长训练再正式测试、GPU与耗时。该部分仅要求说明和评估，不授权更长训练、修改冻结配置或全66评估；根据20项结果给具体建议后停止等待验收。

## 2026-11-05 18:32 本轮完整交付并停止

- 条件式跨实例144/144有效完成：128模型＋16本轮新原生，累计3034.279/5400秒，worker合计11456.993秒，无重试/训练/GPU；与前面开发170/170一起完整收齐。控制器7396、CPU恢复6945、跨实例8519及其求解worker已退出，不再启动任务。
- 跨实例MLP对B0 PDI−4.17%，MLP+BC对MLP−2.18%，完整MLP+BC对B0−6.25%（训练种子11/22分别−7.44%/−5.07%，共同Actor七实例分别−8.09%/−7.81%）。neos-1215259、neos-612125、sp98ir在两个种子均胜B0、B0+BC、MLP；BC在bell5、neos-512201相对MLP均退步。
- 证明率所有学习条件均37.5%，原生75%；MLP+BC对原生PDI仍+44.00%、optimize墙时+48.90%，不能宣称胜过原生。Actor实际调用加权成本26.358→1.799ms（−93.17%），但对B0的optimize墙时仅−3.72%。n3seq24无Actor，贡献完整方案约11.8%的均值改善，不归为神经决策；共同Actor PDI改善仍成立。
- 最终材料`docs/methods/tgppo_light_actor_20261105/MEETING_BRIEF_20261106.md`与`continuation/{RESULTS_REVIEW,DEVELOPMENT_REVIEW,CROSS_INSTANCE_REPORT}.md`。8实例为已查看过的探索对象；本轮120秒/100episode、两训练种子、结构CUDA初始流未完全对齐，边界保留。当前只是BC初始化＋PPO，没有持续BC/CL损失。
- 下一步仅建议现存MLP BC-only vs BC+PPO：neos-1215259/neos-512201，2训练种子×2solver×2条件=16模型＋4新原生，4单线程CPU、累计20分钟，无GPU/训练。它是已观察正反例的机制核对，尚未授权/实现/执行。当前阶段结束等待验收；不因剩余预算自动继续，不改现有权重或冻结训练源码，不重启CR/66全量。

## 2026-11-05 17:42 开发重测完成、条件式跨实例启动

- Silver4214追加统一评估170/170有效（160模型＋10新原生），累计4781.521/5400秒，含前次存储受阻尝试；全部冻结权重未改、CUDA未初始化。旧Gold6148的108项有效记录、失败、2822.625秒预算保留。
- 既定PLAN筛选通过，选MLP+BC；相对B0开发PDI−5.50%、共同Actor四实例−16.37%、证明率40%→60%、Actor17.268→1.707ms。BC相对同结构MLP全体−2.92%、共同Actor−6.39%。对原生PDI−1.36%，但证明率60%<70%、平均optimize墙时+5.14%，不能宣称全面胜原生。
- roi5alpha10n8全模型无Actor且PDI仍变化，约20.5%的MLP+BC−B0完整均值改善来自该实例，不能归因于神经决策。新`continuation/RESULTS_REVIEW.md`给出独立增量与计时贡献；最终原始表和筛选在同目录`DEVELOPMENT_REVIEW.md`、`decision.json`。
- 控制器已按既有条件授权启动`tgppo/runs/light-actor-20261105/cross-instance-8`：B0/MLP×无/有BC、训练种子11/22、固定8探索实例×solver0/1，128模型＋16新原生。沿用单阶段90分钟/22:00截止，无自动重试；当前PID/状态请查真实进程及status，不重复启动。CPU缓存、真实概率/动作fork一致性验证通过，源码与权重只读，无GPU/训练。
- 本阶段继续到跨实例完整或到限后交付并停止，不自动追加CL、更多模型、训练或66全量。

## 2026-11-05 本地同机重测追加授权与存储恢复

- 原Gold6148统一评估在SET+BC-s11两个roi任务watchdog后退出：108有效、52缺失，原90分钟已用2822.625秒。不能把部分批次或其他Actor的roi结果填成完成。
- 用户已明确选择“本地CPU追加90分钟统一重测（推荐）”。新RUN为`tgppo/runs/light-actor-20261105/cpu-recovery-20261105`：160模型+10新原生，独立Silver4214 cohort；旧Gold结果/预算原样保留。只需10开发原生，不做75训练归一化；无GPU/训练。
- 第一次本地恢复因Ceph共享盘元数据等待，4个spawn原生worker在180秒时CPU各仅约1.4秒，已停止。新90分钟预算实际用1067.476秒、剩4332.524秒；后续尝试沿用该预算，不清零。程序准备、导入、验证和求解都记入运行尝试；尝试间的只读分析/代码准备不冒充求解耗时。
- 独立CPU恢复入口`docs/methods/tgppo_light_actor_20261105/continuation/cpu_refinalize_cached.py`，只改进程/缓存调度：本地缓存逐位等值的MPS/Actor，单线程且无CUDA的CPU父进程，实际四种Actor概率/动作一致性通过后才fork；保持冻结worker/SCIP/指标/种子，170任务交错。当前状态与预算以新RUN及同目录`continuation/progress.json`为准。不要再启动原Gold finalize或另起预算。
- 新的条件式监控器`continuation/continue.py`仅等待现有CPU评估，完整后运行`decide.py --results <新RUN>/results`；满足原PLAN条件才自动冻结候选做144项CPU跨实例（最多另90分钟，仍22:00前），不追加训练。进程/状态和失败应先核查，不能单凭旧running标志启动重复任务。

## 2026-11-05 收齐评估与条件式跨实例确认授权

- 用户明确要求先收齐现有评估，分别检查结构/BC及相对relpscost的PDI、证明率、部署成本；若收益跨种子、跨实例成立，再冻结候选做小规模跨实例确认。之后要求从中断处继续。该授权覆盖下面旧快照中“跨实例仅提案、需另行验收”的限制，但不授权GPU训练、修改冻结模型或重启CR。
- 中断没有停止远端评估或本地只读汇总；15:36北京时间已提交60/160项。沿用原统一评估90分钟累计上限，不重置、不重复finalize。
- 筛选规则与有条件的下一步在`docs/methods/tgppo_light_actor_20261105/continuation/PLAN.md`，在读取轻量结构求解数值前写下（部分任务当时已执行，不冒充运行前预注册）。至少2个共同Actor实例、两个训练种子、PDI/证明率/成本均核对；规则不因输赢放宽。
- 仅通过后冻结一个结构的完整2×2与两种子，在固定8个跨实例探索对象上做128模型+16新原生=144项，当前Silver4214本地CPU、最多4个单线程worker，新增执行累计墙时上限90分钟、北京时间22:00前截止，无自动重试。若无候选通过或现有评估不完整，新增跨实例求解为0，交付报告后停止。
- 原生开发参照实际复用server6缓存，匹配Gold6148型号、软件和完整求解身份，并非本轮同步重测；报告保留该边界。跨实例若启动，则在当前本地CPU新测16原生，与开发环境绝对时间不混合。原权重/代码/旧结果只读。

## 2026-11-05 轻量 Actor 训练结果审阅（15:20北京时间快照）

- 用户报告seed=11完成，实际共享盘已核实16/16组（含seed=22）全部100 episode，24/25轮有效PPO更新，无末尾待更新transition；同型号Gold6148/V100-SXM2-32GB。旧结果和冻结代码/权重不改。
- 正式BC全部1000步。MLP/SET两个种子内部选模NLL约0.83–0.85；ISAB约0.90/0.85；B0约1，B0-s11选回0步，与无BC初始化完全相同。这不是求解或测试集收益。
- 新增本地CPU只读权重前向约55.96秒（含启动），与原准备合计144.85/600秒；288个既有内部留出状态上，PPO后MLP+BC/ISAB+BC保留专家信号，SET有回退但仍优于其无BC条件。没有新增训练/GPU/SCIP求解。
- 目标GPU记录暴露CUDA流边界：Critic初始化、CPU Torch流同种子一致，同结构有/无BC的CUDA流一致，但B0与轻量结构之间不同。机制为`fork_rng(devices=[])`内`torch.manual_seed`未恢复CUDA流；今后在新副本修复，不能声称当前结构对照完全对齐GPU随机流；不为此自动重训或修改当前冻结代码。
- 原最后完成的Custom于15:08北京时间启动了已授权的16模型统一开发评估（共享90分钟上限），当前仍在产生结果。不能重复启动finalize。新增只读审阅位于`docs/methods/tgppo_light_actor_20261105/results_review/`，报告注明动态评估进度，汇总进程只读已有输出；不会提交作业或新增求解。完整求解结论以该目录REPORT及原RUN/results/complete.json为准，不将进行中的评估写成已完成。
- 当前不申请额外GPU、不追加CL/训练/66测试集；完成本轮报告后等待验收，跨实例等仅为下一阶段提案。

## 2026-11-05 三种轻量 Actor 实施授权

- 用户明确修订“只选一种轻量结构”：授权先复用合法专家数据验证 BC，再实现三个推荐的轻量 Actor，准备并行 Custom PPO 训练；只在需要申请作业时提醒用户手动启动。此授权覆盖上一条调研后等待验收及单结构筛选限制，当前推进至作业交接，不自行提交 GPU。
- 新独立副本 `tgppo/experiments/light-actor-20261105`：共享树条件化 MLP、Deep Sets 式集合 Actor、单 ISAB；保留原 Actor 的无/有 BC 对照，结构与预训练分别比较。小 Transformer 仅作离线参照，CL/历史/树信用分配不混入首轮。旧基线、CR 权重/结果、Mamba 采集与训练目录只读。
- 本地 CPU 完成数据兼容性核查、BC 小试、延迟、合成 PPO/恢复检查；计算沿用600秒累计上限进行控制。真实 on-policy PPO 留给用户 Custom（100 episode、种子11/22为本轮准备范围），不在本地新增 SCIP 求解。Custom 按实际 CPU 指纹复用/补测归一化，CPU型号不限制，GPU型号只记录、不硬锁V100。
- 用户要求尽快利用夜间空闲 GPU；优先交付可运行的少量入口与数据，不扩展无关框架。准备完成时报告实测初验、作业数、执行时间与排队时间边界，再等待用户运行反馈。

### 三种 Actor 的本地初验与作业交接

- 已在独立副本实现 MLP/SET/ISAB（24493/32749/92205参数）；12个合法训练实例拟合1468状态，另3个TGPPO-train实例留出288状态。原15实例交集之外的4个TGPPO-dev实例不用于拟合。原始60份专家文件SHA、字段与标签映射已核对；SCIP版本、cutoff和两项计数分母差异作为离线迁移边界保留。
- 400步CPU BC的留出一致率43.75%/44.44%/47.22%，随机参考8.35%；留出归一化NLL0.9233/0.9701/1.0135，三个都未通过此前建议的0.90筛选门槛。按用户三个结构探索范围继续准备PPO，不将门槛改写为已通过；正式BC统一1000步、固定内部留出里程碑选初始化，规则在GPU提交前冻结。
- 原Actor/Critic/Agent/H3源码不改，公共Critic/RNG对齐、动态候选、padding/等变/梯度、CPU合成PPO更新/恢复、BC中断与选模等检查通过。具体产物在`docs/methods/tgppo_light_actor_20261105/REPORT.md`及新RUN/offline；没有新增SCIP求解、真实PPO episode或GPU启动。
- Custom入口为`tgppo/experiments/light-actor-20261105/custom_job.sh`，指南同目录`docs/CUSTOM_JOBS.md`。8条件×种子11/22=16作业；可先8个种子11，第二种子8项并行。各100episode、累计140分钟（含BC、归一化和桥接），统一CPU评估累计90分钟，总上限39作业小时、北京时间11月5日22:00截止；不是沿用已结束CR矩阵的16小时预算。
- 原生归一化按完整CPU/软件身份复用，型号不限；CUDA实际检查在用户提交作业后才执行。全16项终态自动统一开发评估；若仅8项，显式`finalize --seeds 11`输出单种子结果。没有授权自动访问66测试集、追加CL/历史/树信用分配或延长预算。

## 2026-11-05 用户确认的主方法与本轮调研

- 用户认可前次分析，确定专家预训练作初步验证；主线转为“轻量 Actor＋有监督约束的表示学习＋PPO”，分别验证轻量结构与专家监督/对比预训练的独立增量。保留现有候选25/节点8/MIP53廉价特征，不只考虑缩小 Transformer。
- 树信用分配后置，先解决全局 PDI 目标适配和未完成子树处理；CR 线仍暂时终止。可以复用已发现的省成本数据入口，但须核实划分与语义兼容，不能把旧专家数据当作当前 on-policy PPO 轨迹。
- 本轮仅开展论文调研与方法设计，交付 `docs/methods/tgppo_light_actor_survey_20261105/REPORT.md`、文献台账和方法汇报摘要。优先共享 MLP/集合汇聚 Actor，ISAB 为第二候选，小 Transformer 为公平结构参照；这是待验证排序，没有新训练/求解结果。
- 下一阶段建议先做现有数据兼容性＋本地 CPU 离线筛选（累计单线程计算600秒为提案）；本阶段交付后等待验收，不自动启动 GPU/PPO、新增 SCIP 求解或历史模型改造。原源码、权重与封存报告保持不改。

## 2026-11-05 用户对改进方向的修订

- BC/专家辅助学习只作低成本初步验证，不能单凭增加预训练视为足够的方法贡献；获得收益后再优化。后续方法设计重点考虑轻量 Actor，保留现有 TreeGate 的 25/8/53 维特征体系，不重新引入完整二部图。
- 神经/原生切换分类器优先级下调，不作为当前主要效率路线。树信用分配需要先解释目标、依据、实现变化和风险，不能把子树节点目标当作全局墙时 PDI 的等价替换。
- 本轮是答疑与方案修订，没有授权启动新训练/求解。说明见 `docs/methods/tgppo_post_cr_20261105/TREE_CREDIT_CLARIFICATION_20261105.md`；原调研报告及其核验记录保留。

## 当前优先指令：CR 研究线暂时终止（2026-11-05，北京时间）

- 用户明确要求“不再继续 CR 扩大训练，这条研究线暂时终止”。停止该线后续训练、调参和追加评测；保留六份冻结权重、全部原始记录和封存报告，不将跨实例阶段的历史建议视为继续授权。
- 当前任务仅为结合既有诊断、阅读相关高质量论文、提出原 TGPPO 的其他改进建议。调研报告位于 `docs/methods/tgppo_post_cr_20261105/REPORT.md`。其中的新实验均为提案，未执行；不改原基线、不自动申请 Custom/GPU 或新增 SCIP 求解。
- 后续优先比较专家辅助学习、奖励目标、部署成本和树信用分配；不以新名称重启候选残差/读出头扫描，不直接引入 HSTU。阶段报告交付后等待验收。

先阅读 `migration/2026-10-22/CODEX_HANDOFF.md` 和 `docs/阶段交付与验收_TGPPO复现与增量改进.md`。

- 用户要求逐阶段交付并等待明确验收。阶段 1—4 已验收；阶段 5 框架与桥接完成，但完整主矩阵中断，尚未完成/验收；阶段 6 未启动。
- 先核查实际进程和已提交 checkpoint；旧 running 标志不是任务正在运行或已经完成的证据。
- 保留原版 repo、历史 worktree、实验原始日志和哈希台账；新实验使用独立目录。
- 研究主线在 `tgppo/experiments/stage5`；自定义绑定来自 `tgppo/experiments/stage4/metrics_binding`。不要用原版启动脚本代替修复研究主线。
- Actor-only 部署；新增信息仅给 Critic。原单位 value/GAE，只有价值损失内部标准化。
- 原预算、数据、奖励、损失及硬件使用规则见冻结配置。迁移不是授权扩大预算或启用 GPU 在线训练。
- 大规模正式实验前核对服务器环境、路径、绑定接口、CPU 配额和基线归一化；明确新的实验配置。
- Codex 会话恢复不会恢复已退出的计算进程。矩阵恢复入口和预算记账需要核实，不能直接删除旧状态后重跑冒充续接。
- 用户本轮关注迁移准备；未提供服务器环境时不要声称迁移已在目标机器验证成功。

## 2026-10-22 最新授权与迁移入口

- 用户随后明确选择独立的服务器正式协议：B0、R0、Rjoint 各两个训练种子（11、22），共六个 GPU 作业；各 500 个完整 episode，每 100 保存，按同一开发集规则分别选权重。这一授权覆盖上面的“迁移本身不授权扩大预算”限制，但只覆盖此明确范围；旧阶段 5 中断矩阵仍是历史证据，不算新实验续接。
- 本轮服务器代码在 `tgppo/experiments/server-20261022`，入口与冻结新协议在 `migration/2026-10-22/server-run/`。先读该目录 README 和 `migration/2026-10-22/HANDOFF_SERVER_2026-10-22.md`。定制绑定仍来自 stage4。
- 源机本地 50 项检查通过；目标 V100、Slurm、500 episode 耗时没有验证。服务器上先重建环境、补实例数据、核查哈希、跑原生基线和 GPU 桥接，再启动已授权六个作业。不得把这些准备步骤或脚本文件当作训练结果。
- 本轮直传压缩包包含 25 个训练/开发及 66 个测试 `.mps.gz` 实例、源码、协议和文档；旧模型、轨迹及求解器二进制不包含。后续公开 GitHub 仓库可能采用较轻的收录范围，应以各自清单为准。不要因为交接包中没有旧 `latest.pt` 就宣布旧矩阵已完成或清零预算。
- 开发集可选里程碑，未见测试集只能在选模冻结后使用。阶段 5 完成并汇报后继续等待用户验收，阶段 6 未启动。

## 2026-11-02 资源与执行方式

- 用户明确要求：无需 GPU、预计几秒至几分钟、现有环境可承载的已授权 CPU 离线实验，默认直接在本地完成；不要让用户为此申请 Custom 作业或 V100。
- 先用实际规模和已有计时判断资源需求；脚本、检查和文档投入应与实验成本相称。确需长时间运行、额外算力或用户明确指定时，再准备外部作业。
- 这不扩大实验范围或取消阶段验收；整理已完成结果时不自动启动下一阶段训练或求解。

## 2026-11-04 方法实验授权

- 用户批准 `docs/methods/tgppo_candidate_residual_20261104/PLAN.md` 并要求立即自动化执行；GPU训练由用户申请多个Custom并行启动，本地完成分钟级CPU准备。
- 独立目录 `tgppo/experiments/candidate-residual-20261104`：原B0、评分捷径、候选残差三组×训练种子11/22，最多各100完整episode，120秒，原Critic/H3/PPO；不改旧基线、不启动HSTU或66测试集。
- 同预算episode100为主；完整矩阵无法达到100时才统一使用预定义20里程碑。新增结果以方法和真实求解对照交付，系统瓶颈核验不计作方法成果。
- 并行GPU入口累计各140分钟×6，统一CPU评估累计60分钟，本地准备计算10分钟，合计15小时10分钟，小于批准方案16小时总作业墙时；恢复不清零预算。北京时间11月5日23:00冻结材料，训练提前1小时结束。
- 用户随后明确放开训练CPU型号限制，优先尽快跑完；`candidate-residual-20261104/cpu-flex`提供兼容入口，原命令自动转入。保留运行中的B0-s11，不修改训练Python/配置/checkpoint协议；归一化按实际CPU/软件分别缓存，最终统一同机评估，原失败记录和预算保留。

## 2026-11-04 方法实验交付状态

- 用户报告六Custom已结束，实际核对六组均100完整episode/25轮PPO批更新，固定episode100；60项模型与10项原生同机开发评估齐全。复核报告和周五汇报位于`docs/methods/tgppo_candidate_residual_20261104/{RESULTS_REVIEW,MEETING_BRIEF_20261106}.md`。
- 全体PDI的CR−B0表面改善2.70%，95.6%的均值差来自未调用Actor的roi5alpha10n8，不能归因于神经决策；Actor参与子集CR相对S方向随训练种子变化。dcmulti有局部平均节点/时间收益、solver重复方向不同，vpm2退步。不得宣称整体改进已经验证。
- 原始自动结果封存保留，新增只读分析在上述方法目录的`review/`。训练代码/配置和旧基线未改。种子22的训练硬件异构必须披露；最终同机评估不消除训练混杂。
- 本阶段已交付、等待明确验收。下一阶段冻结6个Actor、对dcmulti/vpm2增加solver seeds2/3的28项本地CPU复测仅为提案，尚未授权或执行；不自动训练500episode或访问66实例。

## 2026-11-04 冻结权重CPU复测

- 用户已明确验收前阶段，授权上述28项本地CPU复测，累计墙时上限600秒、不追加GPU训练。该授权覆盖前条“尚未授权”的旧状态；现已实际完成28/28，执行累计257.800秒，含一次字段核查错误的11.445秒尝试，保留结果及预算后继续，未重复求解。
- 独立运行目录`tgppo/runs/candidate-residual-20261104/cpu-retest-seeds23`；报告及脚本在`docs/methods/tgppo_candidate_residual_20261104/cpu_retest/`。6个episode100 Actor与完整checkpoint哈希前后不变，源码/配置、旧基线、原始主矩阵产物不改。
- 新solver seeds2/3上，dcmulti的CR相对B0平均PDI−19.52%、节点−65.63%、SCIP时间−54.49%，对B0和S四个配对均改善；vpm2的PDI相对B0+1.92%、相对S+3.46%，跨训练种子不一致。只能支持dcmulti局部收益，不能推为整体改进或未见实例泛化。
- 本地E5-2620 v3单核与原评估E5-2680 v4不同，旧新cohort不混合绝对时间；训练硬件混杂仍需披露。无CUDA初始化、无优化步，无剩余预算追加。本阶段交付待验收；不得自动启动新实例评测、GPU训练、奖励或架构变体。

## 2026-11-04 跨实例探索授权

- 用户随后明确要求冻结当前模型，继续小规模跨实例评估；仅在必要时申请多个Custom。此授权覆盖前条不得启动新实例评测的旧限制，不授权新训练或完整66实例矩阵。
- 本地4.01核/32GiB，采用4个单线程CPU worker；六个episode100 Actor不变。`docs/methods/tgppo_candidate_residual_20261104/cross_instance/PLAN.md`与运行目录`tgppo/runs/candidate-residual-20261104/cross-instance-8/cohort.json`已在评估前冻结。
- 从已分析的66实例按历史原生求解成本分四层，每层取固定盐哈希排名前2：bell5、bell3a、neos-1215259、neos-612125、sp98ir、neos-512201、n3seq24、bc1。不因本轮结果更换；不称全新保留集确认。
- 8实例×2solver种子×6模型=96项，原生16项，共112。120秒原协议，单项watchdog180秒，整个4worker执行经过墙时上限90分钟（另计worker累计墙时），完成/到限后交付停止。当前已启动，本地会话PID和原始任务结果见运行目录；不重复提交Custom。

### 跨实例阶段完成记录

- 112/112有效完成，经过墙时2381.799秒（39分41.8秒），worker墙时合计8978.304秒（2.494小时）；4worker本地完成，无GPU/Custom，无训练、权重更新或额外重试。
- 完整8实例PDI：原生1369.26，B01987.13，S1972.10，CR1970.80；CR相对B0−0.82%、相对S−0.07%，学习组三组证明率均37.5%、原生75%。CR相对S两训练种子方向不同，全局没有稳定优越。
- n3seq24所有模型任务无Actor，不能归因于评分；参与分解7实例CR相对B0/S在两训练种子均小幅改善，仍保留完整主表。新正例neos-612125、bc1，两训练种子均值均改善；bell3a均退步。新增局部证据，不宣称显著/广泛泛化。
- 报告和更新汇报稿：`docs/methods/tgppo_candidate_residual_20261104/cross_instance/{REPORT,MEETING_BRIEF_20261106}.md`。原始主矩阵、前次交付及原基线保留。本阶段已停止、等待验收；后续独立确认数据、同硬件训练重复、全66或新方法均未自动启动。
