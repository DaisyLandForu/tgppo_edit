# 200轮CPU评测准备交付

状态：已准备、未启动求解。用户确认已有四份固定200＋relpscost，完整66、solver0，共330；四个并行4CPU／32GiB／0GPU Custom，分片85／85／85／75。

四份source Actor与对应完整episode-0200实际加载，通过配置／episode／variant／seed／预训练与checkpoint provenance核对，Actor逐张量与完整checkpoint相等；只读副本已保存并核对SHA。原始200权重、latest与旧100结果均未改。

本地13项无Torch的合成调度／覆盖／恢复／统计检查通过，特别包含seed22动作流拒绝错误seed11、跨环境不拼接面板、缺项不排名、不做跨训练种子结构归因和全330汇总。真实四Actor CPU前向与fork子进程的概率／动作逐位一致，单线程、未初始化CUDA，未构造Critic或执行SCIP optimize。结果见包内ACTOR_CPU_CHECK.json与CHECKS.json。

第一次prepare在完成全部模型冻结、写plan之后，于末尾全量输入核验处达到180秒本地外层限额（exit143）；已保留副本与plan，没有求解或训练。随后独立无Torch重复完整源码／输入／模型哈希核验通过（约11.54秒），不是修改参数重跑训练。该180秒仅为本地准备命令保护，不进入评测的3600秒搜索协议。

CLI及原始结果目录与100轮、500正式入口均独立；原旧49／新53／recovery6／resource6文件受保护。全部330项有效配对后自动生成主表、同种子增量、Actor参与分解与作者固定easy33／hard32／未分组1例SGM；不齐仅进度，不假定本轮有收益。

用户手动停止训练并申请评测；本会话没有停止／提交作业，没有真实PPO／GPU或SCIP求解。冻结源与哈希详见新包plan.json，当前训练快照在TRAINING_SNAPSHOT.json，使用说明在新包docs/CUSTOM_JOBS.md。
