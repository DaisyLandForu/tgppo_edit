# MLP-seed11续训作业：容器启动前FailedMount

用户截图：作业`sg-duweiyue25-261111-300b7`、Pod `sg-duweiyue25-261111-300b7-default0-0`仍“启动中”，未显示开始时间。kubelet重复报告`FailedMount`：`/var/lib/lxc/lxcfs/proc/{swaps,stat,uptime,cpuinfo}`的hostPath File类型检查失败，`is not a file`。

这是宿主机文件挂载前置条件失败，训练入口尚未执行。LXCFS提供容器可见的虚拟/proc文件，Kubernetes的hostPath File要求宿主机已有对应文件。可能与断电后LXCFS服务/FUSE挂载未恢复有关，但必须由平台检查节点才能确认；不能据截图断言具体服务状态。

只读核对：MLP-seed11 CPU continuation仍已提交163/500，latest为58,115,420字节，mtime北京时间11月11日00:40:07；state仍为旧SIGTERM，预算仍5个attempt、最新日志仍11月9日启动的旧CPU任务，没有本截图作业ID对应的新训练日志；执行锁空闲。该挂载失败未推进或覆盖训练检查点。未重提、取消或启动任何训练，未改冻结脚本。

处理：用户取消卡在启动中的该作业，联系平台换健康CPU节点／修复当前节点LXCFS，保持4CPU、64GiB、0GPU与原续训命令。若重提仍落到相同故障节点，不应继续反复提交，需平台处理节点健康。不要在训练脚本中创建空/proc文件或绕过平台挂载检查。

```bash
bash /home/duweiyue25/tgppo-transfer/research/tgppo/experiments/paper-resource-resume-20261109/custom_job.sh run --experiment aligned --variant mlp --seed 11 --device cpu --ram-gib 64 --resume --no-total-wall-limit
```

可转发管理员：

> 作业sg-duweiyue25-261111-300b7的Pod反复FailedMount，/var/lib/lxc/lxcfs/proc/cpuinfo、stat、uptime、swaps均报hostPath type check failed / is not a file。请检查所在节点LXCFS服务与FUSE挂载、恢复这些虚拟文件，或将作业重新调度到健康节点。容器尚未启动，训练脚本还未执行。

参考：[Kubernetes hostPath类型检查](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath)、[LXCFS官方说明](https://github.com/lxc/lxcfs#introduction)。
