# 作者正式权重与最终训练设置检索

日期：2026-11-08。只读公开检索；未联系作者、发布issue、训练、求解或修改现有作业。

**结果：没有取得作者正式权重，也没有取得与论文结果明确对应的完整最终训练配置。可以取得论文超参数、公开源码、示例脚本和依赖表；不能把示例配置称为作者实际最终配置。**

| 检索入口 | 实际结果 |
|---|---|
| [官方仓库分支](https://github.com/benmhamed-a/tgppo-aaai-26/branches) | 仅main；本次HEAD为`4f90ccddd7999779c93ff2425ca8dd056113a523` |
| Git标签 | 连接器不支持该列表端点，改用只读`git ls-remote --tags`，成功退出且结果为空 |
| [Releases](https://github.com/benmhamed-a/tgppo-aaai-26/releases) | 返回空，没有可下载的发布附件 |
| [公开提交历史](https://github.com/benmhamed-a/tgppo-aaai-26/commits/main/) | 16个提交的完整文件树均读取成功、未截断；未发现权重文件或最终参数JSON，包括曾发布后删除的此类路径 |
| [Pull requests](https://github.com/benmhamed-a/tgppo-aaai-26/pulls) | 全状态查询返回空 |
| [论文源码压缩包](https://github.com/benmhamed-a/tgppo-aaai-26/blob/main/crc/AAAI_26_Camera_Ready_8_Pages.zip) | 本地副本Git blob与在线树相同；7个文件，仅LaTeX／参考文献／图表，没有模型或运行配置 |
| 论文与README外部入口 | 指向官方代码、论文和T-BranT数据；未定位到独立的TGPPO模型下载入口 |
| Hugging Face／Zenodo等公开搜索 | 按TGPPO及论文题目检索，未找到能确认来自作者的权重或最终运行包；搜索未命中不等于绝不存在 |

已知的公开设置见[复现一致性复核](REPRODUCTION_ALIGNMENT_REVIEW.md)。仍需要作者确认的关键问题：

1. 对应论文结果表的最终Actor权重，若方便再提供Critic及checkpoint；同时给出加载／评估命令。
2. 实际完整启动命令、最终`best_hparams_aggregate.json`、代码commit，以及500episode／3600秒／solver0–4的运行记录。
3. 最终实验究竟逐episode更新，还是合并多条轨迹后更新；公开Agent的train/eval、GAE与终止处理是否为实际使用版本。
4. 全局初始化种子、PyTorch与SCIP／定制绑定版本；最好提供一个固定状态及对应概率用于接口核对。

这些只能向作者或其已发布的运行产物确认，不能由我们推断填补。已准备[英文询问草稿](AUTHOR_REQUEST_DRAFT.md)，尚未发送。原始检索证据保存在[AUTHOR_ARTIFACT_SEARCH.json](AUTHOR_ARTIFACT_SEARCH.json)。
