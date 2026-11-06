# Cutoff 参考值核查

MILP-S 的 8 个测试实例 cutoff 已重新独立核验，不仅依赖历史 TGPPO 文件中的 `status=optimal` 标签。7 个值同时匹配 [MIPLIB 2017 v35](https://miplib.zib.de/downloads/miplib2017-v35.solu) 和 [v37](https://miplib.zib.de/downloads/miplib2017-v37.solu) 的已知最优记录；neos11 匹配 [CORAL 的上下界 9 / 9](https://coral.ise.lehigh.edu/wp-content/uploads/mip-instances/Bounds.html)。原文件副本位于 `sources/`，SHA256 和逐实例证据位于 `setup/cutoffs-audit.json`。

| 测试实例 | cutoff |
|---|---:|
| map18 | -847 |
| mine-166-5 | -566395707.9 |
| neos11 | 9 |
| neos18 | 16 |
| ns1830653 | 20622 |
| nu25-pr12 | 53905 |
| rail507 | 174 |
| seymour1 | 410.76370139 |

核查时发现旧参考文件 `research/tgppo/datasets/cutoffs.json` 将训练实例 misc03 的 cutoff 记作 2，而本轮两次不设 cutoff 的原生求解均以 primal=dual=3360、status=optimal 结束。对应报告为 `runs/collection-pilot/20261027T143801.212377Z/report.json` 和 `20261027T143804.396946Z/report.json`。

这项不一致已记录，**没有修改 TGPPO 文件或历史实验**。本轮 Mamba 采样按论文不提供 cutoff，且 misc03 不属于这 8 个测试实例，因此该旧值没有用于本轮采样或测试配置。要判断它是否影响历史 TGPPO 结果，需另行核对各次运行实际加载的配置、数据和日志；本轮没有完成该审计，不能据此直接给历史结果下结论。
