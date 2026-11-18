# 作者询问草稿（未发送）

可用于官方仓库issue或发送给已核实的作者联系方式。没有推测作者邮箱，也没有代用户联系任何人。

**Subject: Request for TGPPO checkpoint and final experimental configuration**

Dear TGPPO authors,

Thank you for sharing the code for “Learning Branching Policies for MILPs with Proximal Policy Optimization.” I am reproducing the method and would appreciate the following artifacts or clarifications:

1. The trained Actor checkpoint used for the reported results, together with its loading/evaluation command. A full Actor–Critic checkpoint would also be helpful if available.
2. The final training command, code commit, and `best_hparams_aggregate.json`. Sections 4.3/5.3 describe 500 complete episodes, solver seeds 0–4, and a 3600-second limit, whereas the public example script uses 600 seconds, seed 0, and 12 iterations. Which configuration produced the reported results?
3. Whether PPO updates followed each episode or pooled multiple trajectories, and whether the released Agent's rollout-eval/update-train modes, GAE, and episode-end handling were used unchanged in the final experiments.
4. The initialization seed and exact solver/custom PySCIPOpt versions. A sample state with reference action probabilities would help verify the feature interface.

In our local runs with the public Actor architecture, probabilities are nearly uniform on several stored states. Our training wrapper includes local changes, so we have not attributed this observation to the published method. A reference checkpoint would help distinguish interface or training differences from architectural effects.

If these artifacts are already available elsewhere, a link would be greatly appreciated.

Best regards,

[Your name]
