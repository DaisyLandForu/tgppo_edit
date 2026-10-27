#!/usr/bin/env python3
"""Original explanatory schematic of the TGPPO paper, not runtime architecture.

Source: Learning Branching Policies for MILPs with Proximal Policy Optimization,
sections 3--4 and Figure 1. The paper's shared-front-end description is explicitly
distinguished from the independently instantiated networks in the audited code.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_figures import box, arrow, save, BLUE, GREEN, ORANGE, GRAY

fig,ax=plt.subplots(figsize=(16.5,7.2))
ax.set(xlim=(0,18),ylim=(0,8.1));ax.axis('off')
ax.text(9,7.72,'TGPPO paper: learn branching decisions from solver returns',
        ha='center',fontsize=17,fontweight='bold')
box(ax,.35,4.95,3.55,1.2,'SCIP observation\n25 features / candidate\n61 features for tree context',fs=10.5)
box(ax,4.55,4.95,4.1,1.2,'Front-end in paper description\nEmbeddings + Transformer\nTree-candidate matching',fs=10.5)
box(ax,9.35,4.95,3.25,1.2,'Actor: TreeGate head\nProbability over candidates\nSample a branching variable',BLUE,'#E3EEF9',10)
box(ax,13.35,4.95,4.1,1.2,'SCIP executes the branch\nSolve relaxations; update\nbounds and search tree',fs=10.5)
box(ax,9.35,2.55,3.25,1.2,'Critic: value head\nPredict future cumulative\nreward from this observation',GREEN,'#EDF6F1',9.6)
box(ax,13.35,2.55,4.1,1.2,'Trajectory buffer\nObservations, actions, rewards\nold probabilities and values',ORANGE,'#FFF4E9',10)
box(ax,4.55,1.05,4.1,1.2,'PPO training update\nGAE estimates advantages\nUpdate policy and value function',GREEN,'#EDF6F1',10)
box(ax,.35,2.55,3.55,1.2,'H3 shaped reward\nNode efficiency, gap / PDI,\nsearch progress + final status',ORANGE,'#FFF4E9',9.8)
arrow(ax,[(3.9,5.55),(4.55,5.55)],BLUE)
arrow(ax,[(8.65,5.55),(9.35,5.55)],BLUE)
arrow(ax,[(12.6,5.55),(13.35,5.55)],BLUE)
arrow(ax,[(8.65,5.55),(8.98,5.55),(8.98,3.15),(9.35,3.15)],GREEN)
arrow(ax,[(10.975,4.95),(10.975,4.2),(14.1,4.2),(14.1,3.75)],BLUE)
arrow(ax,[(12.6,3.15),(13.35,3.15)],GREEN)
arrow(ax,[(17.45,5.55),(17.75,5.55),(17.75,7.0),(2.125,7.0),(2.125,6.15)],BLUE)
ax.text(9,7.12,'Continue the next branching decision until proof completion or time limit',ha='center',fontsize=9.7,color=BLUE)
arrow(ax,[(15.4,4.95),(15.4,4.47),(2.125,4.47),(2.125,3.75)],ORANGE)
arrow(ax,[(3.9,3.15),(4.2,3.15),(4.2,2.4),(15.4,2.4),(15.4,2.55)],ORANGE)
arrow(ax,[(15.4,2.55),(15.4,1.65),(8.65,1.65)],GREEN)
arrow(ax,[(6.6,2.25),(6.6,4.95)],GREEN,dashed=True)
ax.text(6.82,3.35,'updated\nparameters',fontsize=9,color=GREEN)
ax.text(9,.52,'This figure explains the PAPER: its front-end is described as shared. '
        'The audited official code creates separate Actor / Critic networks.',
        ha='center',fontsize=9.5,fontweight='bold')
ax.text(9,.13,'No added frontier histogram, time branch, static value calibration or memory module is shown here.',
        ha='center',fontsize=10)
save(fig,'tgppo_paper_overview')
print('TGPPO_INTRO_FIGURE_WRITTEN')
