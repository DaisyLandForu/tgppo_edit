#!/usr/bin/env python3
"""Code-native diagrams of the actual server model and training protocol.

Reads existing audited aggregates; never imports or edits experiment runtime.
"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT/'tgppo/runs/server6-20261022/stage5-results-20261023'
plt.rcParams.update({'font.size':10, 'svg.fonttype':'none', 'pdf.fonttype':42})
BLUE, GREEN, ORANGE, GRAY = '#356A9A', '#357A64', '#AE6D2E', '#596673'


def box(ax, x, y, w, h, label, color=BLUE, fill='#EEF4FA', fs=10, weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.025,rounding_size=0.09',
                              linewidth=1.25, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#18242F',
            fontweight=weight,linespacing=1.35,zorder=3)


def arrow(ax, points, color=GRAY, dashed=False):
    for first,last in zip(points[:-2],points[1:-1]):
        ax.plot([first[0],last[0]],[first[1],last[1]],color=color,lw=1.3,
                linestyle='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,
                 color=color,lw=1.3,linestyle='--' if dashed else '-',zorder=1))


def save(fig, stem):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/(stem+'.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def architecture():
    fig,ax=plt.subplots(figsize=(16.5,12.2))
    ax.set(xlim=(0,18.1),ylim=(0,14));ax.axis('off')
    ax.text(9.05,13.72,'Actual Rjoint network: asymmetric inputs, independent parameters',
            ha='center',fontsize=17,fontweight='bold')
    for x,w,c in [(.2,7.7,BLUE),(8.2,9.6,GREEN)]:
        ax.add_patch(FancyBboxPatch((x,1.2),w,12.05,boxstyle='round,pad=.04',
                                  edgecolor=c,facecolor='none',linewidth=1.6))
    ax.text(4.05,12.88,'ACTOR  (training and deployment)',ha='center',color=BLUE,fontsize=13,fontweight='bold')
    ax.text(13,12.88,'CRITIC  (training only)',ha='center',color=GREEN,fontsize=13,fontweight='bold')
    box(ax,.65,11.35,3.25,.85,'Candidate variables\n'+r'$C_t\in\mathbb{R}^{N_t\times25}$')
    box(ax,4.2,11.35,3.25,.85,'Node 8 + MIP 53\n'+r'$x_t\in\mathbb{R}^{61}$')
    box(ax,.65,10,3.25,.85,'LayerNorm + Linear\n25 '+r'$\rightarrow$'+' 256')
    box(ax,4.2,10,3.25,.85,'LayerNorm + Linear\n61 '+r'$\rightarrow$'+' 256')
    box(ax,1.7,8.8,4.8,.85,'Broadcast tree + concatenate\nLinear: 512 '+r'$\rightarrow$'+' 256')
    box(ax,1.7,7.5,4.8,.85,'Transformer encoder\n5 layers, 8 heads, d=256, FFN=1024')
    box(ax,1.7,6.15,4.8,.9,'BiMatching: tree '+r'$\leftrightarrow$'+' candidates\n'+r'$N_t\times256$'+' refined features')
    box(ax,1.7,4.7,4.8,.95,'TreeGate head, one logit / candidate\n256 '+r'$\rightarrow$'+' 128 '+r'$\rightarrow$'+' ... '+r'$\rightarrow$'+' 2 '+r'$\rightarrow$'+' 1')
    box(ax,1.7,3.35,4.8,.85,'Mask padding + Softmax\n'+r'$\pi_\theta(a_t\mid o_t)$')
    box(ax,1.7,2,4.8,.85,'Categorical sampling\nSCIP branches on selected variable',fill='#E3EEF9')
    for x in [2.275,5.825]:arrow(ax,[(x,11.35),(x,10.85)],BLUE)
    arrow(ax,[(2.275,10),(2.275,9.65)],BLUE)
    arrow(ax,[(5.825,10),(5.825,9.65)],BLUE)
    for top,bottom in [(8.8,8.35),(7.5,7.05),(6.15,5.65),(4.7,4.2),(3.35,2.85)]:
        arrow(ax,[(4.1,top),(4.1,bottom)],BLUE)
    for y in [6.6,5.175]:arrow(ax,[(7.45,10.425),(7.68,10.425),(7.68,y),(6.5,y)],GREEN)
    ax.text(7.45,8.1,'tree\ncontext',ha='right',fontsize=8,color=GREEN)

    for x,label in [(8.65,'Candidate variables\n'+r'$C_t\in\mathbb{R}^{N_t\times25}$'),
                    (12.45,'Node 8 + MIP 53\n'+r'$x_t\in\mathbb{R}^{61}$')]:
        box(ax,x,11.35,3.35,.85,label,GREEN,'#EDF6F1')
    box(ax,8.65,10,3.35,.85,'LayerNorm + Linear\n25 '+r'$\rightarrow$'+' 256',GREEN,'#EDF6F1')
    box(ax,12.45,10,3.35,.85,'LayerNorm + Linear\n61 '+r'$\rightarrow$'+' 256',GREEN,'#EDF6F1')
    box(ax,8.65,8.8,3.35,.85,'Transformer encoder\n5 layers, 8 heads, d=256',GREEN,'#EDF6F1',9)
    box(ax,9.4,7.45,4.9,.9,'BiMatching: tree '+r'$\leftrightarrow$'+' candidates\n'+r'$N_t\times256$'+' refined features',GREEN,'#EDF6F1')
    box(ax,9.4,6.05,4.9,.95,'Masked mean over candidates + tree\nAggregation MLP: 512 '+r'$\rightarrow$'+' 256 '+r'$\rightarrow$'+' 256',GREEN,'#EDF6F1',9.4)
    box(ax,9.4,4.7,4.9,.9,'Add training-only context\n'+r'$z_{base}+W_{time}\tau_t+W_{extra}f(g_t)$',ORANGE,'#FFF4E9',10)
    box(ax,14.85,5.95,2.5,1.5,'Frontier input: 69\n8 x 8 histogram\n+ 5 scalars\nMLP 69'+r'$\rightarrow$'+'64'+r'$\rightarrow$'+'32\nProjection 32'+r'$\rightarrow$'+'256',ORANGE,'#FFF4E9',8.7)
    box(ax,14.85,4.7,2.5,.9,'Remaining time\n'+r'$\tau_t:1\rightarrow256$',ORANGE,'#FFF4E9',9)
    box(ax,9.4,3.35,4.9,.9,'TreeGate value head\n256 '+r'$\rightarrow$'+' 128 '+r'$\rightarrow$'+' ... '+r'$\rightarrow$'+' 2 '+r'$\rightarrow$'+' 1',GREEN,'#EDF6F1')
    box(ax,9.4,2,4.9,.85,'Raw-unit value\n'+r'$V_\phi=\mu+\sigma z$',GREEN,'#DCEEE5',11)
    for x in [10.325,14.125]:arrow(ax,[(x,11.35),(x,10.85)],GREEN)
    arrow(ax,[(10.325,10),(10.325,9.65)],GREEN)
    arrow(ax,[(10.325,8.8),(10.325,8.35)],GREEN)
    for top,bottom in [(7.45,7),(6.05,5.6),(4.7,4.25),(3.35,2.85)]:
        arrow(ax,[(11.85,top),(11.85,bottom)],GREEN)
    for y in [7.9,6.525,3.8]:
        arrow(ax,[(14.125,10),(14.55,9.8),(14.55,y),(14.3,y)],GREEN)
    arrow(ax,[(14.85,6.7),(14.65,6.7),(14.65,5.35),(14.3,5.35)],ORANGE)
    arrow(ax,[(14.85,5.15),(14.3,5.15)],ORANGE)
    ax.text(9.05,.68,'Actor and Critic do NOT share weights. Orange modules are absent from Actor inference.',
            ha='center',fontsize=11,fontweight='bold')
    ax.text(9.05,.26,'R0 removes the frontier branch; B0 also removes the time branch and static value calibration. '
            'No recurrent/Mamba module is used.',ha='center',fontsize=10)
    save(fig,'network_architecture')


def training():
    fig,ax=plt.subplots(figsize=(16.5,8))
    ax.set(xlim=(0,18.1),ylim=(0,9.5));ax.axis('off')
    ax.text(9.05,9.1,'Actual training and evaluation: complete CPU episodes, CUDA PPO, frozen Actor testing',
            ha='center',fontsize=15,fontweight='bold')
    for x,label,color in [(.45,'CPU rollout workers',BLUE),(6.3,'CUDA learner',GREEN),(12.15,'Checkpoint / evaluation',ORANGE)]:
        ax.text(x+2.5,8.5,label,ha='center',color=color,fontweight='bold',fontsize=13)
    box(ax,.45,6.9,5,.95,'20 training instances; paired seeds 11 / 22\n4 single-thread SCIP workers; 120 s / task')
    box(ax,.45,5.2,5,1.05,'CPU Actor samples actions; CPU Critic predicts V\nStore old log-prob, raw V, H3 reward, done\nRjoint also stores action-before-branch frontier',fs=9.2)
    box(ax,.45,3.5,5,1.05,'Finish 4 complete episodes; stable merge\nKeep zero-decision episodes in counts\nFewer than 256 transitions: retain buffer',fs=9.4)
    arrow(ax,[(2.95,6.9),(2.95,6.25)],BLUE)
    arrow(ax,[(2.95,5.2),(2.95,4.55)],BLUE)
    box(ax,6.3,6.9,5,.95,'Per-episode GAE in raw return units\n'+r'$\gamma=0.97,\ \lambda=0.92;\ R^\lambda=A^{GAE}+V_{old}$',GREEN,'#EDF6F1')
    box(ax,6.3,5,2.35,1.05,'Actor objective\nPPO clip = 0.16\nentropy = 0.003',GREEN,'#EDF6F1',9.2)
    box(ax,8.95,5,2.35,1.05,'Critic objective\nR0 / Rjoint:\n'+r'$((V-R^\lambda)/\sigma)^2$',GREEN,'#EDF6F1',9.2)
    box(ax,6.3,3.35,5,1.05,'Separate Adam optimizers; 3 epochs\nLogical batch 256; microbatch 16\nGradient clipping 0.5; full Critic trains',GREEN,'#EDF6F1',9.5)
    arrow(ax,[(5.45,4.025),(5.85,4.025),(5.85,7.375),(6.3,7.375)],BLUE)
    for x in [7.475,10.125]:
        arrow(ax,[(x,6.9),(x,6.05)],GREEN)
        arrow(ax,[(x,5),(x,4.4)],GREEN)
    box(ax,12.15,6.9,5,.95,'500 complete episodes per training job\nStable checkpoints at 100 / 200 / 300 / 400 / 500',ORANGE,'#FFF4E9',9.4)
    box(ax,12.15,5.2,5,1.05,'For EACH of the 6 models:\n5 development instances x 3 solver seeds\nChoose lowest instance-equal PDI',ORANGE,'#FFF4E9',9.4)
    box(ax,12.15,3.5,5,1.05,'Freeze all 6 actor_best.pt; retain actor_final.pt\n66 test instances x 3 solver seeds\nCPU Actor only, plus native relpscost',ORANGE,'#FFF4E9',9.4)
    arrow(ax,[(11.3,3.875),(11.7,3.875),(11.7,7.375),(12.15,7.375)],GREEN)
    arrow(ax,[(14.65,6.9),(14.65,6.25)],ORANGE)
    arrow(ax,[(14.65,5.2),(14.65,4.55)],ORANGE)
    arrow(ax,[(8.8,3.35),(8.8,2.45),(.15,2.45),(.15,5.725),(.45,5.725)],GREEN,dashed=True)
    ax.text(5.9,2.14,'Return updated Actor / Critic to the next CPU collection batch',ha='center',color=GREEN,fontsize=9)
    ax.text(9.05,1.25,'Offline diagnostic is separate: fixed Actor, shared Monte Carlo return labels, frozen encoder, '
            'value-head / context fitting.',ha='center',fontsize=10)
    ax.text(9.05,.78,'Formal online models start fresh; only train-only calibration statistics are reused. '
            'No offline-head warm start and no test-based selection.',ha='center',fontsize=10)
    ax.text(9.05,.26,'Evaluation disables Critic, frontier extraction, reward computation and PPO updates.',
            ha='center',fontsize=11,fontweight='bold')
    save(fig,'training_workflow')


def paired():
    rows=list(csv.DictReader((DATA/'paired_uncertainty.csv').open()))
    rows=[r for r in rows if r['method']!='native']
    fig,ax=plt.subplots(figsize=(10.8,3.9))
    for y,r in zip([1,0],rows):
        mid=float(r['mean_delta'])
        for offset,left,right,color,label in [
            (.09,'measurement_mean_delta_lower','measurement_mean_delta_upper',GRAY,'PDI measurement bounds'),
            (-.09,'conditional_instance_bootstrap_95_low','conditional_instance_bootstrap_95_high',ORANGE,'Conditional instance bootstrap 95%')]:
            lo,hi=float(r[left]),float(r[right])
            ax.errorbar([mid],[y+offset],xerr=[[mid-lo],[hi-mid]],fmt='o',color=color,capsize=5,lw=2,
                        label=label if y==1 else None)
        ax.text(mid+.5,y+.23,'mean '+format(mid,'+.3f'),fontsize=10)
    ax.axvline(0,color='#8F9499',ls='--',lw=1)
    ax.set(yticks=[1,0],yticklabels=['R0 - B0','Rjoint - B0'],ylim=(-.6,1.5),xlim=(-16,21),
           xlabel='Instance-equal mean PDI difference (< 0 favors the method)',
           title='Small method differences relative to measurement uncertainty')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax.grid(axis='x',alpha=.15)
    fig.text(.52,-.04,'66 paired instances; both fixed training seeds retained. Measurement bounds are not confidence intervals.\n'
             'Bootstrap conditions on the selected models and does not cover new-training-seed uncertainty.',ha='center',fontsize=9)
    fig.tight_layout()
    save(fig,'paired_uncertainty')


if __name__=='__main__':
    architecture();training();paired()
    (OUT/'figure_environment.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,
        'figures':['network_architecture','training_workflow','paired_uncertainty'],
        'scope':'Documentation only; existing experiment data and runtime unchanged.'},indent=2)+'\n')
    print('WEEKLY_FIGURES_WRITTEN')
