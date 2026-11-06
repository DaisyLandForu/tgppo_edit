"""Render standalone report figures using only the saved diagnostic tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
OLD=ROOT/'tgppo/runs/server6-20261022/stage5-results-20261023'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'savefig.bbox':'tight','svg.fonttype':'none'})


def read(p):return list(csv.DictReader(p.open()))


def finish(fig,name):
    fig.tight_layout(pad=2)
    for suffix in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{suffix}')
    plt.close(fig)


def main():
    models=['B0-s11','B0-s22','R0-s11','R0-s22','Rjoint-s11','Rjoint-s22']
    a=read(OUT/'actor_checkpoint_metrics.csv'); c=read(OUT/'actor_checkpoint_comparison.csv')
    acts=read(OUT/'actor_activation_spread.csv');dev=read(OLD/'development_curves.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    values=[[max(float(r['logit_spread']),1e-12) for r in a if r['model']==m] for m in models]
    axs[0,0].boxplot(values,labels=models,showfliers=True)
    axs[0,0].set(yscale='log',ylabel='Max logit - min logit',title='A. Frozen outputs: 50 TRAIN states x 2 checkpoints')
    axs[0,0].tick_params(axis='x',rotation=20)
    modules=['var_embedding','transformer','bimatching']+[f'head{i}' for i in range(7)]
    for model,ep,color,style in [('B0-s11','100','#147d92','-'),('B0-s11','500','#147d92','--'),('B0-s22','100','#e58d29','-'),('B0-s22','500','#e58d29','--')]:
        ys=[np.median([float(r['mean_feature_candidate_range']) for r in acts if r['model']==model and r['episode']==ep and r['module']==m]) for m in modules]
        axs[0,1].plot(range(len(modules)),ys,style,marker='o',color=color,label=f'{model} ep{ep}')
    axs[0,1].set(yscale='log',ylabel='Median candidate range, averaged over channels',title='B. Candidate signal through the original Actor')
    axs[0,1].set_xticks(range(len(modules)),['embed','TF','BM','128','64','32','16','8','4','2'])
    axs[0,1].set_xlabel('Representation / head output width (last: LN2 + ReLU)');axs[0,1].legend(fontsize=8)
    vals=[[max(float(r['tv_distance']),1e-12) for r in c if r['model']==m] for m in models]
    axs[1,0].boxplot(vals,labels=models,showfliers=True)
    axs[1,0].set(yscale='log',ylabel='Total variation distance',title='C. Same seed: episode 100 vs 500 on identical states')
    axs[1,0].tick_params(axis='x',rotation=20)
    colors={'B0':'#147d92','R0':'#e58d29','Rjoint':'#a55a9b'}
    for model in models:
        rs=sorted([r for r in dev if r['model']==model],key=lambda r:int(r['episode']))
        y=np.array([float(r['mean_pdi']) for r in rs]);method=model.split('-')[0]
        axs[1,1].plot([int(r['episode']) for r in rs],y-y[0],'-' if model.endswith('11') else '--',marker='o',color=colors[method],label=model)
    axs[1,1].axhline(0,color='gray',lw=.7);axs[1,1].legend(fontsize=8,ncol=2)
    axs[1,1].set(xlabel='Training episodes',ylabel='Dev PDI minus own episode-100 score',title='D. Five-instance dev curves (exploratory)')
    finish(fig,'01_learning_diagnostics')

    matrix=read(OUT/'per_instance_diagnostic_matrix.csv');cohorts=read(OUT/'difficulty_cohorts.csv')
    pairs=read(OUT/'both_proven_pairs.csv');cost=read(OLD/'deployment_cost.csv')[0]
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    x=np.array([float(r['native_pdi']) for r in matrix]); y=np.array([float(r['b0_pdi']) for r in matrix])
    axs[0,0].scatter(x+1,y+1,c=[float(r['native_proof_rate']) for r in matrix],cmap='viridis',edgecolors='black',lw=.3)
    axs[0,0].plot([1,12001],[1,12001],'--',color='gray')
    axs[0,0].set(xscale='log',yscale='log',xlabel='Native PDI + 1',ylabel='B0 PDI + 1',title='A. Paired instance means; 66 exploratory instances')
    cs=cohorts[1:4];axs[0,1].bar(range(3),[float(r['b0_minus_native']) for r in cs],color=['#c86456','#dcad55','#789e9e'])
    axs[0,1].set_xticks(range(3),['All 3 seeds proven\n(n=31)','Some proven\n(n=4)','None proven\n(n=31)'])
    axs[0,1].set(ylabel='B0 minus native mean PDI',title='B. Cohorts defined by native 120s proof outcomes')
    opt=float(cost['optimize_wall_seconds']);act=float(cost['actor_seconds']);feat=float(cost['base_feature_seconds']);cb=float(cost['entire_branch_callback_seconds'])
    labels=['Actor policy path','Base features','Other branch callback','Other optimize']
    vals=[act,feat,cb-act-feat,opt-cb];left=0
    for label,value,color in zip(labels,vals,['#147d92','#e58d29','#a55a9b','#c4c9cd']):
        axs[1,0].barh([0],[value/opt*100],left=left,color=color,label=label)
        left+=value/opt*100
    axs[1,0].set(xlim=(0,100),yticks=[],xlabel='% of summed optimize wall time',title='C. B0 nested timers partitioned once')
    axs[1,0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.25),ncol=2)
    px=[float(r['b0_nodes'])/float(r['native_nodes']) if float(r['native_nodes']) else np.nan for r in pairs]
    py=[float(r['b0_optimize_seconds'])/float(r['native_optimize_seconds']) for r in pairs]
    axs[1,1].scatter(px,py,color='#147d92',alpha=.7,s=24)
    axs[1,1].axvline(1,color='gray',ls='--');axs[1,1].axhline(1,color='gray',ls='--')
    axs[1,1].set(xscale='log',yscale='log',xlabel='B0 / native completed-tree nodes',ylabel='B0 / native optimize wall time',title='D. Both-proven pairs: 76 pairs, 14 instances')
    finish(fig,'02_runtime_solver_diagnostics')

    curves=read(OUT/'gap_time_examples.csv')
    chosen=sorted(matrix,key=lambda r:abs(float(r['b0_minus_native'])),reverse=True)[:6]
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,instance in zip(axs.flat,[r['instance'] for r in chosen]):
        for model,color in [('native','#cf654d'),('B0-s11','#147d92')]:
            rs=sorted([r for r in curves if r['instance']==instance and r['model']==model and r['solver_seed']=='0'],key=lambda r:float(r['time']))
            xs=[0.];ys=[100.]
            for r in rs:
                if float(r['time'])<=120:
                    xs.append(float(r['time']));ys.append(float(r['percent_bound_gap']))
            if rs and rs[-1]['proof_completed']=='True' and float(rs[-1]['time'])<=120:
                xs.append(float(rs[-1]['time']));ys.append(0.)
            xs.append(120.);ys.append(ys[-1]);ax.step(xs,ys,where='post',label=model,color=color)
        ax.set(title=instance,xlim=(0,120),ylim=(-2,102));ax.grid(alpha=.2)
    axs[0,0].legend();fig.supxlabel('SCIP solving time (s), a wall-clock progress proxy');fig.supylabel('Bound-derived percent gap / proof extension')
    finish(fig,'03_gap_time_examples')
    print('9 figures written (PNG, SVG, PDF); no solver execution')


if __name__=='__main__':main()
