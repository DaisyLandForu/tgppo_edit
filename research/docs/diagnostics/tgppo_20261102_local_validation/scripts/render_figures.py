"""Standalone figures from completed CSV outputs, no model computation."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','figure.dpi':140,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})


def read(name):return list(csv.DictReader((OUT/name).open()))


def save(fig,name):
    (OUT/'figures').mkdir(exist_ok=True);fig.tight_layout(pad=2)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}')
    plt.close(fig)


def main():
    checkpoints=[('B0-s11','100'),('B0-s11','500'),('B0-s22','100'),('B0-s22','500')]
    rs=read('policy_metrics.csv');gr=read('gradient_summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,4.3))
    colors={'original':'#177f94','ln2_affine_only':'#da8731'}
    for variant in colors:
        tv=[];zero=[];gradient=[]
        for model,ep in checkpoints:
            rows=[r for r in rs if r['model']==model and r['episode']==ep and r['variant']==variant]
            tv.append(max(1e-14,np.median([float(r['tv_to_uniform']) for r in rows])))
            zero.append(100*np.mean([r['head2_all_candidates_zero']=='True' for r in rows]))
            v=next(r for r in gr if r['model']==model and r['episode']==ep and r['variant']==variant and r['module']=='transformer')
            gradient.append(max(1e-16,float(v['median_l2'])))
        axs[0].plot(range(4),tv,'o-',color=colors[variant],label=variant)
        axs[1].plot(range(4),zero,'o-',color=colors[variant],label=variant)
        axs[2].plot(range(4),gradient,'o-',color=colors[variant],label=variant)
    axs[0].axhline(.01,color='gray',ls='--',label='screening threshold = 0.01')
    labels=[f's{m[-2:]} ep{ep}' for m,ep in checkpoints]
    for ax in axs:ax.set_xticks(range(4));ax.set_xticklabels(labels,rotation=20);ax.grid(alpha=.2)
    axs[0].set(yscale='log',ylabel='Median TV(policy, uniform)',title='A. No visible preference recovered')
    axs[0].legend(fontsize=7);axs[0].text(.02,.03,'Zero TV shown at 1e-14',transform=axs[0].transAxes,fontsize=8)
    axs[1].set(ylabel='% states with all-zero LN2-layer ReLU',ylim=(-4,104),title='B. Affine-only path: seed 11 dead ReLU')
    axs[2].set(yscale='log',ylabel='Median score-gradient L2 (Transformer)',title='C. Score gradient, no optimizer')
    axs[2].text(.02,.03,'Zero gradient shown at 1e-16',transform=axs[2].transAxes,fontsize=8)
    save(fig,'01_local_intervention')

    ms=read('microbenchmark_summary.csv');st=read('forward_module_times.csv')
    bins=[x for x in ['2-16','17-64','65-256','257+'] if any(r['bin']==x for r in ms)]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(len(bins));width=.34
    for component,offset,color,label in [('pure_actor_forward',-.17,'#177f94','pure forward'),
                                        ('complete_actor_policy_call',.17,'#da8731','complete policy call')]:
        data=[next(r for r in ms if r['bin']==b and r['component']==component) for b in bins]
        med=np.array([float(r['p50_ms']) for r in data]);p95=np.array([float(r['p95_ms']) for r in data])
        axs[0].bar(x+offset,med,width,color=color,label=label)
        axs[0].errorbar(x+offset,med,yerr=[np.zeros_like(med),p95-med],fmt='none',ecolor='black',capsize=3)
    axs[0].set_xticks(x);axs[0].set_xticklabels(bins)
    axs[0].set(xlabel='Candidate count',ylabel='Milliseconds; bars p50, upper whisker p95',title='A. Warmed single-thread CPU replay')
    axs[0].legend(fontsize=8)
    modules=['var_embedding','tree_embedding','fusion_linear','transformer','bimatching','treegate_head']
    bottom=np.zeros(len(bins))
    for module,color in zip(modules,['#b8d8da','#8dc4cb','#a0a1c9','#177f94','#da8731','#ab6598']):
        means=np.array([np.mean([float(r['fraction_of_instrumented_forward']) for r in st if r['bin']==b and r['module']==module])*100 for b in bins])
        axs[1].bar(x,means,bottom=bottom,color=color,label=module);bottom+=means
    axs[1].set_xticks(x);axs[1].set_xticklabels(bins)
    axs[1].set(xlabel='Candidate count',ylabel='% of separately instrumented forward',ylim=(0,105),title='B. Forward module attribution')
    axs[1].legend(fontsize=7,ncol=2)
    save(fig,'02_cpu_microbenchmark')
    print('Rendered two figures in PNG/SVG/PDF.')


if __name__=='__main__':main()
