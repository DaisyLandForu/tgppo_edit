"""Export review figures from measured results; no model or solver execution."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'analysis.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name):
    for suffix in ('png','pdf'):
        fig.savefig(HERE/(name+'.'+suffix),dpi=190,bbox_inches='tight')
    plt.close(fig)

with (HERE/'pdi_contributions.csv').open() as stream:
    rows=list(csv.DictReader(stream))
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
labels=[r['instance']+(' (no Actor)' if r['actor_participates']=='False' else '') for r in rows]
values=[float(r['contribution_to_all_mean']) for r in rows]
axes[0].barh(labels,values,color=['#1e7b8a']*4+['#8b8c91'])
axes[0].invert_yaxis();axes[0].axvline(0,color='black',lw=.8)
axes[0].set(xlabel='Contribution to all-instance mean PDI change (CR - B0)',
            title='95.6% of apparent gain: no Actor invoked')
for i,value in enumerate(values):
    axes[0].annotate('{:+.2f}'.format(value),(value,i),xytext=(-4 if value<0 else 4,0),
                     textcoords='offset points',ha='right' if value<0 else 'left',va='center',fontsize=9)
axes[0].set_xlim(-103,8)
x=np.arange(2);width=.34
for i,(ref,label,color) in enumerate((('b0','CR vs B0','#1e7b8a'),('score','CR vs S','#ca6e39'))):
    changes=[next(r['percent_delta'] for r in data['differences'] if r['panel']=='actor_participating'
                  and r['training_seed']==seed and r['reference']==ref) for seed in (11,22)]
    bars=axes[1].bar(x+(i-.5)*width,changes,width,label=label,color=color)
    axes[1].bar_label(bars,labels=['{:+.2f}%'.format(v) for v in changes],padding=4,fontsize=9)
axes[1].axhline(0,color='black',lw=.8)
axes[1].set(xticks=x,xticklabels=['Training seed 11','Training seed 22'],
            ylabel='Mean PDI change (%)',ylim=(-3.1,2.1),title='Four Actor-participating instances')
axes[1].legend(frameon=False,loc='lower right')
fig.supxlabel('Lower is better. Same evaluation host; 5 used development instances, 2 solver seeds. '
              'Decomposition is descriptive, not a new primary endpoint.',fontsize=9)
save(fig,'pdi_attribution')

with (HERE/'dcmulti_paired.csv').open() as stream:
    pairs=list(csv.DictReader(stream))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
labels=['Train {} / solver {}'.format(r['training_seed'],r['solver_seed']) for r in pairs]
for ax,metric,title in zip(axes,('pdi','solve_seconds'),('PDI change','SCIP solve time change')):
    changes=[100*(float(r['candidate_'+metric])/float(r['b0_'+metric])-1) for r in pairs]
    bars=ax.bar(np.arange(4),changes,color=['#1e7b8a' if v<0 else '#c55151' for v in changes],width=.65)
    ax.bar_label(bars,labels=['{:+.1f}%'.format(v) for v in changes],padding=4,fontsize=9)
    ax.axhline(0,color='black',lw=.8)
    ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='CR relative to B0 (%)',
           title='dcmulti: '+title,ylim=(-58,38))
    ax.tick_params(axis='x',labelrotation=16)
fig.supxlabel('Lower is better. Mean solve time: B0 6.90 s, CR 5.66 s, native 2.73 s. '
              'Four repeats of one development instance; no confidence interval.',fontsize=9)
save(fig,'dcmulti_repeats')
