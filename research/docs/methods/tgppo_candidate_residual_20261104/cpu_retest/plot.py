"""Export the new-seed paired results; no training or solver invocation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'analysis.json').read_text())
assert data['complete']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
x=np.arange(2);width=.24
for ax,reference,title in zip(axes,('b0','score'),('CR relative to B0','CR relative to score shortcut S')):
    for j,(metric,label,color) in enumerate((('pdi','PDI','#337b8b'),('nodes','Nodes','#7377a9'),('solve_seconds','SCIP solve time','#cf8d48'))):
        values=[r['cr_vs_'+reference+'_'+metric+'_percent'] for r in data['instance_means']]
        bars=ax.bar(x+(j-1)*width,values,width,label=label,color=color)
        ax.bar_label(bars,labels=['{:+.1f}%'.format(v) for v in values],padding=3,fontsize=9)
    ax.axhline(0,color='black',lw=.8)
    ax.set(title=title,xticks=x,xticklabels=data['instances'],ylabel='Relative change (%)',ylim=(-80,15))
    ax.legend(frameon=False,loc='lower right')
fig.supxlabel('New solver seeds 2/3; both training seeds; fixed episode100 Actors. '
              'Same E5-2620 v3 CPU, one core. Lower is better; two selected development instances.',fontsize=9)
for ext in ('png','pdf'):
    fig.savefig(HERE/('new_seed_results.'+ext),dpi=190,bbox_inches='tight')
