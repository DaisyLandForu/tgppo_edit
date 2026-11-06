"""Export all-instance relative PDI and absolute-mean contribution figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'analysis.json').read_text())
assert data['complete']
rows=data['instance_summary'];y=np.arange(len(rows))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
names=[r['instance']+(' *' if not r['actor_participates'] else '') for r in rows]
for j,(ref,label,color) in enumerate((('b0','CR vs B0','#347f8b'),('score','CR vs S','#cb884b'))):
    values=[r['cr_vs_'+ref+'_pdi_percent'] for r in rows]
    bars=axes[0].barh(y+(j-.5)*.35,values,.35,label=label,color=color)
    axes[0].bar_label(bars,labels=[('{:+.2f}%' if abs(v)<.1 else '{:+.1f}%').format(v) for v in values],padding=3,fontsize=8)
axes[0].set(yticks=y,yticklabels=names,title='Relative PDI change on all frozen instances',xlabel='Relative change (%)')
axes[0].invert_yaxis();axes[0].axvline(0,color='black',lw=.8);axes[0].margins(x=.22)
axes[0].legend(frameon=False,loc='best')
values=[r['contribution_to_all_mean'] for r in rows]
bars=axes[1].barh(y,values,color=['#91949a' if not r['actor_participates'] else '#347f8b' for r in rows])
axes[1].bar_label(bars,labels=['{:+.3f}'.format(v) for v in values],padding=3,fontsize=8)
axes[1].set(yticks=y,yticklabels=names,title='Contribution to overall mean PDI change',xlabel='(CR PDI - B0 PDI) / 8')
axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.8);axes[1].margins(x=.22)
fig.supxlabel('Lower is better. Eight cost-stratified exploration instances; two training and two solver seeds. '
              '* = no Actor invocation in any model task. No statistical confidence interval.',fontsize=9)
for ext in ('png','pdf'):
    fig.savefig(HERE/('cross_instance_pdi.'+ext),dpi=190,bbox_inches='tight')
