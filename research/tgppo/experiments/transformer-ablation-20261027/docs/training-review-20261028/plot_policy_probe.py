"""Plot measured head traces and policy uniformity; no model execution."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'late-policy-probe.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
labels=['BiMatching','128','64','32','16','8','4','2','1']
for r in rows:
    if r['variant']=='legacy-B0':continue
    y=[r['layer_candidate_span']['BiMatching']['median']]+[r['layer_candidate_span']['head_layer_'+str(i)]['median'] for i in range(8)]
    axes[0].plot(range(9),y,marker='o',linestyle='-' if r['seed']==11 else '--',color='tab:blue' if r['variant']=='transformer' else 'tab:orange',label=r['variant']+'/s'+str(r['seed']))
axes[0].set_yscale('log');axes[0].set_xticks(range(9),labels,rotation=30)
axes[0].set_xlabel('BiMatching / Actor head output width')
axes[0].set_ylabel('Median within-state candidate feature span')
axes[0].set_title('Candidate differences shrink in the common Actor head')
axes[0].axvline(7,color='gray',alpha=.4,linestyle=':');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
axes[1].bar(range(len(rows)),[r['max_total_variation_from_uniform'] for r in rows],color=['tab:blue' if r['variant']=='transformer' else 'tab:orange' if r['variant']=='bimatching' else 'gray' for r in rows])
axes[1].set_yscale('log');axes[1].set_xticks(range(len(rows)),[r['variant']+'\ns'+str(r['seed']) for r in rows],rotation=15)
axes[1].set_ylabel('Maximum total variation from uniform')
axes[1].set_title('Final policies on sampled late training states')
axes[1].grid(axis='y',alpha=.2)
for suffix in ('.png','.pdf','.svg'):fig.savefig(HERE/('policy-head-diagnostic'+suffix),dpi=180)
plt.close(fig)
