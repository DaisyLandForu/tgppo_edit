"""Render the completed output audit; no model or solver imports."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'policies.csv').open()))
layers=list(csv.DictReader((OUT/'layers.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
models=[('old_light_b0_s11_ep100','Old100\ns11'),('old_light_b0_s22_ep100','Old100\ns22'),
        ('old_server_b0_s11_ep500','Old500\ns11'),('old_server_b0_s22_ep500','Old500\ns22'),
        ('current_b0_s11_latest','B0 now\ns11'),('current_b0_s22_latest','B0 now\ns22'),
        ('current_mlp_s11_latest','MLP now\ns11'),('current_mlp_bc_s11_latest','MLP+BC\ns11')]
for i,(name,label) in enumerate(models):
    values=[float(r['tv_uniform']) for r in rows if r['model']==name]
    color='#376cb4' if 'b0' in name else '#219373'
    axes[0].scatter(i+np.linspace(-.16,.16,len(values)),np.maximum(values,1e-12),
                    s=17,alpha=.6,color=color)
    axes[0].plot([i-.24,i+.24],[max(np.median(values),1e-12)]*2,color='black',lw=2)
axes[0].set_xticks(range(len(models)),[x[1] for x in models],fontsize=8)
axes[0].set(yscale='log',ylabel='TV distance from uniform',
            title='Same 22 saved states; black line = median')
stages=['embedding','fusion','transformer','bimatching','head_8','head_4','ln2_input','ln2','logits']
labels=['Embed','Fusion','Trf','Match','Head8','Head4','PreLN2','LN2','Logits']
for seed,color in [(11,'#376cb4'),(22,'#d08429')]:
    for suffix,style in [('initial','--'),('latest','-')]:
        name='current_b0_s{}_{}'.format(seed,suffix)
        ys=[np.median([float(r['relative_candidate_contrast']) for r in layers
                       if r['model']==name and r['stage']==stage]) for stage in stages]
        axes[1].plot(labels,ys,color=color,ls=style,marker='o',ms=3,
                     label='s{} {}'.format(seed,suffix))
axes[1].set(yscale='log',ylabel='Median relative candidate contrast',
            title='B0 scoring path: contrast declines across stages')
axes[1].tick_params(axis='x',labelrotation=35,labelsize=8)
axes[1].legend(fontsize=8,frameon=False)
for ax in axes:
    ax.grid(alpha=.18);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.suptitle('B0 output audit: local checkpoints, not author-released weights',fontsize=14)
fig.text(.5,-.02,'22 fixed states from 17 training instances; correlated observations, not independent test samples.\n'
         'TV/contrast describe output response, not branching correctness or PDI benefit. Exact TV zeros shown at 1e-12.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.93))
for ext in ('png','pdf','svg'):
    fig.savefig(OUT/('b0_output_audit.'+ext),dpi=160,bbox_inches='tight')
