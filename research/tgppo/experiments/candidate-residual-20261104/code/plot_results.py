"""Standalone exportable method/experiment figures from measured CSVs."""
import csv
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
def rows(name):
    return list(csv.DictReader((root/name).open()))
def save(fig, name):
    fig.tight_layout()
    fig.savefig(root/(name+'.png'), dpi=180)
    fig.savefig(root/(name+'.pdf'))
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis('off')
for x, y, label in [(0.12,.65,'Candidate embedding e_i'), (.48,.65,'Transformer + BiMatching m_i'),
        (.82,.65,'Original score'), (.48,.15,'w^T (m_i + e_i)'), (.82,.15,'Add scores / masked softmax')]:
    ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#e8f1fb'),fontsize=9)
for start, end in [((.24,.65),(.34,.65)), ((.64,.65),(.74,.65)),
        ((.12,.53),(.4,.22)), ((.48,.53),(.48,.26)), ((.62,.15),(.71,.15)), ((.82,.53),(.82,.27))]:
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->'))
ax.set_title('Candidate residual score: 256 extra parameters; w initialized to zero')
save(fig,'method')

data = rows('paired_instances.csv')
names = sorted({r['instance'] for r in data})
fig, ax = plt.subplots(figsize=(8,4))
for seed, marker in [('11','o'),('22','s')]:
    for offset, metric, color in [(-.10,'candidate_minus_b0','#2667a8'),(.10,'candidate_minus_score','#d47722')]:
        selected = [next(r for r in data if r['training_seed']==seed and r['instance']==n) for n in names]
        ax.scatter([i+offset for i in range(len(names))],[float(r[metric]) for r in selected],
            marker=marker,color=color,label='seed '+seed+', '+metric.replace('candidate_minus_','CR - '))
ax.axhline(0,color='gray',linewidth=.8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=15)
ax.set_ylabel('Paired PDI difference (lower is better)'); ax.legend(fontsize=7)
save(fig,'paired')

data = rows('learning.csv')
fig, ax = plt.subplots(figsize=(8,4))
for variant in ('b0','score','candidate'):
    for seed in ('11','22'):
        selected = [r for r in data if r['variant']==variant and r['training_seed']==seed]
        points=[]
        for end in range(20, max([int(r['episode']) for r in selected]+[0])+1,20):
            block=[r for r in selected if end-20 < int(r['episode']) <= end]
            if len(block)==20:
                points.append((end,statistics.mean(float(r['pdi']) for r in block)))
        if points:
            ax.plot([p[0] for p in points],[p[1] for p in points],marker='.',label=variant+' s'+seed)
ax.set_xlabel('Complete training episodes'); ax.set_ylabel('Training PDI / full 20-instance cycle')
ax.set_title('On-policy training curve; not development selection'); ax.legend(fontsize=8)
save(fig,'learning')

data=rows('model_summary.csv')
fig, ax=plt.subplots(figsize=(8,4))
for i, variant in enumerate(('native','b0','score','candidate')):
    selected=[r for r in data if r['variant']==variant]
    values=[float(r['pdi']) for r in selected]
    ax.bar(i,statistics.mean(values),color='#c5d9ee')
    ax.scatter([i]*len(values),values,color='#174a7b',zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(['relpscost','TGPPO','Score shortcut','Candidate residual'])
ax.set_ylabel('Instance-equal development PDI (120s)')
ax.set_title('Dots are training seeds; 5 previously used development instances')
save(fig,'performance')
