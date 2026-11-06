#!/usr/bin/env python3
"""Plot audited aggregate files; no runtime imports or experiment mutations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
data = json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
methods = ['native', 'B0', 'R0', 'Rjoint']
labels = ['relpscost', 'B0', 'R0', 'Rjoint']
colors = ['#626C76', '#3369A8', '#D08032', '#418465']


def save(fig, stem):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(OUT/(stem+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
rows = data['combined']
x = range(4)
axes[0].bar(x, [v['mean_pdi'] for v in rows], color=colors, width=.62, alpha=.83)
axes[0].set(ylabel='Mean 120 s PDI (percent-seconds; lower is better)', ylim=(0, 2800))
for i, value in enumerate(rows):
    axes[0].text(i, value['mean_pdi']+92, '{:.2f}'.format(value['mean_pdi']), ha='center', fontsize=10)
for seed, marker, offset in [(11,'o',-.13), (22,'s',.13)]:
    values = [next(r['mean_pdi'] for r in data['per_training_seed']
                   if r['method']==m and r['training_seed']==seed) for m in methods[1:]]
    axes[0].scatter([i+offset for i in range(1,4)], values, color='black', marker=marker,
                    s=31, label='Training seed '+str(seed), zorder=3)
axes[0].legend(loc='lower right', frameon=False, fontsize=9)
axes[1].bar(x, [v['proof_rate']*100 for v in rows], color=colors, width=.62, alpha=.83)
axes[1].set(ylabel='Proof completion within 120 s (%)', ylim=(0, 59))
for i, value in enumerate(rows):
    axes[1].text(i, value['proof_rate']*100+1.4, '{:.2f}%'.format(value['proof_rate']*100), ha='center')
for ax in axes:
    ax.set_xticks(list(x), labels)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
fig.suptitle('66 held-out instances: both training seeds retained', fontsize=14)
fig.text(.5, -.035, '3 solver seeds per instance; mspp16: common 24 GiB exception. '
         'B0 uses the 120 s / Adam protocol, not the full paper protocol.\n'
         'Proof completion includes cutoff proofs. Evaluation uses CPU Actor inference.',
         ha='center', fontsize=9)
fig.tight_layout()
save(fig, 'main_results')

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
for ax, seed in zip(axes, [11,22]):
    for method, color in zip(methods[1:], colors[1:]):
        points = sorted([v for v in data['development_curves'] if v['method']==method and v['training_seed']==seed],
                        key=lambda v:v['episode'])
        ax.plot([v['episode'] for v in points], [v['mean_pdi'] for v in points], '-o', color=color, label=method, markersize=4)
        chosen = next(v for v in data['selected_models'] if v['model']==method+'-s'+str(seed))
        ax.scatter([chosen['selected_episode']], [chosen['development_mean_pdi']], marker='*', s=165,
                   color=color, edgecolor='black', linewidth=.6, zorder=4)
    ax.set(title='Training seed '+str(seed), xlabel='Complete training episodes', xticks=[100,200,300,400,500])
    ax.set_axisbelow(True)
    ax.grid(color='#DDDDDD', linewidth=.6)
axes[0].set_ylabel('Development mean 120 s PDI (lower is better)')
axes[1].legend(frameon=False)
fig.suptitle('Common-budget development curves (5 instances x 3 solver seeds)', fontsize=13)
fig.text(.5, -.02, 'Stars: checkpoints selected before held-out testing. These are development curves, not test learning curves.',
         ha='center', fontsize=9)
fig.tight_layout()
save(fig, 'development_curves')
(OUT/'plot_environment.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,
    'scope':'isolated temporary plotting environment; shared experiment Conda environment unchanged'},indent=2)+'\n')
print('PLOTS_WRITTEN')
