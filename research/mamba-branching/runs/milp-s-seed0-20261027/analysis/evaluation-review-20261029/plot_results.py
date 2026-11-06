"""Plot this frozen evaluation report; requires matplotlib, no training imports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out = Path(__file__).resolve().parent
r = json.loads((out / 'analysis.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
labels = [v['instance'] for v in r['instances']] + ['Overall (GM)']
for ax, metric, title, color, ticks, limits in zip(
    axes, ['fair_nodes', 'wall_seconds'], ['Fair nodes', 'Wall-clock time'],
    ['#2767a4', '#bd6626'], [[0.5, 1, 2, 4, 8], [0.75, 1, 1.5, 2, 3, 4, 5]],
    [(0.35, 12), (0.65, 6.5)]):
    ax.set_xscale('log')
    ax.set_xlim(*limits)
    ax.axvspan(limits[0], 1, color='#edf7f0', zorder=0)
    ax.axvline(1, color='#66717e', linestyle='--', linewidth=1.1)
    ax.axhline(7.5, color='#c9ced5', linewidth=1)
    for i, item in enumerate(r['instances']):
        group = [x for x in r['raw_rows'] if x['task']['instance'] == item['instance']]
        ratios = [x['policies']['mamba'][metric] / x['policies']['relpscost'][metric] for x in group]
        ax.scatter(ratios, [i + (j - 2) * .055 for j in range(5)], s=20,
                   color='#a5acb5', alpha=.85, zorder=2)
        value = item['metrics'][metric]['mamba_over_relpscost']
        ax.scatter([value], [i], marker='D', s=45, color=color, edgecolors='white', linewidths=.65, zorder=3)
        ax.annotate(f'{value:.2f}x', (value, i), xytext=(7, -1), textcoords='offset points',
                    color=color, fontsize=9, fontweight='bold', va='bottom', zorder=4)
    overall = r['overall'][metric]['mamba_over_relpscost']
    low, high = r['instance_level_statistics'][metric]['instance_bootstrap_95_interval']
    ax.errorbar([overall], [8.25], xerr=[[overall-low], [high-overall]], marker='D',
                 color=color, markersize=7, linewidth=2, capsize=4, zorder=3)
    ax.annotate(f'{overall:.2f}x', (overall, 8.25), xytext=(0, 10), textcoords='offset points',
                 ha='center', color=color, fontweight='bold', fontsize=10)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(ticks, [f'{x:g}x' for x in ticks])
    ax.minorticks_off()
    ax.grid(axis='x', alpha=.14)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.set_xlabel('Mamba / relpscost  (lower is better)', labelpad=12)
    ax.set_ylim(8.85, -.65)
axes[0].set_yticks([*range(8), 8.25], labels)
fig.suptitle('MILP-S: frozen seed0 policy versus relpscost', x=.04, ha='left', fontsize=16,
             fontweight='bold', y=.98)
fig.text(.04, .923, '8 test instances, 5 solver seeds each; same-host paired solves with known-optimum cutoff',
         fontsize=10, color='#555e68')
legend = [Line2D([0], [0], marker='o', linestyle='none', color='#a5acb5', label='Individual solver seed', markersize=5),
          Line2D([0], [0], marker='D', linestyle='none', color='#364152', label='Geometric mean', markersize=6)]
fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.04, .035), frameon=False, ncol=2)
fig.text(.04, .018, 'Overall whiskers: descriptive 95% instance-bootstrap interval; one training seed only.  Left of 1 favors Mamba.',
         fontsize=9, color='#555e68')
fig.subplots_adjust(left=.17, right=.96, top=.85, bottom=.16, wspace=.17)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(out / ('paired-performance.' + ext), dpi=180, facecolor='white')
print(out / 'paired-performance.png')
