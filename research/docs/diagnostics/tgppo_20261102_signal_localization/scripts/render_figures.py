"""Standalone scientific figures; reuse the existing matplotlib environment."""
import os
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(OUT / 'logs/mpl-cache')
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def main():
    (OUT / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    rows = read('layer_summary.csv')
    checkpoints = [('B0-s11','100'),('B0-s11','500'),('B0-s22','100'),('B0-s22','500')]
    colors = ['#0072B2','#56B4E9','#D55E00','#E69F00']
    names = ['raw','var_embedding','fusion','tf0','tf1','tf2','tf3','tf4','bimatching',
             'head0.out','head1.out','head2.out','head3.out','head4.out','head5.out',
             'head6.linear','head6.norm','head6.out','logits']
    labels = ['Input','Embed','Fusion','T1','T2','T3','T4','T5','BiMatch',
              'H128','H64','H32','H16','H8','H4','Pre-LN2','LN2','ReLU2','Logits']
    idx = {(r['model'],r['episode'],r['variant'],r['layer']):float(r['instance_median_relative_rms_p50']) for r in rows}
    fig, ax = plt.subplots(figsize=(12.8,4.7))
    for (model,ep), color in zip(checkpoints,colors):
        vals = [idx[model,ep,'original',name] for name in names]
        ax.plot(range(len(names)),np.maximum(vals,1e-16),'-o',ms=3,lw=1.5,color=color,label=f'{model} ep{ep}')
    ax.axvline(8,color='#777777',ls=':',alpha=.6)
    ax.axvline(16,color='#777777',ls=':',alpha=.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels,rotation=50,ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Centered RMS / total RMS\n(median of instance medians)')
    ax.set_title('Frozen original Actor: where candidate contrasts contract')
    ax.legend(ncol=2,loc='lower left')
    ax.grid(axis='y',alpha=.2)
    fig.text(.01,.01,'Relative contrast is a localization measure, not retained task information. Zero values, if any, are shown at 1e-16.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/'figures'/f'01_layerwise_contrast.{ext}',dpi=180)
    plt.close(fig)

    summary = json.loads((OUT/'summary.json').read_text())
    gr = read('gradient_summary.csv')
    fig, axes = plt.subplots(1,2,figsize=(11.7,4.6))
    xpos = np.arange(4)
    for variant, color, shift, label in [('original','#0072B2',-.13,'Original'),
                                         ('ln2_fixed_state_variance','#D55E00',.13,'Fixed state variance')]:
        selected = [next(r for r in summary['policy'] if r['model']==model and str(r['episode'])==ep and r['variant']==variant)
                    for model,ep in checkpoints]
        med = [max(r['tv_uniform']['p50'],1e-16) for r in selected]
        maximum = [max(r['tv_uniform']['max'],1e-16) for r in selected]
        axes[0].plot(xpos+shift,med,'o',color=color,label=label+' median')
        axes[0].plot(xpos+shift,maximum,'d',mfc='none',color=color,label=label+' max')
        vals = [float(next(r for r in gr if r['model']==model and r['episode']==ep and r['variant']==variant and r['kind']=='parameters' and r['layer']=='transformer')['l2_p50']) for model,ep in checkpoints]
        axes[1].plot(xpos+shift,np.maximum(vals,1e-16),'o-',color=color,label=label)
    axes[0].axhline(.01,ls='--',color='#666666',label='Predeclared visibility screen (0.01)')
    axes[0].set_title('Action distribution, 164 states per checkpoint')
    axes[0].set_ylabel('TV distance to uniform')
    axes[1].set_title('Saved-action score gradient, 17 anchors')
    axes[1].set_ylabel('Transformer parameter gradient L2 median')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xticks(xpos)
        ax.set_xticklabels(['s11\nep100','s11\nep500','s22\nep100','s22\nep500'])
        ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=7,loc='best')
    fig.text(.01,.01,'One predefined frozen-weight intervention; no optimizer or solver. Nonuniformity and larger gradients do not establish better decisions.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/'figures'/f'02_fixed_scale_intervention.{ext}',dpi=180)
    plt.close(fig)
    print('Rendered 2 figures, each PNG/SVG/PDF.')


if __name__ == '__main__':
    main()
