"""Standalone publication-style stage-4 diagnostic figures; no model selection."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOC=Path('/root/research/docs/stage4')
GROUPS=['R0','Rcap','Rprod','Rjoint']
COLORS=['#607080','#bd9a64','#428ea6','#b85255']


def rows(name):
    with (DOC/name).open() as f: return list(csv.DictReader(f))


def main():
    global DOC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=DOC)
    args=parser.parse_args();DOC=args.report_dir.resolve()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'svg.fonttype':'none'})
    data=rows('per_instance_prediction.csv')
    constant=json.loads((DOC/'summary.json').read_text())['constant_predictor_development_mse']
    dev=[r for r in data if r['split']=='development']
    names=sorted({r['instance'] for r in dev})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
    for ax,loss in zip(axes,['mse','hl_gauss']):
        values=[]
        for group in GROUPS:
            values.append([np.mean([float(r['mse']) for r in dev if r['loss']==loss and
                r['group']==group and int(r['seed'])==seed]) for seed in [11,22,33]])
        mean=np.mean(values,axis=1); low=mean-np.min(values,axis=1); high=np.max(values,axis=1)-mean
        ax.bar(GROUPS,mean,color=COLORS,width=.64,alpha=.9)
        ax.errorbar(np.arange(4),mean,yerr=[low,high],fmt='none',ecolor='#303840',capsize=5,lw=1.3)
        ax.axhline(constant,color='#404040',ls='--',lw=1,label='Train-mean constant')
        for i,v in enumerate(mean): ax.text(i,v+high[i]+max(mean)*.02,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_ylabel('Return prediction MSE (lower is better)')
        ax.set_title('MSE training' if loss=='mse' else 'HL-Gauss training')
        ax.set_ylim(0,max(max(mean+high),constant)*1.2)
        ax.legend(frameon=False,fontsize=8,loc='upper right')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f'Fixed-Actor offline diagnosis | {len(names)} held-out instances, 3 fitting seeds',fontsize=12)
    fig.supxlabel('Instance-equal averages; whiskers show fitting-seed range. No online PPO comparison.',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(DOC/f'prediction_comparison.{suffix}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
    for ax,loss in zip(axes,['mse','hl_gauss']):
        for index,control in enumerate(['R0','Rcap','Rprod']):
            values=[]
            for name in names:
                scores={g:np.mean([float(r['mse']) for r in dev if r['loss']==loss and r['group']==g and r['instance']==name])
                        for g in [control,'Rjoint']}
                values.append((scores['Rjoint']/scores[control]-1)*100)
            ax.bar(np.arange(len(names))+(index-1)*.24,values,width=.23,label=f'vs {control}',color=COLORS[index])
        ax.axhline(0,color='#404040',lw=.9)
        ax.set_xticks(np.arange(len(names)),names,rotation=20,ha='right')
        ax.set_ylabel('Rjoint MSE change (%) | negative is better')
        ax.set_title('MSE training' if loss=='mse' else 'HL-Gauss training')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Per-instance paired comparison | mean over 3 fitting seeds',fontsize=12)
    fig.supxlabel('Prediction improvement does not establish faster MILP solving.',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(DOC/f'per_instance_comparison.{suffix}')
    plt.close(fig)
    histogram=rows('value_scale_histogram.csv')
    x=np.asarray([float(r['center']) for r in histogram])
    fig,ax=plt.subplots(figsize=(9.4,3.8),layout='constrained')
    for key,label,color in [('target','Empirical return target','#607080'),
        ('prediction_mse','Rjoint prediction: MSE','#b85255'),
        ('prediction_hl_gauss','Rjoint prediction: HL-Gauss','#428ea6')]:
        ax.step(x,[float(r[key]) for r in histogram],where='mid',label=label,color=color,lw=1.7)
    ax.set(xlabel='Return value',ylabel='Probability mass per bin',
        title='Common value-scale underfitting | held-out instances weighted equally')
    ax.axvline(0,lw=.8,color='#808080');ax.grid(axis='y',alpha=.18)
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.supxlabel('Predictions average the 3 final fitting seeds. No extra training or model selection.',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(DOC/f'value_scale.{suffix}')
    plt.close(fig)


if __name__=='__main__':main()
