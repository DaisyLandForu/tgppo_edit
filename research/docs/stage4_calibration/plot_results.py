"""Export calibrated offline diagnostics as standalone PNG/SVG figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-dir',type=Path,default=Path('/root/research/docs/stage4_calibration'))
    args=parser.parse_args();doc=args.report_dir
    summary=json.loads((doc/'summary.json').read_text())
    rows=summary['prediction_summary'];groups=['R0','Rcap','Rprod','Rjoint']
    colors=['#607080','#bd9a64','#428ea6','#b85255']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none'})
    for split in ('train','development'):
        fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
        for ax,loss in zip(axes,['mse','hl_gauss']):
            selected=[next(r for r in rows if r['group']==g and r['loss']==loss and r['split']==split) for g in groups]
            value=np.array([r['mse'] for r in selected]);low=value-np.array([r['mse_seed_min'] for r in selected]);high=np.array([r['mse_seed_max'] for r in selected])-value
            constant=summary['constants'][split]
            ax.bar(groups,value,color=colors,width=.65)
            ax.errorbar(range(4),value,yerr=[low,high],fmt='none',ecolor='#303840',capsize=5,lw=1.2)
            ax.axhline(constant,color='#404040',ls='--',lw=1,label='Train-mean constant')
            for i,v in enumerate(value):ax.text(i,v+high[i]+max(max(value+high),constant)*.02,f'{v:.2f}',ha='center',fontsize=9)
            ax.set_ylim(0,max(max(value+high),constant)*1.25)
            ax.set_ylabel('Raw-return MSE (lower is better)')
            ax.set_title('MSE fitting' if loss=='mse' else 'HL-Gauss fitting')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.legend(frameon=False,fontsize=8,loc='upper right')
        fig.suptitle('Calibrated Critic | '+('training instances' if split=='train' else 'held-out development instances'),fontsize=12)
        fig.supxlabel('Instance-equal means; whiskers show 3 fitting seeds. Fixed Actor; no online PPO result.',fontsize=9)
        for ext in ('png','svg'):fig.savefig(doc/('calibrated_'+split+'.'+ext))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.1),layout='constrained')
    old=summary['old_stage4_development']
    for ax,loss in zip(axes,['mse','hl_gauss']):
        for offset,key,label,color in [(-.18,old,'Before common calibration','#9ba4ab'),(.18,rows,'After common calibration','#428ea6')]:
            selected=[next(r['mse'] for r in key if r['group']==g and r['loss']==loss and r['split']=='development') for g in ['R0','Rjoint']]
            ax.bar(np.arange(2)+offset,selected,width=.35,color=color,label=label)
        ax.axhline(summary['constants']['development'],color='#404040',ls='--',lw=1)
        ax.set_xticks([0,1],['R0','Rjoint']);ax.set_ylabel('Development raw-return MSE')
        ax.set_title('MSE fitting' if loss=='mse' else 'HL-Gauss fitting')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Calibration is a common fitting adaptation, separate from information benefit',fontsize=11)
    fig.supxlabel('Before: 500 updates. After: train-only selected setting; initialization/scale also changed.',fontsize=9)
    for ext in ('png','svg'):fig.savefig(doc/('before_after_calibration.'+ext))
    plt.close(fig)


if __name__=='__main__':main()
