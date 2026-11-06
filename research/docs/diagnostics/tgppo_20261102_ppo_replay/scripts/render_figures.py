"""Export a standalone figure from recorded measurements only."""
import os
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(OUT/'logs/mpl-cache')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    s=json.loads((OUT/'summary.json').read_text())
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    models=['B0-s11','B0-s22'];x=np.arange(2)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for kind,color,shift in [('actor','#0072B2',-.14),('critic','#D55E00',.14)]:
        values=[next(r for r in s['parameter_updates'] if r['model']==m and r['network']==kind)['parameter_delta_l2'] for m in models]
        axes[0].bar(x+shift,np.maximum(values,1e-18),width=.26,color=color,label=kind.title())
    axes[0].set_ylabel('Parameter change L2')
    axes[0].set_title('Exactly one Adam step per network')
    monitor=[next(r for r in s['response'] if r['model']==m and r['dataset']=='monitor_164') for m in models]
    axes[1].plot(x,[max(r['tv_pre_post']['p50'],1e-18) for r in monitor],'o',color='#0072B2',label='Median, 164 fixed states')
    axes[1].plot(x,[max(r['tv_pre_post']['max'],1e-18) for r in monitor],'d',mfc='none',color='#0072B2',label='Max, 164 fixed states')
    axes[1].axhline(1e-6,ls='--',color='#666666',label='Conservative numerical screen')
    axes[1].set_ylabel('TV(pre-update policy, post-update policy)')
    axes[1].set_title('Measured action-distribution response')
    axes[2].bar(x-.14,[max(r['abs_common_logit_shift']['p50'],1e-18) for r in monitor],width=.26,color='#0072B2',label='Absolute common shift median')
    axes[2].bar(x+.14,[max(r['centered_logit_delta_rms']['p50'],1e-18) for r in monitor],width=.26,color='#D55E00',label='Centered change RMS median')
    axes[2].set_ylabel('Logit change')
    axes[2].set_title('Common shift versus candidate contrast')
    for ax in axes:
        ax.set_yscale('log');ax.set_xticks(x);ax.set_xticklabels(['s11 ep100','s22 ep100'])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=7,loc='best')
    fig.text(.01,.01,'CPU diagnostic replay, not a full historical PPO update. Parameter movement and numerical policy changes do not establish solver benefit.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'01_single_step_response.{ext}',dpi=180)
    plt.close(fig)
    print('Rendered PNG/SVG/PDF from recorded results.')


if __name__=='__main__':main()
