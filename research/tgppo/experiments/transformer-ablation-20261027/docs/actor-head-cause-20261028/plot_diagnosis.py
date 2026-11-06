"""Standalone figures from completed diagnostics, no model execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())


def main():
    cause=read('causal-results.json');bench=read('trained-benchmark-cpu.json');cost=read('cost-attribution.json')
    names=['T/11','T/22','BM/11','BM/22'];x=np.arange(4);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    original=[s['variants']['original64']['logit_span']['median'] for s in cause['summaries']]
    fixed=[s['variants']['ln2_fixed_variance']['logit_span']['median'] for s in cause['summaries']]
    axes[0,0].bar(x-.18,original,.36,label='Original head (FP64)');axes[0,0].bar(x+.18,fixed,.36,label='Fixed variance at real reference')
    axes[0,0].set(yscale='log',xticks=x,xticklabels=names,ylabel='Median candidate logit span',title='A. Same real BiMatching outputs; diagnostic only')
    axes[0,0].legend(fontsize=8)
    g0=[s['gradients']['original64']['contrast_gradient_l2_median'] for s in cause['summaries']]
    g1=[s['gradients']['ln2_fixed_variance']['contrast_gradient_l2_median'] for s in cause['summaries']]
    axes[0,1].bar(x-.18,g0,.36,label='Original head');axes[0,1].bar(x+.18,g1,.36,label='Fixed variance')
    axes[0,1].set(yscale='log',xticks=x,xticklabels=names,ylabel='Median L2 norm of contrast gradient',title='B. Sensitivity to candidate features recovers')
    for i,(a,b) in enumerate(zip(g0,g1)):axes[0,1].text(i,b*1.2,f'{b/a:,.0f}x',ha='center',fontsize=9)
    axes[0,1].set_ylim(top=max(g1)*5);axes[0,1].legend(fontsize=8)
    for variant,label in [('transformer','Transformer + BiMatching'),('bimatching','BiMatching Only')]:
        bystate={}
        for r in bench['rows']:
            if r['model'].startswith(variant):bystate.setdefault((r['job_id'],r['transition'],r['candidates']),[]).extend(r['samples_ms'])
        ordered=sorted(bystate,key=lambda key:key[2])
        axes[1,0].plot([key[2] for key in ordered],[np.mean(bystate[key]) for key in ordered],'o-',label=label)
    axes[1,0].set(xlabel='Candidates (same 12 real states)',ylabel='CPU forward latency (ms)',title='C. Trained weights; local Silver 4214 CPU')
    axes[1,0].legend(fontsize=8)
    bottom=np.zeros(2)
    for key,label in [('learning_wall_seconds','Learner update phase'),('collection_critical_path_lower_bound_seconds','Worker critical-path lower bound'),('scheduling_checkpoint_and_other_residual_seconds','Unresolved parent/IO/other residual')]:
        vals=[cost['comparison'][key][v+'_mean']/3600 for v in ('transformer','bimatching')]
        axes[1,1].bar([0,1],vals,bottom=bottom,label=label);bottom+=vals
    axes[1,1].set(xticks=[0,1],xticklabels=['Transformer + BM','BM Only'],ylabel='Training wall time (h)',title='D. Existing cluster runs; mean of seeds 11/22')
    axes[1,1].legend(fontsize=7,loc='upper right')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for suffix in ('png','pdf'):fig.savefig(HERE/('diagnosis-and-costs.'+suffix),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
