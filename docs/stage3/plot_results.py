"""Export the stage-3 development comparison using the separate base plotting environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

ROOT = Path('/root/research')
DOC = ROOT / 'docs/stage3'
RUN = ROOT / 'tgppo/runs/stage3'


def main():
    global RUN,DOC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--report-dir',type=Path,default=DOC)
    args=parser.parse_args()
    RUN,DOC=args.run_root.resolve(),args.report_dir.resolve()
    summary=json.loads((DOC/'development_summary.json').read_text())
    native=json.loads((RUN/'native/jobs/results.json').read_text())
    b0=json.loads((RUN/'evaluation/jobs/results.json').read_text())
    names=[r['instance'] for r in summary['instances']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,proof)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[2.2,1]})
    colors=['#2563a6','#d47722']
    for method,rows,color,offset in [('relpscost',native,colors[0],-.13),('B0 pilot',b0,colors[1],.13)]:
        for i,name in enumerate(names):
            values=[r['pdi_horizon_midpoint'] for r in rows if r['instance_name']==name and r.get('valid')]
            if values:
                center=float(np.mean(values))
                ax.plot([min(values),max(values)],[i+offset,i+offset],color=color,lw=2,alpha=.6)
                ax.scatter([max(center,1e-5)],[i+offset],color=color,s=44,label=method if i==0 else None,zorder=3)
            successes=sum(bool(r.get('proof_completed') and r.get('scip_solve_time',999)<=120.)
                          for r in rows if r['instance_name']==name)
            proof.barh(i+offset,successes,height=.24,color=color)
    ax.set_xscale('log')
    ax.set_yticks(range(len(names)),names)
    ax.set_ylim(len(names)-.5,-.5)
    ax.set_xlabel('PDI over 120 s (percent-seconds; lower is better)')
    ax.set_title('Mean and seed range of PDI')
    ax.grid(axis='x',alpha=.2)
    ax.legend(loc='best',frameon=False)
    proof.set_yticks(range(len(names)),['']*len(names))
    proof.set_ylim(len(names)-.5,-.5)
    proof.set_xticks([0,1,2,3])
    proof.set_xlim(0,3.3)
    proof.set_xlabel('Completed / 3 solver seeds')
    proof.set_title('Cutoff proof by 120 s')
    fig.suptitle('Corrected original-input TGPPO pilot vs relpscost',fontsize=13)
    fig.text(.5,.01,'5 development instances; 1 training seed. Dots: means; lines: solver-seed ranges. Not paper-scale reproduction.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(DOC/'development_comparison.svg',bbox_inches='tight')
    fig.savefig(DOC/'development_comparison.png',dpi=180,bbox_inches='tight')
    print('Saved standalone SVG and PNG')


if __name__=='__main__':
    main()
