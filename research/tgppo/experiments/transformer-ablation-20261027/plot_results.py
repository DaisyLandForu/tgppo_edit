"""Plot actual loss/reward histories with the separate reporting environment."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parents[2]/'runs/transformer-ablation-20261027')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'docs/training-curves')
    args=p.parse_args()
    series=[]
    for variant in ('transformer','bimatching'):
        for seed in (11,22,33):
            out=args.run_root/variant/'training'/('B0-s'+str(seed))
            if not (out/'updates.json').exists():continue
            updates=json.loads((out/'updates.json').read_text())
            rows=[r for f in sorted(out.glob('collection/*/selected_results.json')) for r in json.loads(f.read_text())]
            if any(not r.get('valid') for r in rows):raise ValueError('Invalid training episode')
            series.append((variant,seed,updates,rows))
    if not series:raise SystemExit('No actual training histories; no curves fabricated.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for variant,seed,updates,rows in series:
        label=variant+'/s'+str(seed);style='-' if variant=='transformer' else '--'
        color={11:'tab:blue',22:'tab:orange',33:'tab:green'}[seed]
        for axis,key in zip(axes.flat[:4],('actor_loss','critic_loss','entropy','clip_fraction')):
            samples=[u for u in updates if u.get('updated')]
            axis.plot([u['episodes_seen'] for u in samples],[u[key] for u in samples],style,color=color,label=label)
            axis.set_title(key);axis.set_xlabel('Completed episodes');axis.grid(alpha=.2)
        axes.flat[4].plot(range(1,len(rows)+1),[r['total_reward'] for r in rows],style,color=color,label=label)
        axes.flat[5].plot(range(1,len(rows)+1),[r['num_transitions'] for r in rows],style,color=color,label=label)
    axes.flat[4].set_title('Episode total H3 reward');axes.flat[5].set_title('Episode transitions')
    for axis in axes.flat[4:]:axis.set_xlabel('Completed episodes');axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf','.svg'):fig.savefig(args.output.with_suffix(suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
