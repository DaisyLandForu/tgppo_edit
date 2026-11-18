"""Read committed seed11 logs through a captured checkpoint; never train/solve."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNS = ROOT/'tgppo/runs'
LABELS = dict(b0='New B0', mlp='MLP', mlp_bc='MLP+BC')
COLORS = dict(b0='#3869ad', mlp='#db8b2d', mlp_bc='#258675')


def read(path):
    return json.loads(path.read_text())


def lineage(folder, limit):
    """Respect fork cursors; exclude an ancestor's uncommitted update JSON."""
    segments = []
    device = folder/'device_continuation.json'
    cpu = folder/'cpu_continuation.json'
    if device.exists():
        m = read(device)
        segments += lineage(Path(m['source']), m['next_episode'])
    elif cpu.exists():
        m = read(cpu)
        source = Path(m['source_run'])
        # These old parents are stopped and guarded by the live continuation.
        segments += lineage(source, read(source/'state.json')['next_episode'])
    segments.append((folder,limit))
    return segments


def summarize(rows):
    good = [r for r in rows if r['updated']]
    result = dict(episodes=len(rows),effective_updates=len(good),
        zero_decision_episodes=len(rows)-len(good),
        clip_nonzero_updates=sum(r['clip_fraction']>0 for r in good))
    windows = []
    definitions = [('first50episodes',1,50),('last50episodes',rows[-1]['episode']-49,rows[-1]['episode']),
        ('last20episodes',rows[-1]['episode']-19,rows[-1]['episode'])]
    definitions += [('cycle'+str(i),(i-1)*125+1,min(i*125,rows[-1]['episode']))
        for i in range(1,1+(rows[-1]['episode']+124)//125)]
    for label,lo,hi in definitions:
        subset = [r for r in good if lo<=r['episode']<=hi]
        item = dict(label=label,start=lo,end=hi,nonempty_updates=len(subset),complete125=(hi-lo+1)==125)
        for key in ['critic_loss','actor_loss','entropy','clip_fraction','learner_wall_seconds']:
            if subset:
                item[key+'_mean'] = st.mean(r[key] for r in subset)
                item[key+'_median'] = st.median(r[key] for r in subset)
        windows.append(item)
    result['windows'] = windows
    for field in ['deployment_tv_to_uniform_after_mean','deployment_tv_before_after_mean']:
        values = [r[field] for r in good if field in r]
        if values:
            result[field] = dict(count=len(values),mean=st.mean(values),median=st.median(values),
                maximum=max(values),under_1e_6=sum(v<1e-6 for v in values),
                scope='at most8 first-minibatch states per nonempty update; training monitor only')
    return result


def draw_line(ax, rows, field, variant, positive=False):
    selected = [r for r in rows if r['updated'] and field in r]
    x = np.array([r['episode'] for r in selected])
    y = np.array([r[field] for r in selected])
    ax.plot(x,y,color=COLORS[variant],alpha=.17,lw=.65)
    width = 25
    ax.plot(x[width-1:],np.convolve(y,np.ones(width)/width,mode='valid'),
        color=COLORS[variant],lw=2,label=LABELS[variant])
    if positive: ax.set_yscale('log')


def save(fig, name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    snapshots = read(OUT/'CHECKPOINT_SNAPSHOT.json')
    all_rows = []
    analysis = dict(observed_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds'),
        scope='seed11 committed prefixes across explicit device lineage; no new evaluation',
        new_solver_runs=0,new_real_training_updates=0,models={})
    model_rows = {}
    schedule = None
    for item in snapshots:
        variant,limit = item['variant'],item['progress']['next_episode']
        folder = Path(item['path'])
        if not folder.is_absolute(): folder = ROOT.parent/folder
        current_schedule = read(folder/'schedule.json')
        if schedule is None: schedule = current_schedule
        assert current_schedule == schedule and len(schedule)==500
        segments = lineage(folder,limit)
        selected = {}
        sources = []
        for source,ceiling in segments:
            sources.append(dict(directory=str(source),committed_prefix=ceiling))
            for path in sorted((source/'updates').glob('episode-*.json')):
                raw = path.read_bytes()
                row = json.loads(raw)
                if row['episode'] <= min(limit,ceiling):
                    row = dict(row,variant=variant,source=str(path),source_sha256=hashlib.sha256(raw).hexdigest())
                    selected[row['episode']] = row
        assert set(selected) == set(range(1,limit+1)), (variant,'missing/duplicate prefix')
        rows = [selected[i] for i in range(1,limit+1)]
        for row in rows:
            expected = schedule[row['episode']-1]
            assert row['instance']==expected['instance'] and row['solver_seed']==expected['solver_seed']
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        summary = summarize(rows)
        assert summary['effective_updates'] == item['update_counter'] == item['policy_version']
        summary.update(checkpoint=item,sources=sources,all_records_finite=True,
            schedule_verified=True,record_count_matches_checkpoint=True,
            latest_evaluation_episode=100 if variant=='b0' else 200)
        recent = rows[-51:]
        if len(recent)==51:
            elapsed = (Path(recent[-1]['source']).stat().st_mtime-Path(recent[0]['source']).stat().st_mtime)/3600
            summary['recent50_throughput'] = dict(episodes=50,elapsed_hours=elapsed,
                episodes_per_hour=50/elapsed,remaining_to500_estimated_hours=(500-limit)*elapsed/50,
                estimate_only=True,assumption='same future task mix and no interruption; not a deadline guarantee')
        analysis['models'][variant] = summary
        model_rows[variant] = rows
        all_rows.extend(rows)
    fields = list(dict.fromkeys(k for row in all_rows for k in row))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(all_rows)
    (OUT/'ANALYSIS.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.18,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(13,8))
    panels = [('critic_loss','Critic MSE (log scale)',True),('actor_loss','PPO Actor loss + entropy',False),
              ('entropy','Policy entropy (candidate counts vary)',False),('clip_fraction','PPO clip fraction',False)]
    for ax,(field,title,positive) in zip(axes.flat,panels):
        for variant in LABELS: draw_line(ax,model_rows[variant],field,variant,positive)
        ax.set(title=title,xlabel='Committed episode')
        ax.axvline(285,color='#666',lw=1,ls=':',alpha=.6)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle('Seed11 training diagnostics: learning convergence is not established',fontsize=16)
    fig.text(.5,.012,'Faint: nonempty episode updates. Bold: trailing25 update means. '
        'Dotted: MLP/BC CPU-to-GPU resume at episode285.\n'
        'Losses use changing on-policy targets and instance mixes; these are not held-out solver performance curves.\n'
        +analysis['observed_beijing'],ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.9))
    save(fig,'01_training_curves')

    fig,axes = plt.subplots(1,2,figsize=(12,4.4))
    for ax,field,title in zip(axes,['deployment_tv_to_uniform_after_mean','deployment_tv_before_after_mean'],
            ['B0: distance from uniform policy','B0: deployment change per PPO update']):
        rows = [r for r in model_rows['b0'] if r['updated'] and field in r]
        ax.scatter([r['episode'] for r in rows],[max(r[field],1e-12) for r in rows],s=9,color=COLORS['b0'],alpha=.6)
        ax.axhline(1e-6,color='#a34a48',ls='--',lw=1,label='1e-6 reference')
        ax.set(xlabel='Committed episode',ylabel='Mean total variation (TV)',title=title,yscale='log',ylim=(5e-13,3e-6))
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('B0 remains effectively uniform on its logged training-state probes',fontsize=14)
    fig.text(.5,.015,'At most8 states per nonempty update; no independent evaluation. '
        'Exact zeros displayed at1e-12.\n'+analysis['observed_beijing'],ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.92))
    save(fig,'02_b0_policy_probes')
    print(json.dumps({v:{k:s[k] for k in ['episodes','effective_updates','zero_decision_episodes','clip_nonzero_updates']}
        for v,s in analysis['models'].items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
