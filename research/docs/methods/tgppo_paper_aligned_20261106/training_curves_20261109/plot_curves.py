"""Read committed training records only; no models, GPU, updates or solver."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
NOW = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
COLORS = dict(b0='#3569b7', mlp='#d58121', mlp_bc='#24936e')
LABELS = dict(b0='B0', mlp='MLP', mlp_bc='MLP+BC')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT/(name+'.'+extension),dpi=150,bbox_inches='tight')
    plt.close(fig)


def collect(run, variant, seed):
    folder = ROOT/'tgppo/runs'/run/'training'/('{}-s{}'.format(variant,seed))
    if not (folder/'state.json').exists():
        return None
    state = read(folder/'state.json')
    n = state.get('next_episode',0)
    rows = []
    for path in sorted((folder/'updates').glob('episode-*.json')):
        raw = path.read_bytes()
        row = json.loads(raw)
        if row['episode'] <= n:
            rows.append(dict(row,run=run,variant=variant,seed=seed,source=str(path),
                source_sha256=hashlib.sha256(raw).hexdigest()))
    errors = [(r['episode'],k) for r in rows for k,v in r.items()
              if isinstance(v,float) and not math.isfinite(v)]
    if errors:
        raise ValueError('Nonfinite committed records: '+str(errors))
    return dict(folder=str(folder),state=state,episodes=n,rows=rows,
        checkpoint100=(folder/'actor-0100.pt').exists(),nonfinite_fields=errors,
        inherited_from=read(folder/'cpu_continuation.json') if (folder/'cpu_continuation.json').exists() else None)


records = {}
for run, variants in [('paper-aligned-20261106',list(LABELS)),
                      ('paper-trainmode-20261108',['b0']),
                      ('paper-recovery-20261109/cpu-aligned',['b0','mlp','mlp_bc']),
                      ('paper-recovery-20261109/cpu-trainmode',['b0'])]:
    for variant in variants:
        for seed in [11,22]:
            result = collect(run,variant,seed)
            if result is not None:
                records[(run,variant,seed)] = result


def line(ax,rows,key,color,label,smooth=10):
    usable = [r for r in rows if r.get('updated') and key in r]
    x, y = [r['episode'] for r in usable],[r[key] for r in usable]
    ax.plot(x,y,color=color,lw=.8,alpha=.3)
    if len(y)>=smooth:
        ax.plot(x[smooth-1:],np.convolve(y,np.ones(smooth)/smooth,mode='valid'),
            color=color,lw=2,label=label)
    else:
        ax.plot(x,y,color=color,marker='o',ms=3,lw=1,label=label)


panels = [('critic_loss','Critic loss (MSE)'),('actor_loss','Actor loss (PPO + entropy)'),
          ('entropy','Policy entropy'),('clip_fraction','PPO clip fraction')]
fig,axes = plt.subplots(2,4,figsize=(17,8),sharex=True)
for ri,seed in enumerate([11,22]):
    for ci,(key,title) in enumerate(panels):
        ax = axes[ri,ci]
        for variant in LABELS:
            d = records[('paper-aligned-20261106',variant,seed)]
            line(ax,d['rows'],key,COLORS[variant],LABELS[variant])
        ax.axvline(100,color='#777',ls=':',alpha=.6)
        ax.set_title(title)
        if ci==0:ax.set_ylabel('Seed '+str(seed))
        if ri==1:ax.set_xlabel('Committed episode')
handles,labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.96))
fig.suptitle('OLD paper-aligned runs: training diagnostics',fontsize=16)
fig.text(.5,.005,'Faint: individual nonempty updates. Bold: trailing10 updates. Dotted: episode100. '
    'These are not solver performance curves.\n'+NOW,ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.91))
save(fig,'01_old_protocol_metrics')

newpanels = [('critic_loss','Critic loss'),('actor_loss','Actor loss'),('clip_fraction','PPO clip fraction'),
             ('entropy','Policy entropy'),('deployment_tv_before_after_mean','Probe TV: before vs after update'),
             ('deployment_tv_to_uniform_after_mean','Probe TV to uniform: after update')]
fig,axes = plt.subplots(2,3,figsize=(14,8))
for ax,(key,title) in zip(axes.flat,newpanels):
    for seed,color in [(11,'#3569b7'),(22,'#d58121')]:
        d = records[('paper-trainmode-20261108','b0',seed)]
        line(ax,d['rows'],key,color,'Seed '+str(seed),smooth=3)
    ax.set(title=title,xlabel='Committed episode')
handles,labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.96))
fig.suptitle('NEW B0 train-mode runs: early training only',fontsize=16)
fig.text(.5,.005,'Faint: individual updates. Bold: trailing3. Probes use at most8 training states per update; '
    'not an independent evaluation.\n'+NOW,ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.91))
save(fig,'02_new_b0_metrics')

summaries = []
allrows = []
for (run,variant,seed),d in records.items():
    good = [r for r in d['rows'] if r.get('updated')]
    s = dict(run=run,variant=variant,seed=seed,episodes=d['episodes'],
        effective_updates=len(good),checkpoint100=d['checkpoint100'],state=d['state'],
        nonfinite_fields=d['nonfinite_fields'],inherited_from=d['inherited_from'])
    s['clip_nonzero_updates'] = sum(r.get('clip_fraction',0)>0 for r in good)
    s['windows'] = []
    for name,chosen in [('first20',good[:20]),('last20',good[-20:]),
                        ('episode81_100',[r for r in good if 81<=r['episode']<=100])]:
        if chosen:
            item = dict(name=name,updates=len(chosen),first_episode=chosen[0]['episode'],last_episode=chosen[-1]['episode'])
            for key in ['critic_loss','actor_loss','entropy','clip_fraction']:
                item[key] = statistics.mean(r[key] for r in chosen)
            s['windows'].append(item)
    if good and 'deployment_tv_to_uniform_after_mean' in good[0]:
        tv = [r['deployment_tv_to_uniform_after_mean'] for r in good]
        s['probe_uniform_tv_after_median'] = statistics.median(tv)
        s['probe_uniform_tv_after_under_1e_6'] = sum(v<1e-6 for v in tv)
        s['probe_updates'] = len(tv)
    summaries.append(s)
    allrows.extend(d['rows'])

fields = list(dict.fromkeys(k for row in allrows for k in row))
with (OUT/'metrics.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=fields)
    writer.writeheader();writer.writerows(allrows)
(OUT/'SNAPSHOT.json').write_text(json.dumps(dict(observed_beijing=NOW,runs=summaries,
    scope='Read-only committed logs. Old/new and CPU continuation records remain separate.',
    scip_runs=0,training_updates=0),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(observed_beijing=NOW,runs=summaries),ensure_ascii=False))
