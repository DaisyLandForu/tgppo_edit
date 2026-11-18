"""Plot existing committed records. No model imports, training, or SCIP solves."""
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[3] / 'tgppo/runs/paper-aligned-20261106/training'
NOW = time.time()
ZONE = dt.timezone(dt.timedelta(hours=8))
STAMP = dt.datetime.fromtimestamp(NOW, ZONE).strftime('%Y-%m-%d %H:%M BJT')
COLORS = dict(b0='#3569b7', mlp='#d58121', mlp_bc='#24936e')
LABELS = dict(b0='B0', mlp='MLP', mlp_bc='MLP+BC')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.grid':True, 'grid.alpha':.16, 'pdf.fonttype':42})


def read(path):
    return json.loads(path.read_text())


def active_seconds(attempts, start, stop):
    return sum(max(0., min(stop, a.get('finished_epoch', NOW))-
                   max(start, a['started_epoch'])) for a in attempts)


def save_figure(fig, name):
    for extension in ('png','pdf','svg'):
        fig.savefig(OUT/(name+'.'+extension), dpi=160, bbox_inches='tight')
    plt.close(fig)


data, all_rows, windows = {}, [], []
for seed in (11,22):
    for variant in LABELS:
        name = '{}-s{}'.format(variant,seed)
        p = RUN/name
        state, ledger = read(p/'state.json'), read(p/'budget.json')
        committed = state.get('next_episode',0)
        attempts = ledger['attempts']
        rows = []
        for f in sorted((p/'updates').glob('episode-*.json')):
            raw = f.read_bytes()
            u = json.loads(raw)
            if u['episode'] > committed:
                continue  # Update JSON can precede the atomic checkpoint commit.
            row = dict(run=name, variant=variant, training_seed=seed, **u,
                       committed_epoch=f.stat().st_mtime,
                       source_sha256=hashlib.sha256(raw).hexdigest())
            row['active_hours'] = active_seconds(attempts, 0, row['committed_epoch'])/3600
            rows.append(row)
            all_rows.append(row)
        if rows and [r['episode'] for r in rows] != list(range(1,len(rows)+1)):
            raise ValueError('Incomplete episode record prefix: '+name)
        finite_errors = [(r['episode'],k) for r in rows for k,v in r.items()
                         if isinstance(v,float) and not math.isfinite(v)]
        used = active_seconds(attempts,0,NOW)
        data[name] = dict(variant=variant, seed=seed, state=state, rows=rows,
                          attempts=attempts, active_hours=used/3600,
                          remaining_hours=(172800-used)/3600,
                          nonfinite_fields=finite_errors,
                          effective_updates=sum(bool(r.get('updated')) for r in rows))
        for lo in range(1,len(rows)+1,20):
            chosen = [r for r in rows if lo<=r['episode']<=lo+19 and r.get('updated')]
            if chosen:
                row = dict(run=name, first_episode=lo, last_episode=min(lo+19,len(rows)),
                           effective_updates=len(chosen))
                for key in ('actor_loss','critic_loss','entropy','clip_fraction'):
                    row[key+'_mean'] = statistics.mean(r[key] for r in chosen)
                windows.append(row)

# Per-episode noise is retained; the smooth line has exactly ten nonempty updates.
fig, axes = plt.subplots(2,4,figsize=(18,8),sharex=True)
panels = [('critic_loss','Critic loss (MSE)',1),
          ('actor_loss','Actor loss (PPO + entropy)',1),
          ('entropy','Policy entropy',1),
          ('clip_fraction','PPO clipped samples (%)',100)]
maximum = max(len(d['rows']) for d in data.values())
for row_index, seed in enumerate((11,22)):
    for column, (key,title,scale) in enumerate(panels):
        ax = axes[row_index,column]
        for variant in LABELS:
            rows = [r for r in data['{}-s{}'.format(variant,seed)]['rows'] if r.get('updated')]
            x = np.array([r['episode'] for r in rows])
            y = np.array([r[key]*scale for r in rows])
            ax.plot(x,y,color=COLORS[variant],alpha=.16,lw=.9)
            smooth = [np.mean(y[i-9:i+1]) if i>=9 else np.nan for i in range(len(y))]
            ax.plot(x,smooth,color=COLORS[variant],lw=2,label=LABELS[variant])
        ax.set_title(title)
        ax.set_xlim(0,maximum+2)
        if column==0:
            ax.set_ylabel('Seed {}'.format(seed))
        if row_index==1:
            ax.set_xlabel('Completed episode')
        if key=='clip_fraction':
            ax.set_ylim(bottom=-.5)
fig.suptitle('TGPPO paper-aligned training: diagnostic metrics',fontsize=17,y=1.01)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
fig.text(.5,-.02,'Faint: individual episodes. Bold: trailing mean of 10 nonempty updates. '
         'Zero-decision episodes excluded; blank tails are unfinished, not zero.\n'+STAMP,
         ha='center',fontsize=10)
fig.tight_layout(rect=(0,.02,1,.94))
save_figure(fig,'01_training_metrics')

# Progress excludes the downtime between allocations. Cost uses an identical prefix.
common = min(len(data[v+'-s11']['rows']) for v in LABELS)
phases = {}
for variant in LABELS:
    d = data[variant+'-s11']
    p = RUN/(variant+'-s11')
    previous = (p/'episode-0000.pt').stat().st_mtime
    initial = previous
    phase = dict(precollection=0.,cpu_rollout=0.,ppo_update=0.,other=0.)
    for row in d['rows'][:common]:
        ep = row['episode']
        valid = []
        for f in sorted((p/'episodes'/str(ep)).glob('attempt-*/collect-*.json')):
            result = read(f)
            if result.get('valid'):
                valid.append((f,result))
        if not valid:
            raise ValueError('Missing valid collection: '+str(ep))
        f, result = valid[-1]
        if result.get('cuda_initialized'):
            raise ValueError('Expected a CPU-only collector')
        start = (f.parent/'job.json').stat().st_mtime
        phase['precollection'] += active_seconds(d['attempts'],previous,start)
        phase['cpu_rollout'] += result['worker_wall_seconds']
        phase['ppo_update'] += row.get('learner_wall_seconds',0.)
        previous = row['committed_epoch']
    residual = active_seconds(d['attempts'],initial,previous)-sum(phase.values())
    if residual < -5:
        raise ValueError('Stage clock inconsistency: '+variant)
    phase['other'] = max(0.,residual)
    phases[variant] = {key:value/3600 for key,value in phase.items()}

fig, axes = plt.subplots(1,2,figsize=(14,6))
for name,d in data.items():
    x = [0.]+[r['active_hours'] for r in d['rows']]
    y = [0]+[r['episode'] for r in d['rows']]
    style = '-' if d['seed']==11 else '--'
    axes[0].step(x,y,where='post',color=COLORS[d['variant']],ls=style,
                 label=LABELS[d['variant']]+' / seed '+str(d['seed']))
    axes[0].plot([x[-1],d['active_hours']],[y[-1],y[-1]],color=COLORS[d['variant']],ls=style,alpha=.5)
    stopped = d['state']['status'].startswith(('stopped','failed'))
    axes[0].scatter(d['active_hours'],y[-1],marker='x' if stopped else 'o',
                    color='#be3030' if stopped else COLORS[d['variant']],s=45)
    axes[0].annotate(str(y[-1]),(d['active_hours'],y[-1]),xytext=(4,4),textcoords='offset points',fontsize=8)
axes[0].set(xlabel='Cumulative allocation time (hours; paused gaps excluded)',
            ylabel='Completed episodes',title='Progress toward fixed 500-episode training')
axes[0].legend(fontsize=8,loc='upper left',ncol=2,frameon=False)
left = np.zeros(3)
for key,label,color in [('precollection','Pre-rollout: normalization / wait','#c6a768'),
                         ('cpu_rollout','CPU solver + rollout','#6b94ba'),
                         ('ppo_update','PPO learner phase','#73a58b'),
                         ('other','Import / save / other','#c7c9cc')]:
    vals = np.array([phases[v][key] for v in LABELS])
    axes[1].barh(list(LABELS.values()),vals,left=left,color=color,label=label)
    for i,val in enumerate(vals):
        if val>=1:
            axes[1].text(left[i]+val/2,i,'{:.2f} h'.format(val),ha='center',va='center',fontsize=9)
    left += vals
axes[1].invert_yaxis()
axes[1].set(xlabel='Hours',title='Same first {} episodes, seed 11'.format(common))
axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False)
fig.suptitle('Training progress and measured stage costs',fontsize=15)
fig.text(.5,-.01,'Red x: stopped in the recorded state. Learner wall time is not GPU utilization. '
         'Current unfinished rollout excluded from stage bars.\n'+STAMP,ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.93))
save_figure(fig,'02_progress_and_cost')

for name,rows in [('metrics.csv',all_rows),('windows.csv',windows)]:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
snapshot = dict(observed_beijing=STAMP, common_seed11_prefix=common,phases_hours=phases,
                runs={name:{k:v for k,v in d.items() if k!='rows'} for name,d in data.items()})
(OUT/'SNAPSHOT.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(observed_beijing=STAMP,common_seed11_prefix=common,
    runs={name:dict(episodes=len(d['rows']),effective_updates=d['effective_updates'],
        status=d['state']['status'],nonfinite_fields=d['nonfinite_fields']) for name,d in data.items()},
    phases_hours=phases),ensure_ascii=False))
