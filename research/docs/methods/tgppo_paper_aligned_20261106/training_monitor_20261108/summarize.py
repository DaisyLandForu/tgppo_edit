"""Read existing JSON records only; no model imports, solves or training."""
import csv
import datetime as dt
import hashlib
import json
import statistics as stats
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[3] / 'tgppo/runs/paper-aligned-20261106'
now = time.time()
tz = dt.timezone(dt.timedelta(hours=8))
stamp = lambda t: dt.datetime.fromtimestamp(t, tz).isoformat()
read = lambda p: json.loads(p.read_text())
snapshot = dict(observed_beijing=stamp(now), runs={}, interrupt_timelines={})
metrics, windows = [], []
for seed in (11, 22):
    for variant in ('b0', 'mlp', 'mlp_bc'):
        name = '{}-s{}'.format(variant, seed)
        p = RUN / 'training' / name
        budget = read(p/'budget.json')
        attempts = budget['attempts']
        used = sum(a.get('elapsed_seconds', now-a['started_epoch']) for a in attempts)
        rows = []
        for f in sorted((p/'updates').glob('episode-*.json')):
            row = read(f)
            rows.append(row)
            metrics.append(dict(run=name, **row))
        row = dict(state=read(p/'state.json'), budget=budget,
                   estimated_used_seconds=used, remaining_seconds=172800-used,
                   episodes=len(rows), effective_updates=sum(bool(x.get('updated')) for x in rows),
                   ppo_wall_seconds=sum(x.get('learner_wall_seconds', 0) for x in rows))
        if seed == 11:
            # Identical committed prefix, independently of subsequent progress.
            prev = (p/'episode-0000.pt').stat().st_mtime
            phase = dict(episodes=86, precollection_seconds=0.,
                         collector_seconds=0., ppo_seconds=0., collector_cuda_initialized=0)
            for u in rows[:86]:
                episode = u['episode']
                results = sorted((p/'episodes'/str(episode)).glob('attempt-*/collect-*.json'))
                a = results[-1]
                result = read(a)
                phase['precollection_seconds'] += max(0, (a.parent/'job.json').stat().st_mtime-prev)
                phase['collector_seconds'] += result['worker_wall_seconds']
                phase['ppo_seconds'] += u.get('learner_wall_seconds', 0)
                phase['collector_cuda_initialized'] += bool(result.get('cuda_initialized'))
                prev = (p/'updates'/('episode-{:04d}.json'.format(episode))).stat().st_mtime
            row['first86_phase'] = phase
        snapshot['runs'][name] = row
        for lo, hi in ((1,20), (21,40), (41,60), (61,80), (81,100)):
            selected = [x for x in rows if lo <= x['episode'] <= hi and x.get('updated')]
            if not selected:
                continue
            window = dict(run=name, start_episode=lo, end_episode=hi, updates=len(selected))
            for key in ('actor_loss','critic_loss','entropy','clip_fraction','learner_wall_seconds'):
                window[key+'_mean'] = stats.mean(x[key] for x in selected)
                window[key+'_median'] = stats.median(x[key] for x in selected)
            windows.append(window)
for name, episode in (('mlp-s22',48), ('mlp_bc-s22',53)):
    p = RUN/'training'/name
    prev = (p/'updates'/('episode-{:04d}.json'.format(episode-1))).stat().st_mtime
    a = p/'episodes'/str(episode)/'attempt-000'
    result = read(next(a.glob('collect-*.json')))
    job = read(a/'job.json')
    start = (a/'job.json').stat().st_mtime
    pending = (p/'pending.pt').stat().st_mtime
    stop = read(p/'state.json')['stopped_epoch']
    snapshot['interrupt_timelines'][name] = dict(
        episode=episode, previous_commit_beijing=stamp(prev),
        collection_start_beijing=stamp(start), pending_saved_beijing=stamp(pending),
        stopped_beijing=stamp(stop), precollection_seconds=start-prev,
        collection_to_pending_seconds=pending-start, pending_to_stop_seconds=stop-pending,
        baseline_host=job['baseline']['host'], baseline_wall_seconds=job['baseline']['result']['worker_wall_seconds'],
        collector_wall_seconds=result['worker_wall_seconds'], collector_cuda_initialized=result['cuda_initialized'])
baseline_files = list((RUN/'normalization').glob('*/*/baseline.json'))
snapshot['normalization'] = dict(cached_pairs=len(baseline_files), target_pairs=125,
    sum_worker_wall_seconds=sum(read(f)['result']['worker_wall_seconds'] for f in baseline_files))
for filename, rows in [('metrics.csv',metrics), ('loss_windows.csv',windows)]:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/filename).open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
(OUT/'SNAPSHOT.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(snapshot, ensure_ascii=False))
