"""Read-only comparison of historical and current B0 update records."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT/'tgppo/runs'
rows = []
for family in ('light-actor-20261105/b0','candidate-residual-20261104/b0','server6-20261022'):
    for seed in (11,22):
        p = RUN/family/'training'/('B0-s'+str(seed))
        config = json.loads((p/'config.json').read_text())
        updates = json.loads((p/'updates.json').read_text())
        valid = [u for u in updates if u.get('updated')]
        rows.append(dict(run=str(p.relative_to(RUN)),time_limit=config['task_time_limit'],
            episodes=max(u.get('episodes_seen',0) for u in updates),valid_updates=len(valid),
            optimizer_steps=sum(u['num_updates'] for u in valid),
            nonzero_clip_updates=sum(u.get('clip_fraction',0)!=0 for u in valid),
            max_clip_fraction=max(u.get('clip_fraction',0) for u in valid),
            advantage_std_min=min(u['advantage_std_before_normalization'] for u in valid)))
for seed in (11,22):
    p = RUN/'paper-aligned-20261106/training'/('b0-s'+str(seed))
    state = json.loads((p/'state.json').read_text())
    updates = [json.loads(f.read_text()) for f in sorted((p/'updates').glob('episode-*.json'))]
    updates = [u for u in updates if u['episode']<=state['next_episode']]
    valid = [u for u in updates if u.get('updated')]
    rows.append(dict(run=str(p.relative_to(RUN)),time_limit=3600,episodes=len(updates),
        valid_updates=len(valid),optimizer_steps=sum(u['num_updates'] for u in valid),
        nonzero_clip_updates=sum(u.get('clip_fraction',0)!=0 for u in valid),
        max_clip_fraction=max(u.get('clip_fraction',0) for u in valid),
        advantage_std_min=min(u['advantage_std_before_normalization'] for u in valid)))
with (OUT/'LOG_COMPARISON.csv').open('w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
source = dict(author_repo='https://github.com/benmhamed-a/tgppo-aaai-26',
    author_blobs=dict(actor_py='11f69657c620a73ed6ee004f33f53e639ba9105f',
                      modules_py='1d26fe8739393bcca06160db20c6929cc30107bb'),files=[])
for base in ('tgppo/repo/code/project/policy','tgppo/experiments/paper-aligned-20261106/code/project/policy',
             'tgppo/experiments/light-actor-20261105/code/project/policy','tgppo/experiments/server-20261022/code/project/policy'):
    for name in ('actor.py','modules.py'):
        p = ROOT/base/name;raw=p.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        expected=source['author_blobs'][name.replace('.','_')]
        assert blob==expected
        source['files'].append(dict(path=str(p.relative_to(ROOT)),git_blob=blob,
                                    sha256=hashlib.sha256(raw).hexdigest(),matches_author=True))
(OUT/'AUTHOR_SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
(OUT/'LOG_COMPARISON.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False))
