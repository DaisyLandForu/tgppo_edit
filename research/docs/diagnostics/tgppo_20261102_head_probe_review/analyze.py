"""Read-only audit of the four finished jobs; no forward/backward or optimizer step."""
import csv
import fcntl
import json
import math
import sys
import time
from pathlib import Path

START = time.monotonic()
OUT = Path(__file__).resolve().parent
RESEARCH = OUT.parents[2]
PACKAGE = RESEARCH / 'tgppo/experiments/offline-head-probe-20261102'
RUN = RESEARCH / 'tgppo/runs/offline-head-probe-20261102'
sys.path.insert(0, str(PACKAGE / 'code'))
import torch
import common as c
import runner

c.set_threads()
before = {}


def freeze(path):
    path = path.resolve()
    key = str(path.relative_to(RESEARCH))
    digest = c.sha(path)
    if key in before:
        assert before[key] == digest, key
    before[key] = digest
    return digest


p, m = c.load_prepared(RUN, check_sources=True)
freeze(RUN / 'prepared/manifest.json')
for entry in m['caches'].values():
    freeze(RUN / entry['path'])
for rel in m['input_source_hashes']:
    freeze(RESEARCH / rel)
    if rel.endswith('/delivery_manifest.json'):
        folder = (RESEARCH / rel).parent
        for name, digest in json.loads((RESEARCH / rel).read_text())['files'].items():
            assert freeze(folder / name) == digest
package_seal = PACKAGE / 'docs/PACKAGE_MANIFEST.json'
freeze(package_seal)
for rel, digest in json.loads(package_seal.read_text())['files'].items():
    assert freeze(PACKAGE / rel) == digest, rel
for name in ['actor.py', 'modules.py']:
    assert freeze(RESEARCH / 'tgppo/repo/code/project/policy' / name) == freeze(PACKAGE / 'code/frozen_model' / name)
for path in (RUN / 'summary').iterdir():
    if path.is_file():
        freeze(path)
summary = json.loads((RUN / 'summary/summary.json').read_text())
assert summary['complete']
rows, by_instance, details = [], [], {}
for seed in p['seeds']:
    cache = torch.load(RUN / m['caches'][str(seed)]['path'], map_location='cpu')
    expected_pair = next(x for x in summary['paired_results'] if x['seed'] == seed)
    for variant in p['variants']:
        job = f'{variant}-s{seed}'
        d = RUN / 'jobs' / job
        for path in d.iterdir():
            if path.is_file() and path.name != '.lock':
                freeze(path)
        with (d / '.lock').open('r+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, history = runner.verify_job(RUN, job)
        budget = json.loads((d / 'budget.json').read_text())
        steps = json.loads((d / 'steps.json').read_text())
        cp = torch.load(d / 'latest.pt', map_location='cpu')
        assert budget['exit_code'] == 0 and budget['elapsed_seconds'] < 600
        assert steps['committed_steps'] == cp['step'] == result['completed_steps'] == 20
        assert not (d / 'inflight.json').exists()
        adam_steps = [int(v['step']) for v in cp['optimizer']['state'].values()]
        assert len(adam_steps) == len(cp['head']) and set(adam_steps) == {20}
        initial = c.make_head(cache, variant).state_dict()
        changes = []
        for key, value in cp['head'].items():
            delta = value.double() - initial[key].double()
            changes.append(dict(parameter=key, changed_elements=int((delta != 0).sum()),
                                squared_l2=float(delta.square().sum()), max_abs=float(delta.abs().max())))
        first, last = history[0], history[-1]
        change = last['surrogate_loss'] - first['surrogate_loss']
        assert math.isclose(change, expected_pair[f'{variant}_surrogate_change'], abs_tol=1e-15)
        metrics = list(csv.DictReader((d / 'state_metrics.csv').open()))
        assert len(metrics) == 21 * (256 + 164)
        for step in range(21):
            for dataset, count in [('train', 256), ('monitor', 164)]:
                subset = [x for x in metrics if int(x['step']) == step and x['dataset'] == dataset]
                assert len(subset) == count and len({x['key'] for x in subset}) == count
                tv = sorted(float(x['tv_initial']) for x in subset)
                median = (tv[len(tv)//2-1] + tv[len(tv)//2]) / 2
                assert math.isclose(median, history[step][dataset]['tv_initial_p50'], abs_tol=1e-15)
        for dataset in ['train', 'monitor']:
            final = [x for x in metrics if int(x['step']) == 20 and x['dataset'] == dataset]
            for instance in sorted({x['instance'] for x in final}):
                values = sorted(float(x['tv_initial']) for x in final if x['instance'] == instance)
                n = len(values)
                by_instance.append(dict(job=job, dataset=dataset, instance=instance, states=n,
                                        tv_p50=(values[(n-1)//2]+values[n//2])/2, tv_max=max(values)))
        row = dict(job=job, seed=seed, variant=variant, wall_seconds=budget['elapsed_seconds'], steps=20,
                   surrogate_change=change, train_tv_p50=last['train']['tv_initial_p50'],
                   monitor_tv_p50=last['monitor']['tv_initial_p50'],
                   train_kl=last['train']['mean_kl_initial'], monitor_tv_max=last['monitor']['tv_initial_max'],
                   monitor_response_states=last['monitor']['numerical_response_states'],
                   centered_logit_rms_p50=last['monitor']['centered_logit_delta_rms_p50'],
                   common_logit_shift_p50=last['monitor']['abs_common_logit_shift_p50'],
                   last_gradient_norm=steps['last_gradient_norm'],
                   changed_parameter_elements=sum(x['changed_elements'] for x in changes),
                   parameter_delta_l2=math.sqrt(sum(x['squared_l2'] for x in changes)),
                   visible_nonuniform_states=last['train']['visible_nonuniform_states']+last['monitor']['visible_nonuniform_states'])
        rows.append(row)
        details[job] = dict(overview=row, parameter_changes=changes, adam_parameter_tensors=len(adam_steps),
                            adam_steps_unique=sorted(set(adam_steps)),
                            max_train_tv_over_all_steps=max(h['train']['tv_initial_max'] for h in history),
                            max_monitor_tv_over_all_steps=max(h['monitor']['tv_initial_max'] for h in history),
                            all_clip_fractions_zero=all(h['clip_fraction']==0 for h in history))
        del initial, cp, metrics
    del cache
for rel, digest in before.items():
    assert c.sha(RESEARCH / rel) == digest, f'Source changed during review: {rel}'
assert not torch.cuda.is_initialized() and 'pyscipopt' not in sys.modules
c.write_csv(OUT / 'results.csv', rows)
c.write_csv(OUT / 'per_instance.csv', by_instance)
c.save(OUT / 'input_hashes.json', before)
c.save(OUT / 'analysis.json', details)
c.save(OUT / 'verification.json', dict(status='passed', verified_at_utc=runner.utc(),
    package_seal_verified=True, four_job_seals_verified=True, summary_values_recomputed=True,
    actor_and_modules_match_preserved_original=True,
    state_metric_rows_verified=4*21*420, all_inputs_unchanged=len(before),
    completed_historical_head_steps=80, new_optimizer_steps=0, new_solver_tasks=0,
    cuda_initialized=False, live_job_locks=0, review_seconds=time.monotonic()-START,
    four_job_total_wall_seconds=sum(x['wall_seconds'] for x in rows)))
print(json.dumps(rows, ensure_ascii=False, indent=2))
print('Read-only verification passed; no new optimizer/solver steps.')
