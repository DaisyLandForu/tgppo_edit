"""Stage 4: fixed-Actor collection and controlled frozen-encoder value diagnosis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import signal
import time
import traceback
import numpy as np
import torch
from project.runtime import (make_job, job_identity, run_jobs, load_agent, seed_all,
                             cpu_seconds)
from project.utils.protocol import fingerprint, sha256, write_json, validate_baseline
from project.search_features import discounted_returns, SCHEMA
from project.policy.search_critic import SearchCritic

ROOT = Path('/root/research')
RUN = ROOT / 'tgppo/runs/stage4'
CHECKPOINT = ROOT / 'tgppo/runs/stage3/pilot-repaired/latest.pt'
CHECKPOINT_SHA = '1b912eec46bdf7ec3eaff04570c4aed0c3a57389b6d290bc14f3cc213442d4d1'
GROUPS = ('R0', 'Rcap', 'Rprod', 'Rjoint')
LOSSES = ('mse', 'hl_gauss')
FIT_SEEDS = (11, 22, 33)


def protocol():
    return json.loads((ROOT / 'docs/stage1/protocol_draft.json').read_text())


def check_checkpoint():
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Frozen stage3 checkpoint changed')


def frozen_config():
    p = protocol()
    return dict(stage=4, authorization='进入阶段4', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=CHECKPOINT_SHA, time_limit=120., max_workers=4, watchdog=180.,
        solver_seeds=[0, 1], train=p['data']['train'], development=p['data']['development'],
        groups=GROUPS, losses=LOSSES, fit_seeds=FIT_SEEDS, max_updates=500, batch_size=256,
        critic_lr=.00012, gamma=.97, affinity=[0, 1, 2, 3],
        target='full finite-horizon empirical discounted return, no bootstrap or GAE',
        cache_device='cpu', fit_device='cuda if available, cached pure tensors only',
        offline_cpu_cap_seconds=14400., encoder_microbatch_size=8,
        initialization='common B0 head + zero time projection; identical extra initialization by fit seed',
        selection='fixed final 500-update head, never choose by development outcomes',
        weighting='uniform train instance, then uniform transition within instance',
        frontend='frozen B0 Critic embeddings/transformer/refinement/aggregation',
        action_features='unchanged upstream Actor inputs', online_training_started=False)


def collect():
    check_checkpoint()
    p = protocol()
    cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    baselines = json.loads((ROOT / 'tgppo/runs/stage3/native/baselines.json').read_text())
    names = p['data']['train'] + p['data']['development']
    jobs = []
    for i, name in enumerate(names):
        instance = ROOT / 'tgppo/datasets/train_instances' / (name+'.mps.gz')
        if sha256(instance) != p['data']['hashes'][name]:
            raise ValueError('Instance hash changed')
        for seed in (0, 1):
            j = make_job('offline-{:02d}-{}-s{}'.format(i, name, seed),
                instance.parent, name, 120., seed, 11, cutoffs, 'collect', CHECKPOINT)
            identity = job_identity(j)
            j['baseline'] = baselines[fingerprint(identity)]
            validate_baseline(j['baseline'], identity)
            j.update(collect_search_features=True, address_space_limit_bytes=8*1024**3)
            jobs.append(j)
    write_json(RUN / 'collection/jobs_manifest.json', jobs)
    rows = run_jobs(jobs, RUN / 'collection/jobs', 4, 180.)
    write_json(RUN / 'collection/summary.json', dict(jobs=len(rows), valid=sum(bool(r.get('valid')) for r in rows),
        transitions=sum(r.get('num_transitions', 0) for r in rows),
        worker_cpu_seconds=sum(r.get('worker_cpu_seconds') or 0 for r in rows),
        parent_cpu_seconds=cpu_seconds(),
        search_extraction_wall_seconds=sum(r.get('search_extraction_wall_seconds', 0) for r in rows),
        checkpoint_sha256_after=sha256(CHECKPOINT)))
    if any(not r.get('valid') for r in rows):
        raise RuntimeError('Collection failures retained; no automatic sample deletion')


def bridge():
    check_checkpoint()
    cutoffs = json.loads((ROOT / 'tgppo/datasets/cutoffs.json').read_text())
    comparisons = []
    for name in ('lseu', 'stein27'):
        j = make_job('bridge-native-'+name, ROOT / 'tgppo/datasets/train_instances', name,
                     120., 0, 11, cutoffs, 'native')
        native = run_jobs([j], RUN/'bridge/native'/name, 1, 180.)[0]
        baseline = dict(identity=job_identity(j), result=native)
        paired = []
        for enabled in (False, True):
            # Identical job ID gives identical action RNG. Separate output directories.
            k = make_job('bridge-actor-'+name, ROOT/'tgppo/datasets/train_instances', name,
                         120., 0, 11, cutoffs, 'collect', CHECKPOINT, baseline)
            k['collect_search_features'] = enabled
            directory = RUN/'bridge'/('enabled' if enabled else 'disabled')/name
            row = run_jobs([k], directory, 1, 180.)[0]
            if not row.get('valid'):
                raise RuntimeError('Bridge invalid: '+str(row))
            with open(directory/(k['job_id']+'.pkl'), 'rb') as f:
                trajectory = pickle.load(f)
            transitions = [t for e in trajectory['episodes'] for t in e['transitions']]
            paired.append((row, transitions, trajectory))
        a, b = paired
        equal = all(a[0][key] == b[0][key] for key in ('nnodes', 'status', 'primal_bound', 'dual_bound', 'gap'))
        actions_equal = [t['action'] for t in a[1]] == [t['action'] for t in b[1]]
        features_equal = len(a[1]) == len(b[1]) and all(
            np.array_equal(x[key], y[key]) for x,y in zip(a[1],b[1])
            for key in ('cands_state','node_state'))
        # MIP state includes solving-time-dependent PDI, so it need not be bit-identical.
        comparisons.append(dict(instance=name, solver_outcome_equal=equal, actions_equal=actions_equal,
            static_features_equal=features_equal, transitions=len(a[1]),
            sidecar_aligned=len(b[2]['search_features']['rows']) == len(b[1])))
        # Full Actor inputs contain real elapsed time/PDI. Identical actions in
        # separate timed runs are diagnostic, not a bitwise read-only guarantee.
        if not comparisons[-1]['sidecar_aligned']:
            raise RuntimeError('Unaligned real search sidecar')
    write_json(RUN/'bridge/verification.json', dict(passed=True, comparisons=comparisons))


def offline_prior_cpu():
    return sum(json.loads(p.read_text()).get('cpu_seconds', 0.)
               for p in (RUN/'offline').glob('*_cost.json'))


def budget_signal(signum, frame):
    raise RuntimeError('Stage4 shared offline CPU cap reached; partial results retained')


def offline_budget():
    prior = offline_prior_cpu()
    available = 14400.-prior
    if available <= 10:
        raise RuntimeError('Offline budget exhausted')
    signal.signal(signal.SIGXCPU, budget_signal)
    resource.setrlimit(resource.RLIMIT_CPU, (int(available-5), resource.getrlimit(resource.RLIMIT_CPU)[1]))
    return prior


def cache():
    prior = offline_budget()
    check_checkpoint()
    start_cpu, start_wall = cpu_seconds(), time.perf_counter()
    out = RUN/'offline'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'cache.pt').exists():
        raise FileExistsError('Preserve existing cache')
    agent, _ = load_agent(CHECKPOINT, training=False, restore_random_state=False)
    agent.critic_network.eval()
    model = SearchCritic(agent.critic_network).eval()
    for par in model.parameters(): par.requires_grad_(False)
    summaries, parts = [], []
    for path in sorted((RUN/'collection/jobs').glob('*.pkl')):
        result = json.loads(path.with_suffix('.json').read_text())
        if not result['valid']: raise ValueError('Invalid source trajectory')
        with path.open('rb') as f: payload = pickle.load(f)
        side = payload['search_features']
        if side['schema'] != SCHEMA: raise ValueError('Search schema mismatch')
        episodes = payload['episodes']
        if len(episodes) != 1: raise ValueError('One finite task per file required')
        ep = episodes[0]; rows = ep['transitions']; extras = side['rows']; n = len(rows)
        if n != len(extras) or n != payload['num_transitions']: raise ValueError('Sidecar length mismatch')
        if not ep['task_terminal'] or ep['collector_truncated'] or ep['bootstrap_value'] != 0:
            raise ValueError('Only complete finite-horizon targets')
        if n and (not rows[-1]['done'] or sum(t['done'] for t in rows) != 1):
            raise ValueError('Missing/duplicate terminal')
        name = result['instance_name']
        summaries.append(dict(job_id=path.stem, instance=name, transitions=n,
            terminal_reward=rows[-1]['reward'] if n else None, source_sha256=sha256(path)))
        if not n: continue
        target = discounted_returns([r['reward'] for r in rows])
        if not np.isfinite(target).all() or target.min() < -36 or target.max() > 52:
            raise ValueError('Frozen return support violated; stop before fitting')
        aggregated = np.empty((n, model.hidden_dim), dtype=np.float32)
        tree = np.empty_like(aggregated)
        ordering = sorted(range(n), key=lambda i: len(rows[i]['cands_state']))
        with torch.no_grad():
            for start in range(0, n, 8):
                ids = ordering[start:start+8]
                cands = torch.nn.utils.rnn.pad_sequence(
                    [torch.from_numpy(rows[i]['cands_state']) for i in ids], batch_first=True)
                lengths = torch.tensor([len(rows[i]['cands_state']) for i in ids])
                mask = torch.arange(cands.shape[1])[None, :] >= lengths[:, None]
                nodes = torch.from_numpy(np.stack([rows[i]['node_state'] for i in ids]))
                mips = torch.from_numpy(np.stack([rows[i]['mip_state'] for i in ids]))
                a, t = model.encode(cands, nodes, mips, mask)
                aggregated[ids], tree[ids] = a.numpy(), t.numpy()
                # Full/base consistency is checked on the first microbatch of every trajectory.
                if start == 0:
                    expected = agent.critic_network(cands, nodes, mips, mask).reshape(-1)
                    predicted = model.forward_cached(a, t, torch.cat([nodes,mips], -1),
                        torch.tensor([[extras[i]['remaining_time']] for i in ids]),
                        torch.from_numpy(np.stack([extras[i]['frontier'] for i in ids])))
                    torch.testing.assert_close(predicted, expected)
        for i,(row, extra) in enumerate(zip(rows, extras)):
            if row['action'] != extra['action'] or extra['decision_index'] != i:
                raise ValueError('Action alignment mismatch')
        parts.append(dict(aggregated=aggregated, tree=tree,
            tree_state=np.stack([np.r_[r['node_state'],r['mip_state']] for r in rows]),
            remaining_time=np.asarray([[x['remaining_time']] for x in extras], dtype=np.float32),
            frontier=np.stack([x['frontier'] for x in extras]), target=target,
            instance=np.asarray([name]*n), job_id=np.asarray([path.stem]*n)))
        print(json.dumps(dict(event='cached', instance=name, transitions=n,
                              cpu_seconds=cpu_seconds()-start_cpu)), flush=True)
    if len(summaries) != 50: raise ValueError('Expected all 50 episode records')
    dataset = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    torch.save(dict(schema=SCHEMA, checkpoint_sha256=CHECKPOINT_SHA, arrays=dataset,
                    episodes=summaries), out/'cache.pt')
    cost = dict(cpu_seconds=cpu_seconds(), execution_cpu_seconds=cpu_seconds()-start_cpu,
                wall_seconds=time.perf_counter()-start_wall,
                startup_cpu_seconds=start_cpu, prior_offline_cpu_seconds=prior,
                transitions=len(dataset['target']), episodes=50, cache_sha256=sha256(out/'cache.pt'),
                target_min=float(dataset['target'].min()), target_max=float(dataset['target'].max()),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(out/'cache_cost.json', cost)


def metrics(prediction, target, train_variance):
    error = prediction-target
    variance = float(np.var(target))
    return dict(mae=float(np.abs(error).mean()), mse=float(np.square(error).mean()),
        nmse=float(np.square(error).mean()/train_variance) if train_variance > 0 else None,
        explained_variance=float(1.-np.var(error)/variance) if variance > 1e-12 else None,
        target_variance=variance, transitions=len(target))


def fit():
    prior = offline_budget()
    check_checkpoint()
    start_cpu, start_wall = cpu_seconds(), time.perf_counter()
    out = RUN/'offline'
    if (out/'fits').exists(): raise FileExistsError('Preserve existing fits')
    (out/'fits').mkdir()
    payload = torch.load(out/'cache.pt', map_location='cpu')
    data = payload['arrays']; p = protocol()
    train_names = [name for name in p['data']['train'] if (data['instance']==name).any()]
    dev_names = [name for name in p['data']['development'] if (data['instance']==name).any()]
    if set(train_names) & set(dev_names): raise ValueError('Instance leakage')
    buckets = [np.flatnonzero(data['instance']==name) for name in train_names]
    mean = float(np.mean([data['target'][idx].mean() for idx in buckets]))
    variance = float(np.mean([np.square(data['target'][idx]-mean).mean() for idx in buckets]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent, _ = load_agent(CHECKPOINT, training=False, restore_random_state=False)
    features = {k: torch.as_tensor(data[k], device=device) for k in
                ('aggregated','tree','tree_state','remaining_time','frontier','target')}
    feature_keys = ('aggregated','tree','tree_state','remaining_time','frontier')
    fit_rows, per_instance, curves, parameters = [], [], [], []
    gpu_ms = 0.
    schedules = {}
    for fit_seed in FIT_SEEDS:
        rng = np.random.RandomState(fit_seed)
        schedule = []
        for step in range(500):
            which = rng.randint(len(buckets), size=256)
            ids = np.asarray([buckets[j][rng.randint(len(buckets[j]))] for j in which], dtype=np.int64)
            schedule.append(ids)
        schedules[fit_seed] = schedule
        np.save(out/('sampling-indices-s{}.npy'.format(fit_seed)), np.stack(schedule))
    # Interleave groups within loss/seed. Never choose head/step using development loss.
    for loss in LOSSES:
        for fit_seed in FIT_SEEDS:
            for group in GROUPS:
                seed_all(fit_seed)
                model = SearchCritic(agent.critic_network, group, loss)
                model.freeze_encoder()
                model.to(device).train()
                # Frozen modules stay in eval mode. No encoder is run during cached fits.
                optimizer = torch.optim.Adam([x for x in model.parameters() if x.requires_grad], lr=.00012)
                initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                parameters.append(dict(group=group, loss=loss, seed=fit_seed,
                    total=sum(x.numel() for x in model.parameters()),
                    trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)))
                fit_cpu, fit_wall = cpu_seconds(), time.perf_counter()
                if device.type == 'cuda':
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                for step, ids in enumerate(schedules[fit_seed]):
                    idx = torch.as_tensor(ids, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.logits_cached(*[features[k][idx] for k in feature_keys])
                    objective = model.training_loss(logits, features['target'][idx])
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], .5)
                    optimizer.step()
                    if (step+1) % 50 == 0:
                        curves.append(dict(group=group, loss=loss, seed=fit_seed, update=step+1,
                                           training_objective=float(objective.detach().cpu())))
                if device.type == 'cuda':
                    end.record(); torch.cuda.synchronize(); gpu_ms += begin.elapsed_time(end)
                # Evaluate exactly once at the predeclared final update.
                model.eval()
                predicted = []
                with torch.no_grad():
                    for start in range(0, len(data['target']), 2048):
                        predicted.append(model.forward_cached(*[features[k][start:start+2048]
                            for k in feature_keys]).cpu().numpy())
                predicted = np.concatenate(predicted)
                if not np.isfinite(predicted).all(): raise ValueError('Nonfinite value prediction')
                np.save(out/'fits'/('{}-{}-s{}-predictions.npy'.format(group,loss,fit_seed)), predicted)
                # Save trainable weights and reference the shared B0 encoder to avoid 24 copies.
                trainable_keys = {name for name,par in model.named_parameters() if par.requires_grad}
                state = {k:v.detach().cpu() for k,v in model.state_dict().items() if k in trainable_keys}
                change = sum((v-initial[k]).double().square().sum().item() for k,v in state.items())**.5
                torch.save(dict(group=group, loss=loss, seed=fit_seed, updates=500,
                    checkpoint_sha256=CHECKPOINT_SHA, cache_sha256=sha256(out/'cache.pt'),
                    trainable_state_dict=state, optimizer=optimizer.state_dict(),
                    training_instance_names=train_names), out/'fits'/('{}-{}-s{}.pt'.format(group,loss,fit_seed)))
                for name in train_names+dev_names:
                    ids = np.flatnonzero(data['instance']==name)
                    per_instance.append(dict(group=group, loss=loss, seed=fit_seed, instance=name,
                        split='train' if name in train_names else 'development',
                        **metrics(predicted[ids],data['target'][ids],variance)))
                row = dict(group=group, loss=loss, seed=fit_seed, updates=500, parameter_l2_change=change,
                    cpu_seconds=cpu_seconds()-fit_cpu, wall_seconds=time.perf_counter()-fit_wall)
                fit_rows.append(row)
                write_json(out/'fit_progress.json', dict(completed=len(fit_rows), total=24, last=row,
                    combined_offline_cpu_seconds=prior+cpu_seconds()-start_cpu))
                print(json.dumps(row), flush=True)
                del model, optimizer
    for filename, value in [('per_instance.json',per_instance),('fit_runs.json',fit_rows),
                            ('curves.json',curves),('parameters.json',parameters)]:
        write_json(out/filename,value)
    constants = [dict(instance=name, split='train' if name in train_names else 'development',
                     **metrics(np.full(len(ids),mean),data['target'][ids],variance))
                 for name in train_names+dev_names for ids in [np.flatnonzero(data['instance']==name)]]
    write_json(out/'constant_predictor.json', dict(training_mean=mean, training_variance=variance,
               per_instance=constants))
    write_json(out/'fit_cost.json', dict(cpu_seconds=cpu_seconds(), execution_cpu_seconds=cpu_seconds()-start_cpu,
        startup_cpu_seconds=start_cpu, wall_seconds=time.perf_counter()-start_wall,
        device=str(device), gpu_fit_event_seconds=gpu_ms/1000.,
        combined_offline_cpu_seconds=prior+cpu_seconds(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
        train_names=train_names, development_names=dev_names, fits=24,
        checkpoint_sha256_after=sha256(CHECKPOINT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['bridge','collect','cache','fit'])
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    global RUN
    if args.run_root: RUN = args.run_root.resolve()
    os.sched_setaffinity(0, [0,1,2,3])
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    config = frozen_config()
    config_path = RUN/'execution_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Frozen stage4 execution config changed')
    write_json(config_path, config)
    try:
        globals()[args.command]()
    except Exception:
        if args.command in ('cache','fit'):
            write_json(RUN/'offline'/('failed-{}-{}_cost.json'.format(args.command,os.getpid())),
                dict(cpu_seconds=cpu_seconds(), failed=True, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
