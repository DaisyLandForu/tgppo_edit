"""Exact original learn(), interrupted after one minibatch per seed.

Only independent checkpoint copies are updated. A durable step ledger prevents
accidental repeats. No runtime/environment/SCIP modules are imported.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

START = time.perf_counter()
import numpy as np
import torch
from torch.distributions import Categorical

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
SOURCE = ROOT / 'tgppo/experiments/server-20261022/code/project'
PRIOR = OUT.parent / 'tgppo_20261102_signal_localization'
RUN = ROOT / 'tgppo/runs/server6-20261022/training'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(OUT / 'model_copy'))
from project.agent.agent import Agent
from project.policy.actor import Actor
from project.policy.critic import Critic


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write(name, rows):
    if not rows:
        return
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def guard():
    assert time.perf_counter() - START < 465, 'Internal compute stop; retain delivery budget'


def quant(values):
    vals = [float(x) for x in values if x is not None]
    return dict(zip(['min', 'p50', 'p95', 'max'], map(float, np.quantile(vals, [0, .5, .95, 1])))) if vals else {}


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def norm(t):
    return math.sqrt(float(t.detach().double().square().sum()))


def state_summary(optimizer, net):
    result = {}
    for name, p in net.named_parameters():
        state = optimizer.state.get(p, {})
        result[name] = dict(step=float(state.get('step', 0)),
                            exp_avg_l2=norm(state['exp_avg']) if 'exp_avg' in state else None,
                            exp_avg_sq_l2=norm(state['exp_avg_sq']) if 'exp_avg_sq' in state else None,
                            state_present=bool(state))
    return result


def gradient_rows(model, kind, stage, net):
    groups = defaultdict(list)
    for name, p in net.named_parameters():
        groups[name.split('.')[0]].append(p)
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (model, kind, name)
    result = []
    for module, params in groups.items():
        gs = [p.grad for p in params if p.grad is not None]
        result.append(dict(model=model, network=kind, stage=stage, module=module,
                           gradient_l2=math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs)),
                           parameter_tensors=len(params), tensors_with_grad=len(gs),
                           gradient_max_abs=max((float(g.detach().abs().max()) for g in gs), default=0.)))
    return result


def distribution(p):
    a = p.detach().cpu().numpy().astype(np.float64)
    assert np.isfinite(a).all() and (a > 0).all()
    return a / a.sum()


def evaluate_batch(actor, batch):
    """Use the exact same length-sorted chunks as the original learn()."""
    actor.eval()
    lengths = (~batch['mask']).sum(1).cpu().numpy()
    order = np.argsort(lengths, kind='stable')
    result = {}
    captured = {}

    def hook(module, inp, value):
        captured['logits'] = value.detach().clone()

    handle = actor.output_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(order), 16):
                guard()
                ids = order[start:start + 16]
                width = int(lengths[ids].max())
                prob = actor(batch['cands'][ids, :width], batch['node_state'][ids],
                             batch['mip_state'][ids], batch['mask'][ids, :width])
                dist = Categorical(prob)
                logps = dist.log_prob(batch['action'][ids])
                entropies = dist.entropy()
                for j, pos in enumerate(ids):
                    L = int(lengths[pos])
                    result[int(pos)] = dict(p=distribution(prob[j, :L]),
                                            logits=captured['logits'][j, :L].double().numpy(),
                                            log_prob=float(logps[j]), entropy=float(entropies[j]))
    finally:
        handle.remove()
    return result


def evaluate_monitors(actor, observations):
    actor.eval()
    result = {}
    captured = {}

    def hook(module, inp, value):
        captured['logits'] = value.detach().clone()

    handle = actor.output_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for key, instance, row in observations:
                guard()
                x, n, m = [torch.as_tensor(row[k], dtype=torch.float32)[None] for k in ['cands_state', 'node_state', 'mip_state']]
                prob = actor(x, n, m)[0]
                dist = Categorical(prob)
                result[key] = dict(p=distribution(prob), logits=captured['logits'][0].double().numpy(),
                                   log_prob=float(dist.log_prob(torch.tensor(row['action']))),
                                   entropy=float(dist.entropy()))
    finally:
        handle.remove()
    return result


def tv(p, q):
    return float(.5 * np.abs(p - q).sum())


def compare(model, dataset, metadata, before, repeat, after):
    records = []
    for key, meta in metadata.items():
        p, q = before[key]['p'], after[key]['p']
        noise = tv(p, repeat[key]['p'])
        change = tv(p, q)
        L = len(p)
        old_logits, new_logits = before[key]['logits'], after[key]['logits']
        delta = new_logits - old_logits
        uniform = np.full(L, 1 / L)
        old_logprob = meta.get('old_log_prob')
        records.append(dict(model=model, dataset=dataset, state_key=str(key), instance=meta['instance'], L=L,
                            global_index=meta.get('global_index'),
                            pre_tv_uniform=tv(p, uniform), post_tv_uniform=tv(q, uniform),
                            pre_post_tv=change, pre_repeat_tv=noise,
                            above_repeat_noise=change > noise,
                            above_conservative_float32_screen=change > max(10 * noise, 1e-6),
                            pre_post_kl=max(0., float(np.sum(p * np.log(p / q)))),
                            pre_normalized_entropy=(math.log(L) - max(0., float(np.sum(p * np.log(p * L))))) / math.log(L) if L > 1 else None,
                            post_normalized_entropy=(math.log(L) - max(0., float(np.sum(q * np.log(q * L))))) / math.log(L) if L > 1 else None,
                            mean_logit_shift=float(delta.mean()),
                            centered_logit_delta_rms=float(np.sqrt(np.mean((delta - delta.mean()) ** 2))),
                            pre_logit_spread=float(np.ptp(old_logits)), post_logit_spread=float(np.ptp(new_logits)),
                            pre_action_logprob=before[key]['log_prob'], post_action_logprob=after[key]['log_prob'],
                            old_action_logprob=old_logprob,
                            pre_old_ratio=math.exp(before[key]['log_prob'] - old_logprob) if old_logprob is not None else None,
                            post_old_ratio=math.exp(after[key]['log_prob'] - old_logprob) if old_logprob is not None else None,
                            post_pre_action_ratio=math.exp(after[key]['log_prob'] - before[key]['log_prob'])))
    return records


def loss_probe(evaluated, batch, advantages, entropy_weight, clip):
    # Full-batch recomputation is a descriptive check; actual weighted microbatch
    # losses are read from the original learn() callback without changing math.
    logps = torch.tensor([evaluated[i]['log_prob'] for i in range(len(evaluated))])
    ratios = torch.exp(logps - batch['log_prob'])
    a = advantages[batch['indices']]
    surrogate = -torch.min(ratios * a, ratios.clamp(1 - clip, 1 + clip) * a).mean()
    entropy = float(np.mean([evaluated[i]['entropy'] for i in range(len(evaluated))]))
    return dict(surrogate_loss=float(surrogate), entropy=entropy,
                entropy_loss=-entropy_weight * entropy,
                total_actor_loss=float(surrogate) - entropy_weight * entropy,
                sampled_action_ratio=quant(ratios.numpy()),
                clip_fraction=float(((ratios - 1).abs() > clip).float().mean()))


class StopAfterOneMinibatch(Exception):
    pass


def main():
    assert not json.loads((OUT / 'optimizer_steps.json').read_text())['entries'], 'Never replay optimizer steps twice'
    protocol_hash = sha(OUT / 'frozen_protocol.json')
    sources = json.loads((PRIOR / 'runtime_sources_after.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in sources.items())
    copies = {str(p.relative_to(OUT)):sha(p) for p in (OUT / 'model_copy/project').rglob('*.py')}
    for rel, h in copies.items():
        assert sha(SOURCE / str(Path(rel).relative_to('model_copy/project'))) == h
    history = {}
    for directory in ['tgppo_20261102_signal_localization', 'tgppo_20261102_local_validation']:
        folder = OUT.parent / directory
        manifest = json.loads((folder / 'delivery_manifest.json').read_text())['files']
        assert all(sha(folder / p) == h for p, h in manifest.items()), directory
        history[directory] = dict(manifest_sha256=sha(folder / 'delivery_manifest.json'), files=len(manifest))
    observations = pickle.loads((PRIOR / 'observations.pkl').read_bytes())
    assert len(observations) == 164
    save('runtime_sources_before.json', sources)
    old_specs = json.loads((PRIOR / 'frozen_inputs.json').read_text())['checkpoint_specs']
    specs = [r for r in old_specs if r['episode'] == 100]
    assert len(specs) == 2
    input_hashes = {}
    paths_by_model = {}
    for spec in specs:
        paths = []
        folder = RUN / spec['model'] / 'collection/e00100/attempt-000'
        for episode in range(100, 104):
            matches = list(folder.glob(f'e{episode:05d}-*.pkl'))
            assert len(matches) == 1, (episode, matches)
            p = matches[0]
            paths.append(p)
            for source in [p, p.with_suffix('.json')]:
                input_hashes[str(source.relative_to(ROOT))] = sha(source)
        paths_by_model[spec['model']] = paths
        config_path = RUN / spec['model'] / 'config.json'
        input_hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
    save('frozen_inputs.json', dict(protocol_sha256=protocol_hash,
                                   source_plan_sha256=sha(OUT.parent / 'tgppo_20261102_ppo_replay_plan/PLAN.md'),
                                   model_copy_sha256=copies, original_sources=sources, checkpoint_specs=specs,
                                   trajectory_and_config_hashes=input_hashes, prior_deliveries=history,
                                   monitor_pool_sha256=sha(PRIOR / 'observations.pkl')))

    all_advantages, all_batch, all_responses = [], [], []
    all_gradients, all_clips, all_parameters, all_deltas = [], [], [], []
    summaries, imports = [], []
    for spec in specs:
        guard()
        model = spec['model']
        checkpoint = ROOT / spec['path']
        assert sha(checkpoint) == spec['sha256']
        cp = torch.load(checkpoint, map_location='cpu')
        required = ['actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'memory', 'rng', 'policy_version', 'config']
        assert all(k in cp for k in required), 'Original optimizer/RNG checkpoint fields missing'
        cfg = cp['config']
        assert 'online_group' not in cfg and cfg.get('server_group') == 'B0'
        assert cfg == json.loads((RUN / model / 'config.json').read_text())
        assert cfg['batch_size'] == 256 and cfg['n_epochs'] == 3
        arch = dict(cfg['state_dims'], **{k: cfg[k] for k in ['hidden_dim', 'num_heads', 'num_layers', 'dropout']})
        actor, critic = Actor(**arch), Critic(**arch)
        actor.load_state_dict(cp['actor'])
        critic.load_state_dict(cp['critic'])
        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
        actor_opt.load_state_dict(cp['actor_optimizer'])
        critic_opt.load_state_dict(cp['critic_optimizer'])
        agent = Agent(actor, actor_opt, critic, critic_opt, device='cpu', state_dims=cfg['state_dims'],
                      **{k: cfg[k] for k in ['policy_clip', 'entropy_weight', 'gamma', 'gae_lambda', 'batch_size', 'n_epochs']})
        agent.policy_version = cp['policy_version']
        agent.update_counter = cp['update_counter']
        agent.memory.import_dict(cp['memory'])
        row_meta = []
        for ep in agent.memory.episodes:
            for i, row in enumerate(ep['transitions']):
                row_meta.append(dict(episode=ep['episode_id'], transition=i, instance='checkpoint-memory'))
        for path in paths_by_model[model]:
            meta = json.loads(path.with_suffix('.json').read_text())
            payload = pickle.loads(path.read_bytes())
            assert meta['valid'] and meta['policy_version'] == agent.policy_version
            assert len(payload['episodes']) == 1
            ep = payload['episodes'][0]
            assert ep['policy_version'] == agent.policy_version
            assert ep['task_terminal'] and not ep['collector_truncated']
            assert ep['bootstrap_value'] == 0
            assert len(ep['transitions']) == meta['num_transitions']
            agent.memory.import_dict(payload)
            imports.append(dict(model=model, episode=ep['episode_id'], instance=meta['instance_name'],
                                transitions=len(ep['transitions']), policy_version=ep['policy_version'],
                                solver_seed=meta['solver_seed'], task_terminal=ep['task_terminal'], bootstrap=ep['bootstrap_value']))
            for i, row in enumerate(ep['transitions']):
                row_meta.append(dict(episode=ep['episode_id'], transition=i, instance=meta['instance_name']))
        rows = agent.memory.rows()
        assert len(row_meta) == len(rows) >= 256
        raw, returns = agent.compute_targets()
        assert torch.isfinite(raw).all() and torch.isfinite(returns).all()
        raw_std = raw.std(unbiased=False)
        assert float(raw_std) > 0, 'Exactly degenerate real advantages: stop'
        advantages = (raw - raw.mean()) / raw_std.clamp_min(1e-8)
        for i, (row, meta) in enumerate(zip(rows, row_meta)):
            all_advantages.append(dict(model=model, global_index=i, **meta, L=len(row['cands_state']),
                                       reward=row['reward'], saved_value=row['value'], done=row['done'],
                                       raw_advantage=float(raw[i]), normalized_advantage=float(advantages[i]),
                                       return_target=float(returns[i])))
        restore_rng(cp['rng'])
        original_rng = rng_snapshot()
        batch = next(agent.memory.get_batch_generator())
        assert len(batch['indices']) == 256
        batch_indices = batch['indices'].copy()
        metadata = {}
        for pos, index in enumerate(batch_indices):
            meta = row_meta[index]
            row = rows[index]
            all_batch.append(dict(model=model, position=pos, global_index=int(index), **meta,
                                  L=len(row['cands_state']), action=row['action'],
                                  raw_advantage=float(raw[index]), normalized_advantage=float(advantages[index]),
                                  return_target=float(returns[index]), old_log_prob=row['log_prob']))
            metadata[pos] = dict(instance=meta['instance'], global_index=int(index), old_log_prob=row['log_prob'])
        save(f'frozen_batches/{model}.json', dict(indices=batch_indices.tolist(), policy_version=agent.policy_version,
                                                full_buffer_transitions=len(rows), logical_size=256,
                                                original_rng_source='checkpoint CPU Python/NumPy/Torch', cuda_rng_restored=False))
        before_parameters = {kind:{k:p.detach().clone() for k,p in net.named_parameters()}
                             for kind,net in [('actor',actor),('critic',critic)]}
        opt_before = {kind:state_summary(opt,net) for kind,opt,net in [('actor',actor_opt,actor),('critic',critic_opt,critic)]}
        coverage = {}
        for kind, opt, net in [('actor',actor_opt,actor),('critic',critic_opt,critic)]:
            network_ids = {id(p) for p in net.parameters()}
            optimizer_ids = [id(p) for group in opt.param_groups for p in group['params']]
            assert set(optimizer_ids) == network_ids and len(optimizer_ids) == len(network_ids)
            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v,torch.Tensor):
                        assert v.device.type == 'cpu' and torch.isfinite(v).all()
            coverage[kind] = dict(parameter_tensors=len(network_ids), state_tensors=len(opt.state),
                                  param_group_count=len(opt.param_groups), learning_rate=opt.param_groups[0]['lr'],
                                  restored=True, state_steps_before=quant(v['step'] for v in opt_before[kind].values()))

        # All diagnostic forwards precede restoring original_rng again. They
        # cannot consume the dropout or minibatch RNG used by learn().
        before_batch = evaluate_batch(actor,batch)
        repeat_batch = evaluate_batch(actor,batch)
        old_logps = batch['log_prob']
        actual_logps = torch.tensor([before_batch[i]['log_prob'] for i in range(256)])
        pre_ratio_error = float((torch.exp(actual_logps-old_logps)-1).abs().max())
        assert pre_ratio_error <= 1e-4, 'Whole-minibatch old/new probability mismatch'
        before_monitor = evaluate_monitors(actor,observations)
        repeat_monitor = evaluate_monitors(actor,observations)
        pre_loss = loss_probe(before_batch,batch,advantages,cfg['entropy_weight'],cfg['policy_clip'])
        learned = {}
        original_generator = agent.memory.get_batch_generator

        def only_first_generator():
            emitted = False
            for candidate in original_generator():
                assert not emitted, 'A second minibatch is forbidden'
                assert np.array_equal(candidate['indices'],batch_indices), 'Diagnostic RNG changed the selected minibatch'
                emitted = True
                yield candidate
                raise AssertionError('The original progress callback must stop before this point')

        def stop_callback(progress):
            # Read-only measurement of original learn() locals, after both
            # step calls; no transformed copy of the PPO math is executed.
            frame = sys._getframe(1)
            assert frame.f_code.co_name == 'learn'
            local = frame.f_locals
            assert local['updates'] == 1
            assert torch.equal(local['raw_advantages'],raw) and torch.equal(local['advantages'],advantages)
            learned.update(progress=progress, actual_weighted_losses=dict(local['totals']),
                           initial_ratio_max_error=local['initial_ratio_error'],
                           full_buffer_samples=local['n'])
            raise StopAfterOneMinibatch()

        original_clip = torch.nn.utils.clip_grad_norm_
        identities = {frozenset(id(p) for p in net.parameters()):(kind,net) for kind,net in [('actor',actor),('critic',critic)]}

        def measured_clip(parameters,max_norm,*args,**kwargs):
            parameters = list(parameters)
            kind, net = identities[frozenset(id(p) for p in parameters)]
            pre = gradient_rows(model,kind,'before_clip',net)
            assert sum(r['gradient_l2']**2 for r in pre) > 0, 'Exactly zero gradient: stop before step'
            total = original_clip(parameters,max_norm,*args,**kwargs)
            post = gradient_rows(model,kind,'after_clip',net)
            all_gradients.extend(pre+post)
            norm_after = math.sqrt(sum(r['gradient_l2']**2 for r in post))
            all_clips.append(dict(model=model,network=kind,norm_before=float(total),norm_after=norm_after,
                                  max_norm=float(max_norm),clip_scale=norm_after/float(total) if float(total) else None))
            return total

        def wrap_step(opt,kind):
            original_step = opt.step

            def measured_step(*args,**kwargs):
                guard()
                ledger = json.loads((OUT/'optimizer_steps.json').read_text())
                assert len(ledger['entries']) < 4
                assert not any(r['model']==model and r['network']==kind for r in ledger['entries'])
                entry = dict(model=model,network=kind,status='entered',entered_at_utc=datetime.now(timezone.utc).isoformat())
                ledger['entries'].append(entry)
                save('optimizer_steps.json',ledger)
                result = original_step(*args,**kwargs)
                entry.update(status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat())
                save('optimizer_steps.json',ledger)
                return result
            opt.step = measured_step

        agent.memory.get_batch_generator = only_first_generator
        wrap_step(actor_opt,'actor')
        wrap_step(critic_opt,'critic')
        restore_rng(original_rng)
        torch.nn.utils.clip_grad_norm_ = measured_clip
        try:
            agent.learn(progress_callback=stop_callback,microbatch_size=16)
            raise AssertionError('Full learn() must never return normally')
        except StopAfterOneMinibatch:
            pass
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        assert learned['progress']['minibatches_completed'] == 1
        assert agent.policy_version == cp['policy_version'] and agent.update_counter == cp['update_counter']
        # Save the state before additional analysis. Its schema explicitly
        # prevents confusing a diagnostic minibatch with a completed update.
        target = OUT/'updated'/model/'single-minibatch.pt'
        target.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(schema_version='diagnostic-single-minibatch-v1',source_checkpoint=spec,
                        config=cfg,actor=actor.state_dict(),critic=critic.state_dict(),
                        actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
                        diagnostic_steps=dict(actor=1,critic=1),full_ppo_update_completed=False),target)
        post_batch = evaluate_batch(actor,batch)
        post_monitor = evaluate_monitors(actor,observations)
        post_loss = loss_probe(post_batch,batch,advantages,cfg['entropy_weight'],cfg['policy_clip'])
        responses = compare(model,'update_batch',metadata,before_batch,repeat_batch,post_batch)
        monitor_meta = {key:dict(instance=instance) for key,instance,row in observations}
        responses += compare(model,'monitor_164',monitor_meta,before_monitor,repeat_monitor,post_monitor)
        all_responses.extend(responses)
        for kind,opt,net in [('actor',actor_opt,actor),('critic',critic_opt,critic)]:
            after_state = state_summary(opt,net)
            per_module = defaultdict(list)
            for name,p in net.named_parameters():
                delta = p.detach() - before_parameters[kind][name]
                first,last = opt_before[kind][name],after_state[name]
                assert last['step']-first['step'] == 1, (model,kind,name)
                assert torch.isfinite(p).all()
                d = dict(model=model,network=kind,parameter=name,module=name.split('.')[0],numel=p.numel(),
                         parameter_l2_before=norm(before_parameters[kind][name]),parameter_delta_l2=norm(delta),
                         parameter_delta_max_abs=float(delta.abs().max()),changed_elements=int((delta!=0).sum()),
                         adam_step_before=first['step'],adam_step_after=last['step'],
                         exp_avg_l2_before=first['exp_avg_l2'],exp_avg_l2_after=last['exp_avg_l2'],
                         exp_avg_sq_l2_before=first['exp_avg_sq_l2'],exp_avg_sq_l2_after=last['exp_avg_sq_l2'])
                all_parameters.append(d)
                per_module[d['module']].append(d)
            for module,rs in per_module.items():
                all_deltas.append(dict(model=model,network=kind,module=module,
                                       parameter_l2_before=math.sqrt(sum(r['parameter_l2_before']**2 for r in rs)),
                                       parameter_delta_l2=math.sqrt(sum(r['parameter_delta_l2']**2 for r in rs)),
                                       max_abs_delta=max(r['parameter_delta_max_abs'] for r in rs),
                                       changed_elements=sum(r['changed_elements'] for r in rs),
                                       total_elements=sum(r['numel'] for r in rs)))
        assert sha(checkpoint) == spec['sha256']
        summary = dict(model=model, source_checkpoint=spec, source_policy_version=cp['policy_version'],
                       episodes_imported=4, full_buffer_samples=len(rows), selected_samples=256,
                       selected_L1=sum(len(rows[i]['cands_state'])==1 for i in batch_indices),
                       selected_candidates=quant(len(rows[i]['cands_state']) for i in batch_indices),
                       full_buffer_raw_advantages=quant(raw.numpy()), raw_advantage_std=float(raw_std),
                       full_buffer_normalized_advantages=quant(advantages.numpy()),
                       selected_normalized_advantages=quant(advantages[batch_indices].numpy()),
                       selected_positive_advantages=int((advantages[batch_indices]>0).sum()),
                       preupdate_ratio_max_error=pre_ratio_error, optimizer_coverage=coverage,
                       original_learn=learned, pre_loss=pre_loss, post_loss=post_loss,
                       cuda_rng_present_in_source='cuda' in cp['rng'],cuda_rng_restored=False,
                       diagnostics_saved=str(target.relative_to(OUT)),diagnostic_checkpoint_sha256=sha(target),
                       full_ppo_update_completed=False,actor_steps=1,critic_steps=1,
                       elapsed_seconds=time.perf_counter()-START)
        summaries.append(summary)
        save('seed_summaries.json',summaries)
        for name,data in [('advantages.csv',all_advantages),('batch_manifest.csv',all_batch),('policy_responses.csv',all_responses),
                          ('module_gradients.csv',all_gradients),('gradient_clipping.csv',all_clips),
                          ('parameter_adam_changes.csv',all_parameters),('module_parameter_changes.csv',all_deltas),
                          ('trajectory_imports.csv',imports)]:
            write(name,data)
        print('completed',model,'Actor=1 Critic=1','elapsed',round(time.perf_counter()-START,3),flush=True)
        del cp,agent,actor,critic,actor_opt,critic_opt,before_parameters,before_batch,repeat_batch,post_batch,before_monitor,repeat_monitor,post_monitor

    assert all(sha(ROOT/p)==h for p,h in sources.items())
    assert all(sha(ROOT/p)==h for p,h in input_hashes.items())
    assert sha(OUT/'frozen_protocol.json')==protocol_hash
    assert 'pyscipopt' not in sys.modules and not torch.cuda.is_initialized()
    entries = json.loads((OUT/'optimizer_steps.json').read_text())['entries']
    assert len(entries)==4 and all(r['status']=='completed' for r in entries)
    save('runtime_sources_after.json',{p:sha(ROOT/p) for p in sources})
    save('validation.json',dict(status='passed',scope_complete=True,actor_steps=2,critic_steps=2,total_optimizer_steps=4,
                                new_solver_tasks=0,cuda_initialized=False,pyscipopt_imported=False,
                                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                                cpu_affinity=sorted(os.sched_getaffinity(0)),
                                selected_transitions=512,monitor_states=164,policy_response_rows=len(all_responses),
                                original_sources_unchanged=True,original_checkpoints_unchanged=True,
                                original_trajectories_unchanged=True,full_ppo_updates_completed=0,
                                elapsed_seconds=time.perf_counter()-START))
    print(json.dumps(json.loads((OUT/'validation.json').read_text()),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json',dict(error=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-START,
                                 optimizer_steps=json.loads((OUT/'optimizer_steps.json').read_text()),
                                 instruction='Do not rerun entered optimizer steps; inspect or finish analysis only.'))
        raise
