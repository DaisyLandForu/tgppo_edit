"""One frozen, single-CPU localization experiment. No optimizer or solver."""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from collections import defaultdict

START = time.perf_counter()
import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
PRIOR = OUT.parent / 'tgppo_20261102_local_validation'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(20261102)
sys.path.insert(0, str(OUT))
from model_copy.actor import Actor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write(name, rows):
    if rows:
        with (OUT / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def q(values):
    v = [float(x) for x in values if x is not None]
    return dict(zip(['min', 'p50', 'p95', 'max'], map(float, np.quantile(v, [0, .5, .95, 1])))) if v else {}


def inputs(row, dtype=torch.float32, grad=False):
    x = torch.as_tensor(row['cands_state'], dtype=dtype)[None].clone()
    x.requires_grad_(grad)
    return (x, torch.as_tensor(row['node_state'], dtype=dtype)[None],
            torch.as_tensor(row['mip_state'], dtype=dtype)[None])


def matrix(t, name):
    if name.startswith('tf') or name == 'transformer':
        t = t.transpose(0, 1)
    if t.dim() == 2:  # logits B,L
        t = t.unsqueeze(-1)
    assert t.dim() == 3 and t.shape[0] == 1, (name, t.shape)
    return t[0].double()


def measures(t, name, rank=False):
    a = matrix(t, name)
    assert torch.isfinite(a).all(), name
    c = a - a.mean(0, keepdim=True)
    energy = float(a.square().mean())
    contrast = float(c.square().mean())
    d = dict(width=a.shape[1], total_rms=math.sqrt(energy), centered_rms=math.sqrt(contrast),
             relative_centered_rms=math.sqrt(contrast / energy) if energy else 0.,
             mean_candidate_range=float((a.max(0).values - a.min(0).values).abs().mean()),
             exactly_constant=bool((a == a[:1]).all()))
    if rank:
        small = c.T @ c if c.shape[1] <= c.shape[0] else c @ c.T
        trace = float(torch.trace(small))
        norm_sq = float(small.square().sum())
        d.update(effective_rank=trace * trace / norm_sq if norm_sq else 0.,
                 pc1_energy_fraction=float(torch.linalg.eigvalsh(small)[-1]) / trace if trace else None)
    return d


def centered(t, name):
    a = matrix(t, name)
    return a - a.mean(0, keepdim=True)


def probability(t):
    p = t.detach().double().numpy().reshape(-1)
    assert np.isfinite(p).all() and (p > 0).all()
    return p / p.sum()


def tv(p, q_):
    return float(.5 * np.abs(p - q_).sum())


class FixedStateVariance(nn.Module):
    """Keep candidate centering; freeze just the denominator at this state."""
    def __init__(self, layer):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias
        self.eps = layer.eps
        self.register_buffer('reference_scale', torch.tensor(1.), persistent=False)

    def set_reference(self, pre_ln):
        self.reference_scale = torch.sqrt(pre_ln.var(-1, unbiased=False).mean() + self.eps).detach().clone()

    def forward(self, x):
        return (x - x.mean(-1, keepdim=True)) / self.reference_scale * self.weight + self.bias


RANK_NAMES = {'raw', 'var_embedding', 'fusion', 'transformer', 'bimatching',
              'head5.out', 'head6.linear', 'head6.norm', 'head6.out', 'logits'}
LIVE_NAMES = {'var_embedding', 'fusion', 'transformer', 'bimatching',
              'head5.out', 'head6.linear', 'head6.out', 'logits'}


def forward_capture(net, x, n, m, grad=False):
    seen = {'raw': x.detach().clone()}
    live = {}
    hooks = []

    def hook(name):
        def take(module, inp, value):
            # Clone before an in-place ReLU changes the LayerNorm output.
            seen[name] = value.detach().clone()
            if grad and name in LIVE_NAMES:
                value.retain_grad()
                live[name] = value
        return take

    modules = [('var_embedding', net.var_embedding), ('fusion', net.global_embedding),
               ('transformer', net.transformer), ('bimatching', net.tree_refinement),
               ('logits', net.output_layer)]
    modules += [(f'tf{i}', layer) for i, layer in enumerate(net.transformer.layers)]
    for i, layer in enumerate(net.output_layer.BranchingNet):
        modules.append((f'head{i}.linear', layer[0]))
        if len(layer) > 1:
            modules.append((f'head{i}.norm', layer[1]))
            modules.append((f'head{i}.out', layer))
    for name, module in modules:
        hooks.append(module.register_forward_hook(hook(name)))
    try:
        with torch.enable_grad() if grad else torch.no_grad():
            prob = net(x, n, m)[0]
    finally:
        for handle in hooks:
            handle.remove()
    return prob, seen, live


def bimatching_probe(net, seen, n, m):
    with torch.no_grad():
        tree = net.tree_embedding(torch.cat([n, m], -1)).unsqueeze(1)
        z = seen['transformer'].transpose(0, 1)
        mod = net.tree_refinement
        alpha = torch.softmax(torch.bmm(mod.linear1_1(tree), z.transpose(1, 2)), -1)
        beta = torch.softmax(torch.bmm(mod.linear1_2(z), tree.transpose(1, 2)).squeeze(-1), -1)
        stc = torch.relu(mod.linear2_1(torch.bmm(alpha, z)))
        # Candidate-specific path reconstructed from a single beta_i and the
        # common tree vector; no direct candidate embedding enters below.
        sct = torch.relu(mod.linear2_2(beta.unsqueeze(-1) * tree))
        gate = torch.sigmoid(mod.linear3_1(stc) + mod.linear3_2(sct))
        reconstructed = gate * stc + (1 - gate) * sct
        p = probability(beta)
        return dict(beta_tv_uniform=tv(p, np.full(len(p), 1 / len(p))),
                    beta_coefficient_variation=float(np.std(p) / np.mean(p)),
                    beta_max_min=float(np.ptp(p)),
                    reconstruction_max_error=float((reconstructed - seen['bimatching']).abs().max()))


def guard():
    assert time.perf_counter() - START < 455, 'Internal stop: reserve budget for evidence delivery'


def main():
    protocol_hash = sha(OUT / 'frozen_protocol.json')
    expected_sources = json.loads((PRIOR / 'runtime_sources_after.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in expected_sources.items())
    save('runtime_sources_before.json', expected_sources)
    for name in ['actor.py', 'modules.py']:
        assert sha(OUT / 'model_copy' / name) == sha(ROOT / 'tgppo/experiments/server-20261022/code/project/policy' / name)
    prior_trajectories = json.loads((PRIOR / 'input_trajectory_hashes.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in prior_trajectories.items())
    for name in ['observations.pkl', 'observation_manifest.csv', 'observation_coverage.csv']:
        shutil.copy2(PRIOR / name, OUT / name)
    observations = pickle.loads((OUT / 'observations.pkl').read_bytes())
    assert len(observations) == 164 and len({name for _, name, _ in observations}) == 17
    first = {}
    for i, (_, name, _) in enumerate(observations):
        first.setdefault(name, i)
    anchors = set(first.values())
    checkpoint_specs = json.loads((PRIOR / 'checkpoint_audit.json').read_text())
    frozen = dict(protocol_sha256=protocol_hash, prior_observations_sha256=sha(PRIOR / 'observations.pkl'),
                  manifest_sha256=sha(OUT / 'observation_manifest.csv'), anchor_indices=sorted(anchors),
                  checkpoint_specs=checkpoint_specs, source_trajectory_hashes=prior_trajectories)
    save('frozen_inputs.json', frozen)

    layers, ranks, policies, paired, beta_rows = [], [], [], [], []
    gradients, invariants, perturbations, precisions = [], [], [], []
    completed, summaries = [], []
    for spec in checkpoint_specs:
        guard()
        checkpoint = ROOT / spec['path']
        assert sha(checkpoint) == spec['sha256']
        payload = torch.load(checkpoint, map_location='cpu')
        cfg = payload['config']
        a = Actor(**cfg['state_dims'], **{k: cfg[k] for k in ['hidden_dim', 'num_heads', 'num_layers', 'dropout']})
        a.load_state_dict(payload['actor'])
        a.eval()
        del payload
        before = {k: v.detach().clone() for k, v in a.state_dict().items()}
        b = copy.deepcopy(a)
        ln = b.output_layer.BranchingNet[6][1]
        assert isinstance(ln, nn.LayerNorm) and tuple(ln.normalized_shape) == (2,)
        b.output_layer.BranchingNet[6][1] = FixedStateVariance(ln)
        b.eval()
        double_a = copy.deepcopy(a).double().eval()
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in before.items())
        assert not ({id(p) for p in a.parameters()} & {id(p) for p in b.parameters()})
        model, episode = spec['model'], spec['episode']
        for idx, (key, instance, row) in enumerate(observations):
            guard()
            x, n, m = inputs(row)
            L = x.shape[1]
            meta = dict(model=model, episode=episode, observation=key, instance=instance, L=L, anchor=idx in anchors)
            p32, original, _ = forward_capture(a, x, n, m)
            b.output_layer.BranchingNet[6][1].set_reference(original['head6.linear'])
            pmod, modified, _ = forward_capture(b, x, n, m)
            all_seen = {'original': original, 'ln2_fixed_state_variance': modified}
            probs = {'original': probability(p32), 'ln2_fixed_state_variance': probability(pmod)}
            for variant, seen in all_seen.items():
                for name, value in seen.items():
                    # Upstream tensors are unchanged; retaining both paths
                    # makes this identity auditable rather than assumed.
                    d = dict(meta, variant=variant, layer=name, **measures(value, name, rank=idx in anchors and name in RANK_NAMES))
                    if 'effective_rank' in d:
                        ranks.append(d.copy())
                    layers.append({k: v for k, v in d.items() if k not in ['effective_rank', 'pc1_energy_fraction']})
                p = probs[variant]
                logits = seen['logits'][0].double().numpy()
                policies.append(dict(meta, variant=variant, tv_uniform=tv(p, np.full(L, 1 / L)),
                                     kl_uniform=max(0., float(np.sum(p * np.log(p * L)))),
                                     logit_spread=float(np.ptp(logits)),
                                     ln2_relu_all_zero=bool((seen['head6.out'] == 0).all())))
            scale_original = measures(original['head6.norm'], 'head6.norm')['total_rms']
            scale_modified = measures(modified['head6.norm'], 'head6.norm')['total_rms']
            relu_original = measures(original['head6.out'], 'head6.out')['total_rms']
            relu_modified = measures(modified['head6.out'], 'head6.out')['total_rms']
            paired.append(dict(meta, tv_original_modified=tv(probs['original'], probs['ln2_fixed_state_variance']),
                               ln2_rms_ratio=scale_modified / scale_original if scale_original else None,
                               relu_rms_ratio=relu_modified / relu_original if relu_original else None,
                               reference_scale=float(b.output_layer.BranchingNet[6][1].reference_scale),
                               relu_pattern_change_fraction=float(((original['head6.out'] > 0) != (modified['head6.out'] > 0)).float().mean())))
            beta_rows.append(dict(meta, **bimatching_probe(a, original, n, m)))

            if idx in anchors:
                rng = np.random.RandomState(20261102 + idx)
                permutation = torch.tensor(rng.permutation(L))
                inverse = np.argsort(permutation.numpy())
                for variant, net in [('original', a), ('ln2_fixed_state_variance', b)]:
                    net.zero_grad(set_to_none=True)
                    xx, nn_, mm = inputs(row, grad=True)
                    pg, _, live = forward_capture(net, xx, nn_, mm, grad=True)
                    loss = -torch.distributions.Categorical(pg).log_prob(torch.tensor(row['action']))
                    loss.backward()
                    for name, module in [('var_embedding', net.var_embedding), ('fusion', net.global_embedding),
                                         ('transformer', net.transformer), ('bimatching', net.tree_refinement), ('treegate', net.output_layer)]:
                        norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in module.parameters() if p.grad is not None))
                        assert math.isfinite(norm)
                        gradients.append(dict(meta, variant=variant, kind='parameters', layer=name, l2_norm=norm,
                                              candidate_centered_l2=None))
                    for name, value in live.items():
                        assert value.grad is not None, name
                        gm = matrix(value.grad, name)
                        assert torch.isfinite(gm).all(), name
                        gradients.append(dict(meta, variant=variant, kind='activation', layer=name,
                                              l2_norm=float(gm.norm()),
                                              candidate_centered_l2=float((gm - gm.mean(0, keepdim=True)).norm())))
                    gradients.append(dict(meta, variant=variant, kind='input', layer='raw', l2_norm=float(xx.grad.norm()),
                                          candidate_centered_l2=float((xx.grad - xx.grad.mean(1, keepdim=True)).norm())))
                    net.zero_grad(set_to_none=True)
                    del pg, live, loss
                    with torch.no_grad():
                        repeat = net(x, n, m)[0].numpy()
                        permuted = net(x[:, permutation], n, m)[0].numpy()[inverse]
                        mask = torch.tensor([[False] * L + [True] * 7])
                        padded = net(torch.cat([x, torch.zeros((1, 7, x.shape[-1]))], 1), n, m, mask)[0].numpy()
                    baseline = p32.numpy() if variant == 'original' else pmod.numpy()
                    inv = dict(meta, variant=variant,
                               repeat_max_error=float(np.max(np.abs(repeat - baseline))),
                               permutation_max_error=float(np.max(np.abs(permuted - baseline))),
                               padding_max_error=float(np.max(np.abs(padded[:L] - baseline))),
                               padded_probability=float(padded[L:].sum()))
                    invariants.append(inv)
                    assert max(inv[k] for k in ['repeat_max_error', 'permutation_max_error', 'padding_max_error', 'padded_probability']) <= 1e-6
                    altered = x + .001 * (x - x.mean(1, keepdim=True))
                    pa, alt_seen, _ = forward_capture(net, altered, n, m)
                    for name in sorted(RANK_NAMES):
                        base_c = centered(all_seen[variant][name], name)
                        change = centered(alt_seen[name] - all_seen[variant][name], name)
                        perturbations.append(dict(meta, variant=variant, layer=name,
                                                  centered_output_change_rms=float(change.square().mean().sqrt()),
                                                  relative_centered_output_change=float(change.norm() / base_c.norm()) if float(base_c.norm()) else None,
                                                  policy_tv_change=tv(probability(pa), probs[variant])))
                xd, nd, md = inputs(row, dtype=torch.float64)
                pd, double_seen, _ = forward_capture(double_a, xd, nd, md)
                ppd = probability(pd)
                precisions.append(dict(meta, float32_tv_uniform=tv(probs['original'], np.full(L, 1 / L)),
                                       float64_tv_uniform=tv(ppd, np.full(L, 1 / L)),
                                       float32_float64_tv=tv(probs['original'], ppd),
                                       float64_logit_spread=float(torch.ptp(double_seen['logits'])) if hasattr(torch, 'ptp') else float(double_seen['logits'].max() - double_seen['logits'].min())))

        for net in [a, b]:
            assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items()), 'Parameters changed'
        assert sha(checkpoint) == spec['sha256']
        completed.append(dict(spec, parameter_values_unchanged=True, independent_parameter_objects=True))
        for name, rows in [('layer_metrics.csv', layers), ('effective_ranks.csv', ranks), ('policy_metrics.csv', policies),
                           ('paired_intervention.csv', paired), ('bimatching_probe.csv', beta_rows),
                           ('score_gradients.csv', gradients), ('invariance_checks.csv', invariants),
                           ('input_perturbation.csv', perturbations), ('precision_checks.csv', precisions)]:
            write(name, rows)
        print('completed', model, episode, 'elapsed', round(time.perf_counter() - START, 3), flush=True)
        del a, b, double_a, before

    assert sha(OUT / 'frozen_protocol.json') == protocol_hash
    assert all(sha(ROOT / p) == h for p, h in expected_sources.items())
    assert all(sha(ROOT / p) == h for p, h in prior_trajectories.items())
    assert 'pyscipopt' not in sys.modules and not torch.cuda.is_initialized()
    save('checkpoint_audit.json', completed)
    save('runtime_sources_after.json', {p: sha(ROOT / p) for p in expected_sources})
    validation = dict(status='passed', scope_complete=True, no_parameters_updated=True,
                      original_sources_unchanged=True, input_trajectories_unchanged=True,
                      copied_observations_unchanged=sha(OUT / 'observations.pkl') == frozen['prior_observations_sha256'],
                      protocol_unchanged=True, optimizer_steps=0, new_solver_tasks=0,
                      cuda_initialized=torch.cuda.is_initialized(), pyscipopt_imported='pyscipopt' in sys.modules,
                      threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      observations=len(observations), instances=len(first), checkpoints=len(completed),
                      policy_rows=len(policies), layer_rows=len(layers), gradient_rows=len(gradients),
                      anchor_cases=len(precisions), invariant_cases=len(invariants),
                      elapsed_seconds=time.perf_counter() - START)
    save('validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
