"""CPU inference on existing expert holdout; never trains or constructs a solver."""
import hashlib
import json
from pathlib import Path
import time

import torch
from expert_data import load_arrays
from pretrain import evaluate
from project.policy.light_actor import make_actor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'tgppo/runs/light-actor-20261105'
VARIANTS = ('b0', 'b0_bc', 'mlp', 'mlp_bc', 'set', 'set_bc', 'isab', 'isab_bc')


def digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write(value):
    path = HERE / 'retention.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    torch.set_num_threads(1)
    start = time.monotonic()
    data = load_arrays()
    result = dict(label='Existing train-internal expert holdout; not solver performance',
                  checkpoint_episode=100, split='validation', rows=[], complete=False,
                  cuda_initialized=False)
    for variant in VARIANTS:
        for seed in (11, 22):
            out = RUN / variant / 'training' / ('B0-s' + str(seed))
            checkpoint = out / 'episode-0100.pt'
            payload = torch.load(checkpoint, map_location='cpu')
            c = payload['config']
            assert payload['progress']['next_episode'] == 100
            assert payload['progress']['phase'] == 'ready'
            arch = dict(c['state_dims'], **{k: c[k] for k in ('hidden_dim', 'num_layers', 'num_heads', 'dropout')})
            torch.manual_seed(seed if c['actor_kind'] == 'b0' else 5000 + seed)
            actor = make_actor(arch, c['actor_kind'], c['actor_width'], c['inducing_points']).eval()
            if variant.endswith('_bc'):
                actor.load_state_dict(torch.load(out / 'pretrain/actor.pt', map_location='cpu')['actor'])
            initial = {k: v.clone() for k, v in actor.state_dict().items()}
            initialization = json.loads((out / 'initialization.json').read_text())
            assert digest(actor) == initialization['actor_fingerprint'], (variant, seed, 'initial weights')
            actor.load_state_dict(payload['actor'])
            squared_delta = sum(float((v - initial[k]).double().square().sum()) for k, v in actor.state_dict().items())
            squared_initial = sum(float(v.double().square().sum()) for v in initial.values())
            row = dict(variant=variant, seed=seed, actor_parameters=sum(v.numel() for v in actor.parameters()),
                       weight_delta_l2=squared_delta ** .5,
                       relative_weight_delta_l2=(squared_delta / squared_initial) ** .5,
                       validation=evaluate(actor, data, 'validation', 'cpu'))
            result['rows'].append(row)
            result['seconds'] = time.monotonic() - start
            assert not torch.cuda.is_initialized()
            write(result)
            print(variant, seed, row['validation']['macro'], flush=True)
    result.update(complete=True, seconds=time.monotonic() - start)
    write(result)


if __name__ == '__main__':
    main()
