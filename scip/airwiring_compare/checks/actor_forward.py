"""Compare the frozen Actor across the original and deployment Torch versions."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/tgppo/experiments/server-20261022/code'))
import torch
from project.policy.actor import Actor

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
payload = torch.load(ROOT / 'research/tgppo/runs/server6-20261022/training/B0-s11/actor_best.pt', map_location='cpu')
config = payload['config']
actor = Actor(**config['state_dims'], **{key: config[key] for key in ('hidden_dim', 'num_layers', 'num_heads', 'dropout')})
actor.load_state_dict(payload['actor'])
actor.eval()
cases = []
with torch.no_grad():
    for n in (1, 7, 31):
        cands = torch.sin(torch.arange(n*25, dtype=torch.float32).reshape(1,n,25)*.1)
        node = torch.cos(torch.arange(8, dtype=torch.float32).reshape(1,8)*.1)
        mip = torch.sin(torch.arange(53, dtype=torch.float32).reshape(1,53)*.1)
        probs = actor(cands,node,mip)
        cases.append(dict(candidates=n, probabilities=probs.squeeze(0).tolist(), action=int(probs.argmax())))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(dict(torch_version=torch.__version__, cases=cases),indent=2)+'\n')
print(torch.__version__, str(args.output))
