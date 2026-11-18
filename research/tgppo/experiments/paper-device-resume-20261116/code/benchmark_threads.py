"""Read-only CPU forward/backward microbenchmark using actual pending states."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from project import runtime as rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', choices=[1,2,4,8], default=[1,2,4])
    args = parser.parse_args()
    from server import resource_limits
    limits = resource_limits()
    if max(args.threads) > limits['effective_cpu_count']:
        parser.error('Requested threads exceed actual CPU quota: '+str(limits))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    agent, payload = rt.load_agent(args.checkpoint, device='cpu', restore_random_state=False)
    rows = agent.memory.rows()
    ordered = sorted(range(len(rows)), key=lambda i:len(rows[i]['cands_state']))
    selected = [ordered[int((len(ordered)-1)*q)] for q in (.1, .5, .9)]
    # Actual complete candidate sets, four copies per batch (production chunks
    # may use 1..8); no optimizer step, checkpoint writes, SCIP or CUDA calls.
    agent.actor_network.eval()
    agent.critic_network.train()
    measurements = []
    begin = time.monotonic()
    for index in selected:
        row = rows[index]
        size = len(row['cands_state'])
        copies = min(4, max(1, agent.attention_cell_budget//(size*size)))
        cands = torch.tensor(row['cands_state'])[None].repeat(copies,1,1)
        node = torch.tensor(row['node_state'])[None].repeat(copies,1)
        mip = torch.tensor(row['mip_state'])[None].repeat(copies,1)
        mask = torch.zeros((copies,size), dtype=torch.bool)
        for threads in args.threads:
            torch.set_num_threads(threads)
            trials = []
            for trial in range(4):
                agent.actor_network.zero_grad(set_to_none=True)
                agent.critic_network.zero_grad(set_to_none=True)
                torch.manual_seed(901)
                start = time.monotonic()
                p = agent.actor_network(cands,node,mip,mask)
                (-p[:,row['action']].clamp_min(1e-30).log().mean()).backward()
                value = agent.critic_network(cands,node,mip,mask).reshape(-1)
                (value.square().mean()).backward()
                elapsed = time.monotonic()-start
                if trial: trials.append(elapsed)
            measurements.append(dict(candidates=size,batch=copies,threads=threads,
                median_seconds=float(np.median(trials)),trials=trials))
    torch.set_num_threads(1)
    result = dict(checkpoint=str(args.checkpoint), type='CPU Actor+Critic forward/backward only',
        optimizer_steps=0, solver_runs=0, gpu_used=False, full_episode_benchmark=False,
        cpu_quota=limits['effective_cpu_count'], torch=torch.__version__, elapsed_seconds=time.monotonic()-begin,
        candidate_percentiles=np.percentile([len(r['cands_state']) for r in rows],[0,10,50,90,100]).tolist(),
        measurements=measurements)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
