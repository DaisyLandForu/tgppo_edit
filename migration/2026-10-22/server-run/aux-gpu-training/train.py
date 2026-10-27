#!/usr/bin/env python3
"""Independent, time-bounded synthetic GPU training; never imports TGPPO/SCIP."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
import uuid

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(model):
    value = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value.update(name.encode())
        value.update(parameter.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def train(args):
    if not (math.isfinite(args.minutes) and 0 < args.minutes <= 240):
        raise ValueError('--minutes must be positive and at most 240')
    if min(args.batch_size, args.width, args.samples) < 1 or args.steps < 0:
        raise ValueError('Positive model/data sizes and nonnegative step cap required')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA device allocated. Request one GPU; CPU fallback is disabled.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    parent = args.output_root.resolve()
    # Every invocation gets an exclusive new directory, including identical seeds.
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = parent/(stamp+'-'+socket.gethostname()+'-'+uuid.uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), 'unknown')
    config = dict(schema='aux-synthetic-training-v1', formal_experiment=False,
                  task='fit a synthetic teacher MLP; no MILP instances, TGPPO models or formal metrics',
                  device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  cpu_model=cpu, host=socket.gethostname(), seed=args.seed,
                  minutes=args.minutes, step_cap=args.steps, batch_size=args.batch_size,
                  width=args.width, samples=args.samples, torch_version=torch.__version__,
                  output_directory=str(output))
    write_json(output/'config.json', config)
    print(json.dumps(dict(event='AUX_TRAINING_STARTED', **config)), flush=True)

    model = nn.Sequential(nn.Linear(128, args.width), nn.ReLU(),
                          nn.Linear(args.width, args.width), nn.ReLU(),
                          nn.Linear(args.width, 64)).to(device)
    teacher = nn.Sequential(nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 64)).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        inputs = torch.randn(args.samples, 128, device=device)
        targets = teacher(inputs)
    del teacher
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = digest(model)
    stop_signal = []
    handlers = {}

    def request_stop(signum, frame):
        stop_signal.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)

    started = time.monotonic()
    deadline = started+args.minutes*60
    step = 0
    last_log = started-30
    last_save = started
    loss_number = None
    first_loss = None
    reason = 'time_limit'

    def save_checkpoint(name):
        payload = dict(schema='aux-synthetic-training-v1', config=config, step=step,
                       elapsed_seconds=time.monotonic()-started, loss=loss_number,
                       model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                       optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        target = output/name
        temp = target.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(target)

    try:
        with (output/'metrics.jsonl').open('a', buffering=1) as metrics:
            while time.monotonic() < deadline:
                if stop_signal:
                    reason = 'signal_{}'.format(stop_signal[-1])
                    break
                if args.steps and step >= args.steps:
                    reason = 'step_limit'
                    break
                indexes = torch.randint(args.samples, (args.batch_size,), device=device)
                prediction = model(inputs[indexes])
                loss = torch.mean((prediction-targets[indexes])**2)
                loss_number = float(loss.detach().item())
                if not math.isfinite(loss_number):
                    raise RuntimeError('Nonfinite auxiliary training loss')
                if first_loss is None:
                    first_loss = loss_number
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                current = time.monotonic()
                if current-last_log >= 30 or (args.steps and step == args.steps):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    row = dict(event='AUX_TRAINING_PROGRESS', step=step, loss=loss_number,
                               elapsed_seconds=time.monotonic()-started,
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
                    metrics.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                    last_log = current
                if current-last_save >= 120:
                    save_checkpoint('latest.pt')
                    last_save = current
            if device.type == 'cuda':
                torch.cuda.synchronize()
            save_checkpoint('latest.pt')
            save_checkpoint('final.pt')
        summary = dict(event='AUX_TRAINING_FINISHED', formal_experiment=False,
                       stop_reason=reason, steps=step, first_loss=first_loss, last_loss=loss_number,
                       elapsed_seconds=time.monotonic()-started, initial_model_sha256=before,
                       final_model_sha256=digest(model), output_directory=str(output),
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
        summary['parameters_changed'] = summary['initial_model_sha256'] != summary['final_model_sha256']
        write_json(output/'summary.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
    except Exception as exc:
        write_json(output/'failure.json', dict(error=repr(exc), step=step,
                   elapsed_seconds=time.monotonic()-started, formal_experiment=False))
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                        help='CPU is only for local engineering checks; Custom jobs use CUDA by default')
    parser.add_argument('--steps', type=int, default=0, help='Optional step cap, 0 means time cap only')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT/'tgppo/runs/server6-20261022/aux-gpu-training')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
