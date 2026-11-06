#!/usr/bin/env python3
"""Validate frozen CPU Mamba against upstream reference and tiny SCIP branches.

Never optimizes the aviation model. Run using custom_job.sh check-mamba-cpu.
"""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import compare as c
sys.path.insert(0, str(c.MAMBA / 'code'))

import torch
import mamba_ssm.modules.mamba_simple as upstream
import mamba_ssm.ops.selective_scan_interface as upstream_scan
from network import GroupedMamba, TreeGateNet
from mamba_cpu import install_cpu_backend, selective_scan_cpu


def forbidden(*_args, **_kwargs):
    raise AssertionError('CPU check attempted CUDA initialization or a CUDA Mamba kernel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'checks/mamba-cpu')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    assert not torch.cuda.is_initialized()
    reference_cases, scan_cases = [], []
    selected = json.loads((c.MAMBA_RUN / 'training/selected.json').read_text())
    checkpoint_path = c.MAMBA_RUN / selected['checkpoint']
    assert c.sha(checkpoint_path) == selected['checkpoint_sha256'] == c.MAMBA_SHA
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert checkpoint['config_sha256'] == selected['config_sha256']
    assert checkpoint['dataset_sha256'] == selected['dataset_sha256']
    embedding, native = TreeGateNet(infimum=8), GroupedMamba(8, 100)
    embedding.load_state_dict(checkpoint['embedding'])
    native.load_state_dict(checkpoint['sequence'])
    native.eval()
    embedding.eval()
    cpu = install_cpu_backend(copy.deepcopy(native)).eval()
    for key, value in native.state_dict().items():
        assert torch.equal(value, cpu.state_dict()[key]), key
    native.mamba.use_fast_path = False
    # Execute the installed upstream Mamba.forward unchanged, selecting its
    # documented PyTorch conv path and independent official scan reference.
    # Patches exist only within this diagnostic process, never in production.
    with patch.object(torch.cuda, '_lazy_init', forbidden), \
            patch.object(upstream, 'mamba_inner_fn', forbidden), \
            patch.object(upstream, 'causal_conv1d_fn', None), \
            patch.object(upstream, 'selective_scan_fn', upstream_scan.selective_scan_ref), \
            patch.object(upstream_scan.selective_scan_cuda, 'fwd', forbidden), \
            torch.inference_mode():
        for batch, length in [(1, 1), (2, 3), (1, 17), (2, 257)]:
            u, delta, z = [torch.randn(batch, 16, length) for _ in range(3)]
            A = -torch.exp(torch.randn(16, 64))
            B, C = [torch.randn(batch, 64, length) for _ in range(2)]
            D, bias = torch.randn(16), torch.randn(16)
            expected = upstream_scan.selective_scan_ref(u, delta, A, B, C, D, z,
                                                         delta_bias=bias, delta_softplus=True)
            actual = selective_scan_cpu(u, delta, A, B, C, D, z, bias)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            scan_cases.append(dict(batch=batch, tokens=length,
                                   max_absolute_error=(actual-expected).abs().max().item()))
        for batch, counts in [(1, [1]), (2, [3]), (1, [31, 1, 7]),
                              (1, [1, 1, 257]), (1, [3, 1, 31, 1] * 12 + [257])]:
            groups = []
            for count in counts:
                candidates = torch.randn(batch, count, 25)
                state = torch.randn(batch, 61)
                embeddings, _ = embedding(candidates, state)
                groups.append(embeddings)
            inputs = torch.cat(groups, dim=1)
            positions = torch.cat([torch.full((batch, count), i//2, dtype=torch.long)
                                   for i, count in enumerate(counts)], dim=1)
            expected = native(inputs, positions)
            before = time.monotonic()
            actual = cpu(inputs, positions)
            elapsed = time.monotonic() - before
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            old_logits = expected[:, -counts[-1]:].mean(-1)
            new_logits = actual[:, -counts[-1]:].mean(-1)
            assert torch.equal(old_logits.argmax(-1), new_logits.argmax(-1))
            reference_cases.append(dict(batch=batch, history_groups=len(counts), tokens=sum(counts),
                                         current_candidates=counts[-1], cpu_seconds=elapsed,
                                         max_absolute_error=(actual-expected).abs().max().item(),
                                         actions_identical=True))
        # Stress the full 24-history-state bound using all 5736 original binaries
        # as candidates per state. This is synthetic, not a recorded aviation LP.
        counts = [5736, 1] * 24 + [5736]
        inputs = torch.randn(1, sum(counts), 8)
        positions = torch.cat([torch.full((1, n), i//2, dtype=torch.long)
                               for i, n in enumerate(counts)], dim=1)
        before = time.monotonic()
        output = cpu(inputs, positions)
        long_seconds = time.monotonic() - before
        assert output.shape == inputs.shape and torch.isfinite(output).all()
        assert torch.equal(cpu(inputs[:, :17], positions[:, :17]),
                           cpu(inputs[:, :17], positions[:, :17]))
        torch.testing.assert_close(output[:, :17], cpu(inputs[:, :17], positions[:, :17]),
                                   rtol=1e-5, atol=1e-5)
        print('Reference checks passed; long synthetic history:', sum(counts),
              'tokens in', round(long_seconds, 3), 'seconds', flush=True)
        smoke_args = argparse.Namespace(results=args.output / 'smoke', method='mamba',
                                         device='cpu', seed=0)
        c.smoke(smoke_args)
        smoke = json.loads((smoke_args.attempt / 'result.json').read_text())
        assert smoke['branches'] >= 25 and not smoke.get('policy_error')
        assert smoke['policy']['device'] == 'cpu'
        assert smoke['policy']['gpu_model'] is None
        assert smoke['policy']['inference_backend'] == 'pytorch-cpu-mamba1'
    assert not torch.cuda.is_initialized()
    for key, value in native.state_dict().items():
        assert torch.equal(value, cpu.state_dict()[key]), key
    protected_path = HERE / 'checks/mamba-cpu/protected-inputs.json'
    protected_count = None
    if protected_path.exists():
        protected = json.loads(protected_path.read_text())
        for name, checksum in protected.items():
            assert c.sha(c.ROOT / name) == checksum, name
        protected_count = len(protected)
    report = dict(status='passed', checked_utc=c.stamp(), elapsed_seconds=time.monotonic()-started,
                  checkpoint_sha256=c.MAMBA_SHA, backend_sha256=c.sha(HERE/'mamba_cpu.py'),
                  driver_sha256=c.sha(HERE/'compare.py'),
                  torch_version=torch.__version__, hardware=c.hardware(),
                  upstream_source_sha256={str(p): c.sha(p) for p in
                      (Path(upstream.__file__), Path(upstream_scan.__file__))},
                  cuda_initialized=False, cuda_initialization_and_scan_forbidden_during_checks=True,
                  scan_reference_cases=scan_cases, frozen_network_reference_cases=reference_cases,
                  long_synthetic_history=dict(groups=49, tokens=sum(counts), current_candidates=5736,
                                               cpu_seconds=long_seconds, finite=True, causal_prefix_check=True),
                  process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  smoke_result=str(smoke_args.attempt/'result.json'), smoke_branches=smoke['branches'],
                  protected_inputs_unchanged=protected_count,
                  original_weights_unchanged=True, aviation_optimization_performed=False,
                  cuda_numerical_parity='not checked: no GPU; CPU matches upstream CPU reference, '
                                        'near-tie actions can differ across devices')
    c.write_json(args.output/'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
