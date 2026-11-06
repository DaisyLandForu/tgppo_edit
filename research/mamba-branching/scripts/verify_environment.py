#!/usr/bin/env python3
"""Validate installed packages and Mamba imports without executing CUDA kernels."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'status': 'failed', 'timestamp_utc': stamp, 'host': platform.node(),
              'python': sys.executable, 'scope': 'CPU imports and constructors; CUDA execution not tested',
              'training_started': False}
    try:
        expected = Path(os.environ.get('MAMBA_ENV', '/home/duweiyue25/conda/envs/mamba-branching')).resolve()
        assert Path(sys.prefix).resolve() == expected
        assert sys.version_info[:2] == (3, 10)
        import numpy as np
        import torch
        import h5py
        import pyscipopt
        import causal_conv1d_cuda
        import selective_scan_cuda
        from torch.utils.tensorboard import SummaryWriter
        from mamba_ssm import Mamba
        sys.path.insert(0, str(ROOT / 'code'))
        import main as upstream_training
        import dataset_scip
        import env
        from network import TreeGateNet, GroupedMamba
        versions = {name: importlib.metadata.version(name) for name in
                    ['numpy', 'torch', 'pyscipopt', 'mamba-ssm', 'causal-conv1d', 'triton', 'transformers', 'h5py']}
        for name, version in {'numpy': '1.26.4', 'torch': '2.2.2+cu118', 'pyscipopt': '4.3.0',
                              'mamba-ssm': '2.2.2', 'causal-conv1d': '1.4.0', 'triton': '2.2.0'}.items():
            assert versions[name] == version, (name, versions[name])
        assert torch.version.cuda == '11.8'
        assert not torch._C._GLIBCXX_USE_CXX11_ABI
        for module in [np, torch, pyscipopt, causal_conv1d_cuda, selective_scan_cuda]:
            assert expected in Path(module.__file__).resolve().parents, module.__file__
        torch.manual_seed(0)
        embed = TreeGateNet(infimum=8)
        sequence = GroupedMamba(d_model=8, max_seq_length=100)
        assert isinstance(sequence.mamba, Mamba)
        assert sequence.position_embed.num_embeddings == 100
        with torch.no_grad():
            features, logits = embed(torch.randn(1, 5, 25), torch.randn(1, 61))
        assert features.shape == (1, 5, 8) and logits.shape == (1, 5)
        assert torch.isfinite(features).all()
        report.update(status='passed', versions=versions, cuda_available=torch.cuda.is_available(),
                      cuda_runtime=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                      embedding_parameters=sum(p.numel() for p in embed.parameters()),
                      sequence_parameters=sum(p.numel() for p in sequence.parameters()),
                      extension_paths={'causal_conv1d_cuda': causal_conv1d_cuda.__file__,
                                       'selective_scan_cuda': selective_scan_cuda.__file__})
    except Exception:
        report['error'] = traceback.format_exc()
    directory = ROOT / 'runs/environment'
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / (stamp + '.json')
    file.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(file), 'error': report.get('error')}, indent=2))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
