#!/usr/bin/env python3
"""Read-only preparation status; training still enforces target provenance itself."""
import argparse
import json
from pathlib import Path
import sys


ARTIFACTS=(('native_baselines','normalization/baselines.json'),
           ('gpu_bridge','gpu-bridge/verification.json'),
           ('throughput','throughput/verification.json'))


def inspect(run):
    states={};loaded={};errors=[]
    for name,relative in ARTIFACTS:
        path=run/relative
        if not path.exists():
            states[name]='pending';continue
        try:
            payload=json.loads(path.read_text())
            if not isinstance(payload,dict):raise ValueError('expected an object')
        except (OSError,ValueError) as exc:
            states[name]='invalid';errors.append(relative+': '+str(exc));continue
        loaded[name]=payload
        if name=='native_baselines':
            rows=payload.get('baselines',{})
            valid=(isinstance(rows,dict) and len(rows)==75 and
                   all(isinstance(row,dict) and isinstance(row.get('result'),dict) and
                       row['result'].get('valid') is True for row in rows.values()))
        else:valid=payload.get('passed') is True
        states[name]='passed' if valid else 'failed'
        if not valid:errors.append(relative+': preparation did not pass')
    if len(loaded)==len(ARTIFACTS) and all(state=='passed' for state in states.values()):
        base=loaded['native_baselines'];bridge=loaded['gpu_bridge'];pilot=loaded['throughput']
        protocol=base.get('protocol_sha256');stack=base.get('stack')
        if not protocol or not stack or any(p.get('parent_protocol_sha256')!=protocol or p.get('stack')!=stack for p in (bridge,pilot)):
            errors.append('Preparation artifacts use different protocol/CPU/software fingerprints')
        if not bridge.get('gpu') or bridge.get('gpu')!=pilot.get('gpu'):
            errors.append('Bridge and throughput GPU signatures differ')
    pending=[relative for name,relative in ARTIFACTS if states[name]=='pending']
    return dict(ready=not pending and not errors,stages=states,pending_files=pending,errors=errors,
                scope='Preparation files only; training additionally checks the current protocol, CPU/software and V100')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parents[3]/'tgppo/runs/server6-20261022')
    args=parser.parse_args();result=inspect(args.run_root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if result['ready']:return 0
    if result['errors']:
        print('Preparation failed or artifacts disagree. Inspect the existing prepare log; do not delete results.',file=sys.stderr)
        return 1
    print('Preparation is still running/incomplete. Keep the existing prepare job; submit training only after all three stages pass. No training episode has started.',file=sys.stderr)
    return 75


if __name__=='__main__':sys.exit(main())
