#!/usr/bin/env python3
"""Run engineering checks with resource accounting and a spawn-safe main file."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep previous verification attempts')
    root=args.root.resolve()
    sys.path[:0]=[str(root/'tgppo/experiments/stage4/metrics_binding/src'),
                  str(root/'tgppo/experiments/server-20261022/code')]
    start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(root/'tgppo/experiments/server-20261022/tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    torch=sys.modules.get('torch')
    report=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='engineering checks on this node; not formal training or target throughput evidence',
        host=socket.gethostname(),cuda_available=torch.cuda.is_available() if torch else None,
        tests_run=result.testsRun,skipped=len(result.skipped),errors=len(result.errors),
        failures=len(result.failures),passed=result.wasSuccessful(),wall_seconds=time.monotonic()-start,
        self_cpu_seconds=own.ru_utime+own.ru_stime,children_cpu_seconds=children.ru_utime+children.ru_stime,
        self_peak_rss_kib=own.ru_maxrss,children_max_process_rss_kib=children.ru_maxrss,
        skip_reasons=[str(reason) for _,reason in result.skipped])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
