"""Memory resource override around the frozen recovery entry, not model changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parents[1]
BASE = PACKAGE.parent/'paper-recovery-20261109'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    release = json.loads((PACKAGE/'release.json').read_text())
    if not release.get('passed'):
        raise ValueError('Resource wrapper checks have not passed')
    for relative,digest in release['files'].items():
        if sha(PACKAGE/relative) != digest:
            raise ValueError('Changed resource wrapper: '+relative)
    if sha(BASE/'release.json') != release['recovery_release_sha256']:
        raise ValueError('Recovery release changed')


def memory_preflight(original, requested_gib):
    def preflight(root,c,device):
        # Only this resource check receives an override. The original c and its
        # training/checkpoint fingerprint, worker limits and hyperparameters stay.
        checked = dict(c,ram_gib_per_training_job=requested_gib)
        info = original(root,checked,device)
        actual = info.get('memory_limit_bytes')
        if actual is not None and actual < requested_gib*1024**3:
            raise RuntimeError('Requested resource minimum {}GiB, actual {:.2f}GiB'.format(
                requested_gib,actual/1024**3))
        info['resource_override'] = dict(original_minimum_ram_gib=c['ram_gib_per_training_job'],
            requested_minimum_ram_gib=requested_gib,actual_container_limit_bytes=actual,
            worker_address_space_limit_bytes=c['worker_address_space_bytes'],
            algorithm_config_unchanged=True)
        return info
    return preflight


def memory_sample(root=Path('/sys/fs/cgroup')):
    def number(paths):
        for relative in paths:
            try:
                value = (root/relative).read_text().strip()
                return int(value) if value != 'max' else None
            except (OSError,ValueError):
                continue
        return None
    events = {}
    try:
        events = {k:int(v) for k,v in (line.split() for line in (root/'memory.events').read_text().splitlines())}
    except (OSError,ValueError):
        pass
    return dict(epoch=time.time(),
        cgroup_current_bytes=number(['memory.current','memory/memory.usage_in_bytes']),
        cgroup_peak_bytes=number(['memory.peak','memory/memory.max_usage_in_bytes']),
        cgroup_limit_bytes=number(['memory.max','memory/memory.limit_in_bytes']),
        cgroup_events=events,parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        terminated_children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


class MemoryMonitor:
    def __init__(self,path,interval=30):
        self.path = Path(path)
        self.interval = interval
        self.stop = threading.Event()
        self.write_lock = threading.Lock()
        self.report = dict(samples=0,observed_peak_bytes=0,
            note='Container RAM, distinct from CUDA VRAM. Sampled peaks may miss transients if cgroup peak is unavailable.')

    def sample(self):
        with self.write_lock:
            self._sample()

    def _sample(self):
        sample = memory_sample()
        if self.report['samples']==0:
            self.report['initial_events'] = sample['cgroup_events']
        self.report['samples'] += 1
        self.report['latest'] = sample
        self.report['observed_peak_bytes'] = max([self.report['observed_peak_bytes']]+
            [sample[k] for k in ['cgroup_current_bytes','cgroup_peak_bytes'] if sample[k] is not None])
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report,indent=2)+'\n')
        temporary.replace(self.path)

    def run(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except OSError as exc:
                print('Memory monitoring I/O error: '+str(exc),file=sys.stderr,flush=True)

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()


def forward(args):
    device = 'cuda' if args.device=='gpu' else args.device
    result = [str(BASE/'code/recover.py'),args.command,'--experiment',args.experiment,
              '--seed',str(args.seed),'--device',device,'--resume']
    if args.variant:
        result += ['--variant',args.variant]
    if args.no_total_wall_limit:
        result += ['--no-total-wall-limit']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','check','status'])
    parser.add_argument('--experiment',choices=['aligned','trainmode'],required=True)
    parser.add_argument('--variant',choices=['b0','mlp','mlp_bc'])
    parser.add_argument('--seed',type=int,choices=[11,22],default=11)
    parser.add_argument('--device',choices=['cpu','cuda','gpu'],default='cuda')
    parser.add_argument('--ram-gib',type=int,choices=[48,64],default=48)
    parser.add_argument('--resume',action='store_true',default=True)
    parser.add_argument('--no-total-wall-limit',action='store_true')
    args = parser.parse_args()
    verify()
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0,str(BASE/'code'))
    import recover
    paper = recover.boot(args.experiment)
    sys.argv = forward(args)
    if args.command!='run':
        recover.main()
        return
    if not args.variant:
        parser.error('--variant required for run')
    import server
    server.preflight = memory_preflight(server.preflight,args.ram_gib)
    directory = paper.ROOT/'tgppo/runs/paper-resource-resume-20261109'
    memory_log = directory/'memory'/('{}-{}-s{}-{}-{}.json'.format(
        args.experiment,args.variant,args.seed,int(time.time()),os.getpid()))
    print(json.dumps(dict(entry='paper-resource-resume-20261109',experiment=args.experiment,
        variant=args.variant,seed=args.seed,device_requested=args.device,
        device_effective='cuda' if args.device=='gpu' else args.device,
        requested_ram_gib=args.ram_gib,memory_log=str(memory_log)),ensure_ascii=False),flush=True)
    with MemoryMonitor(memory_log):
        recover.main()


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        print('Another job owns this condition; keep the existing execution lock.',file=sys.stderr)
        sys.exit(2)
