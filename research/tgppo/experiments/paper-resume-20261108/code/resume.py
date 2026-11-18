"""Resume frozen paper-aligned training across CPU models without losing records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
UPSTREAM = PACKAGE.parent/'paper-aligned-20261106'
sys.path.insert(0,str(UPSTREAM/'code'))
import paper


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_wrapper():
    manifest=json.loads((PACKAGE/'release.json').read_text())
    if not manifest.get('checks_passed'):
        raise ValueError('Resume wrapper checks have not passed')
    for relative,digest in manifest['files'].items():
        if sha(PACKAGE/relative)!=digest:
            raise ValueError('Changed resume wrapper file: '+relative)
    if sha(UPSTREAM/'docs/release.json')!=manifest['upstream_release_sha256']:
        raise ValueError('Frozen upstream release changed')
    return manifest


def cohort_cached_job(original):
    def cached(job,folder,identity,c,resume=False):
        folder=Path(folder)
        path=folder/'identity.json'
        if job['mode']!='collect' or not path.exists() or paper.read(path)==identity:
            return original(job,folder,identity,c,resume)
        old=paper.read(path)
        # A CPU move is allowed; changing policy, solver task, action RNG, or
        # software is not a license to reuse or silently replace that episode.
        if {k:v for k,v in old.items() if k!='stack'} != \
                {k:v for k,v in identity.items() if k!='stack'}:
            raise ValueError('Resume task/checkpoint/action identity changed; only CPU model migration allowed')
        previous,current=old.get('stack',{}),identity.get('stack',{})
        if (not previous.get('cpu_model') or not current.get('cpu_model') or
                {k:v for k,v in previous.items() if k!='cpu_model'} !=
                {k:v for k,v in current.items() if k!='cpu_model'}):
            raise ValueError('Resume software/architecture identity changed; only CPU model migration allowed')
        if not resume:
            raise ValueError('CPU model migration requires explicit resume')
        destination=folder/'resume-cohorts'/paper.fp(identity)
        provenance=dict(reason='uncommitted collection restarted on a different CPU model',
            source_identity=str(path),source_identity_sha256=sha(path),
            previous_stack=previous,current_stack=current,
            checkpoint_sha256=identity['checkpoint_sha256'],
            sampling_seed=identity['sampling_seed'],wrapper_sha256=sha(Path(__file__)),
            original_record_preserved=True,optimizer_or_protocol_changed=False)
        record=destination/'migration.json'
        if record.exists() and paper.read(record)!=provenance:
            raise ValueError('Migration provenance changed')
        paper.write(record,provenance)
        return original(job,destination,identity,c,resume)
    return cached


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','check'])
    parser.add_argument('--variant',choices=paper.VARIANTS)
    parser.add_argument('--seed',type=int,choices=paper.SEEDS,default=11)
    parser.add_argument('--resume',action='store_true',default=True)
    args=parser.parse_args()
    verify_wrapper()
    if args.command=='check':
        paper.verify()
        print('Resume wrapper and frozen upstream verified; no GPU, solver or training started.')
        return
    if not args.variant:
        parser.error('--variant required')
    out=paper.RUN/'training'/('{}-s{}'.format(args.variant,args.seed))
    if not (out/'latest.pt').is_file():
        raise ValueError('Resume-only entry requires an existing latest.pt; cannot start fresh training')
    print(json.dumps(dict(entry='paper-resume-20261108',wrapper_sha256=sha(Path(__file__)),
        upstream=str(UPSTREAM),variant=args.variant,seed=args.seed,
        cumulative_budget_preserved=True,training_modes_unchanged=True)),flush=True)
    paper.cached_job=cohort_cached_job(paper.cached_job)
    sys.argv=[str(UPSTREAM/'code/paper.py'),'run','--variant',args.variant,
              '--seed',str(args.seed),'--resume']
    paper.main()


if __name__=='__main__':
    main()
