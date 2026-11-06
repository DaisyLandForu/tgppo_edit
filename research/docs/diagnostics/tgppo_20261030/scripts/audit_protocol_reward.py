"""Additional metadata-only protocol/reward audit, no solves or model changes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
RUN=ROOT/'tgppo/runs/server6-20261022'


def fingerprint(x):
    # Match project.utils.protocol.fingerprint byte-for-byte (default spacing).
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile(x):
    return dict(zip(['min','p50','p95','max'],map(float,np.quantile(x,[0,.5,.95,1]))))


def main():
    original=json.loads((ROOT/'docs/stage1/protocol_draft.json').read_text())
    protocol=json.loads((ROOT/'migration/2026-10-22/server-run/protocol.json').read_text())
    code=ROOT/'tgppo/experiments/server-20261022/code'
    runtime_hash=fingerprint({str(p.relative_to(code)):sha(p) for p in sorted(code.rglob('*.py'))})
    protocol.update(train=original['data']['train'],development=original['data']['development'],
                    instance_hashes=original['data']['hashes'],
                    public_hparams_sha256=sha(ROOT/'tgppo/configs/paper_hparams.json'),
                    calibration_statistics_sha256=sha(ROOT/'tgppo/runs/stage4-calibration/final_train_only_statistics.json'),
                    runtime_sources_sha256=runtime_hash)
    expected=json.loads((RUN/'stage5-results-20261023/audit.json').read_text())['parent_protocol_sha256']
    assert fingerprint(protocol)==expected
    frozen=json.loads((RUN/'post-training-20261023/frozen_selection.json').read_text())
    baselines=json.loads((RUN/'normalization/baselines.json').read_text())['baselines']
    bases=[]
    for b in baselines.values():
        r=b['result']
        bases.append(dict(instance=r['instance_name'],solver_seed=r['solver_seed'],nodes=r['nnodes'],
                          gap=r['gap'],pdi_raw=r['primalDualIntegral'],pdi_complete=r['pdi_native_complete_at_stop'],
                          pdi_tail=r['pdi_tail_completed_readonly'],status=r['status'],
                          split='train' if r['instance_name'] in protocol['train'] else 'development'))
    write('reward_baseline_normalizers.csv',bases)
    entries=[]
    for model in sorted((RUN/'training').iterdir()):
        for p in sorted((model/'collection').rglob('e*.json')):
            d=json.loads(p.read_text())
            entries.append(dict(model=model.name,instance=d['instance_name'],solver_seed=d['solver_seed'],episode_id=d['episode_id'],
                                transitions=d['num_transitions'],pdi=d['pdi_horizon_midpoint'],pdi_raw=d['primalDualIntegral'],
                                pdi_complete=d['pdi_native_complete_at_stop'],pdi_tail=d['pdi_tail_completed_readonly'],
                                pdi_tail_fraction=d['pdi_tail_completed_readonly']/max(d['pdi_native_complete_at_stop'],1e-12),
                                proof=d['proof_completed'],total_reward=d['total_reward'],gap=d['gap'],status=d['status']))
    write('training_metadata_diagnostics.csv',entries)
    trainbase=[r for r in bases if r['split']=='train']
    summary=dict(protocol_expanded_sha256=fingerprint(protocol),runtime_sources_sha256=runtime_hash,
                 raw_protocol_sha256=sha(ROOT/'migration/2026-10-22/server-run/protocol.json'),
                 source_python_files=len(list(code.rglob('*.py'))),train=protocol['train'],development=protocol['development'],
                 baseline_records=len(bases),train_baseline_records=len(trainbase),
                 train_baseline_gap_zero=sum(r['gap']==0 for r in trainbase),
                 train_baseline_pdi_floor_active=sum(r['pdi_raw']<1 for r in trainbase),
                 train_baseline_gap_zero_instances=sorted({r['instance'] for r in trainbase if r['gap']==0}),
                 training_records=len(entries),training_zero_transition_tasks=sum(r['transitions']==0 for r in entries),
                 training_zero_transition_instances=sorted({r['instance'] for r in entries if r['transitions']==0}),
                 terminal_raw_pdi_tail=quantile([r['pdi_tail'] for r in entries]),
                 terminal_raw_pdi_tail_fraction=quantile([r['pdi_tail_fraction'] for r in entries]),
                 terminal_raw_pdi_tail_gt_1pct=sum(r['pdi_tail_fraction']>.01 for r in entries),
                 terminal_raw_pdi_tail_gt_10pct=sum(r['pdi_tail_fraction']>.1 for r in entries),
                 terminal_raw_pdi_tail_note='At-stop getter mismatch only; no per-step decomposition is inferred.',
                 status_counts=dict(Counter(r['status'] for r in entries)),
                 frozen_selection_top_level_keys=list(frozen))
    (OUT/'protocol_reward_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
