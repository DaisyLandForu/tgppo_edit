"""Close the compute ledger after the last runner returns, then seal delivery."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parents[1]


def main():
    ledger=json.loads((OUT/'compute_budget.json').read_text())
    assert ledger['status']=='open'
    assert all(r['status']=='completed' for r in ledger['runs']), ledger['runs']
    total=sum(r['elapsed_seconds'] for r in ledger['runs'])
    assert total+ledger['preparation_reserve_seconds']<ledger['limit_seconds']
    receipt=json.loads((OUT/'delivery_validation.json').read_text())
    assert receipt['status']=='passed'
    ledger.update(status='closed',closed_at_utc=datetime.now(timezone.utc).isoformat(),used_seconds=total,
                  remaining_seconds=ledger['limit_seconds']-ledger['preparation_reserve_seconds']-total,
                  followup_requires_acceptance=True)
    (OUT/'compute_budget.json').write_text(json.dumps(ledger,indent=2)+'\n')
    receipt.update(controlled_subprocess_wall_seconds=total,preparation_reserve_seconds=ledger['preparation_reserve_seconds'],
                   budget_with_reserve_seconds=total+ledger['preparation_reserve_seconds'],budget_limit_seconds=600,budget_closed=True)
    (OUT/'delivery_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='delivery_manifest.json' and 'mpl-cache' not in p.parts}
    manifest=dict(algorithm='sha256',excludes=['delivery_manifest.json','logs/mpl-cache/*'],files=files)
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    assert all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in files.items())
    print(json.dumps(dict(status='closed',compute_seconds=total,reserve_seconds=ledger['preparation_reserve_seconds'],files_verified=len(files))))


if __name__=='__main__':
    main()
