"""Close this stage's budget and seal the complete delivery including new states."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    budget=json.loads((OUT/'compute_budget.json').read_text())
    assert budget['status']=='open' and all(r['status']=='completed' for r in budget['runs'])
    used=sum(r['elapsed_seconds'] for r in budget['runs'])
    assert used+budget['preparation_reserve_seconds']<600
    receipt=json.loads((OUT/'delivery_validation.json').read_text());assert receipt['status']=='passed'
    steps=json.loads((OUT/'optimizer_steps.json').read_text())
    assert len(steps['entries'])==4 and all(r['status']=='completed' for r in steps['entries'])
    steps.update(stage_status='closed',reason='Authorized per-seed steps reached; no further updates in this stage')
    (OUT/'optimizer_steps.json').write_text(json.dumps(steps,ensure_ascii=False,indent=2)+'\n')
    budget.update(status='closed',used_seconds=used,remaining_seconds=600-used-budget['preparation_reserve_seconds'],
                  closed_at_utc=datetime.now(timezone.utc).isoformat(),followup_requires_acceptance=True)
    (OUT/'compute_budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    receipt.update(controlled_subprocess_wall_seconds=used,preparation_reserve_seconds=budget['preparation_reserve_seconds'],
                   budget_with_reserve_seconds=used+budget['preparation_reserve_seconds'],budget_limit_seconds=600,budget_closed=True)
    (OUT/'delivery_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json' and 'mpl-cache' not in p.parts}
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(algorithm='sha256',excludes=['delivery_manifest.json','logs/mpl-cache/*'],files=files),ensure_ascii=False,indent=2)+'\n')
    assert all(sha(OUT/p)==h for p,h in files.items())
    print(json.dumps(dict(status='closed',controlled_compute_seconds=used,with_reserve_seconds=used+budget['preparation_reserve_seconds'],files_verified=len(files))))


if __name__=='__main__':main()
