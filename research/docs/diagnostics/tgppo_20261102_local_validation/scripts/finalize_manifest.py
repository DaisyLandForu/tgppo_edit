"""Close the finished scope and fingerprint every delivered file (except self-manifest)."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta

OUT=Path(__file__).resolve().parents[1]


def main():
    path=OUT/'compute_budget.json';budget=json.loads(path.read_text())
    assert all(r['status']=='completed' for r in budget['runs'])
    assert budget['used_seconds']+budget['preparation_reserve_seconds']<=budget['limit_seconds']
    budget.update(status='closed',closed_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  followup_requires_acceptance=True)
    path.write_text(json.dumps(budget,indent=2)+'\n')
    p=OUT/'delivery_validation.json';validation=json.loads(p.read_text())
    validation.update(controlled_subprocess_wall_seconds=budget['used_seconds'],
                      preparation_reserve_seconds=budget['preparation_reserve_seconds'],budget_limit_seconds=600,
                      budget_with_reserve_seconds=budget['used_seconds']+budget['preparation_reserve_seconds'],
                      budget_closed=True)
    p.write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(algorithm='sha256',excludes='delivery_manifest.json',files=files),indent=2)+'\n')
    print(json.dumps(dict(status='closed',files=len(files),compute_seconds=budget['used_seconds'],
                         reserve_seconds=budget['preparation_reserve_seconds'],limit_seconds=budget['limit_seconds'])))


if __name__=='__main__':main()
