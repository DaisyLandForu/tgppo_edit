"""Verify hashes and parse all 91 downloaded MPS files with the installed SCIP."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from pyscipopt import Model
ROOT = Path(__file__).resolve().parents[1]

def check(row):
    path = ROOT / row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['name']
    model = Model()
    model.hideOutput()
    model.readProblem(str(path))
    variables = model.getVars()
    out = dict(name=row['name'], split=row['split'], variables=len(variables),
               constraints=len(model.getConss()),
               binary_variables=sum(v.vtype() == 'BINARY' for v in variables),
               integer_variables=sum(v.vtype() == 'INTEGER' for v in variables),
               hash_ok=True, mps_parse_ok=True)
    assert out['variables'] > 0 and out['constraints'] > 0, out
    assert out['binary_variables'] + out['integer_variables'] > 0, out
    model.freeProb()
    return out

if __name__ == '__main__':
    rows = json.loads((ROOT / 'datasets/download_manifest.json').read_text())
    assert len(rows) == 91 and sum(r['split'] == 'train' for r in rows) == 25
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(check, rows))
    (ROOT / 'setup/dataset-validation.json').write_text(json.dumps(result, indent=2))
    print('All 91 hashes and MPS parsing checks passed.')
