"""Download Table 2 raw MILPs from their original public libraries.

This is NOT the author's Mendeley archive. Record each original source and
SHA256 so that future archive comparisons are possible.
"""
import concurrent.futures
import bz2
import gzip
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
TRAIN = '''30n20b8 air04 air05 cod105 comp21-2idx dcmulti eil33-2
istanbul-no-cutoff l152lav lseu misc03 neos20 neos21 neos-476283 neos648910
pp08aCUTS rmatr100-p10 rmatr100-p5 rmatr200-p5 roi5alpha10n8 sp150x300d
stein27 supportcase7 swath1 vpm2'''.split()
TEST = '''aflow40b app1-2 atlanta-ip bab5 bc1 bell3a bell5 biella1 binkar10_1
blend2 dano3_5 fast0507 harp2 map10 map16715-04 map18 map20 mik-250-20-75-4
mine-166-5 misc07 msc98-ip mspp16 n2seq36q n3seq24 neos11 neos12 neos-1200887
neos-1215259 neos13 neos18 neos-4722843-widden neos-4738912-atrato neos-480878
neos-504674 neos-504815 neos-512201 neos-584851 neos-603073 neos-612125
neos-612162 neos-662469 neos-686190 neos-801834 neos-803219 neos-807639
neos-820879 neos-829552 neos-839859 neos-892255 neos-950242 ns1208400 ns1830653
nu25-pr12 nw04 opm2-z7-s2 p0201 pg pigeon-10 pp08a rail507 roll3000 rout
satellites1-25 seymour1 sp98ir unitcal_7'''.split()
TEMPLATES = [
    'https://miplib.zib.de/WebData/instances/{name}.mps.gz',
    'https://miplib2010.zib.de/download/{name}.mps.gz',
    'https://miplib2010.zib.de/miplib3/miplib3/{name}.mps.gz',
    'https://plato.asu.edu/ftp/milp/{name}.mps.gz',
    'https://coral.ise.lehigh.edu/wp-content/uploads/mip-instances/instances/{name}.mps.bz2',
    'https://coral.ise.lehigh.edu/wp-content/uploads/mip-instances/instances/{name}.mps.gz',
    'https://coral.ise.lehigh.edu/wp-content/uploads/mip-instances/{name}.mps.gz',
    'https://coral.ise.lehigh.edu/wp-content/uploads/mip-instances/{name}.gz',
]

def download(job):
    split, name = job
    dst = ROOT / 'datasets' / (split + '_instances') / (name + '.mps.gz')
    dst.parent.mkdir(parents=True, exist_ok=True)
    previous = ROOT / 'datasets' / 'download_manifest.json'
    existing = json.loads(previous.read_text()) if previous.exists() else []
    old = next((r for r in existing if r['name'] == name and r.get('sha256')), None)
    if dst.exists() and old and hashlib.sha256(dst.read_bytes()).hexdigest() == old['sha256']:
        return old
    errors = []
    for template in TEMPLATES:
        url = template.format(name=name)
        try:
            r = requests.get(url, timeout=(10, 40))
            if r.status_code != 200:
                errors.append([url, r.status_code]); continue
            data = r.content
            raw = (gzip.decompress(data) if data.startswith(b'\x1f\x8b') else
                   bz2.decompress(data) if data.startswith(b'BZh') else data)
            if b'NAME' not in raw[:20000] or b'ROWS' not in raw[:20000] or b'<html' in raw[:1000].lower():
                errors.append([url, 'not MPS']); continue
            if not data.startswith(b'\x1f\x8b'):
                data = gzip.compress(raw, mtime=0)
            dst.write_bytes(data)
            print('OK', split, name, len(data), url, flush=True)
            return dict(name=name, split=split, path=str(dst.relative_to(ROOT)),
                        source_url=url, sha256=hashlib.sha256(data).hexdigest(),
                        bytes=len(data), uncompressed_bytes=len(raw))
        except Exception as e:
            errors.append([url, str(e)[:150]])
    print('MISSING', split, name, flush=True)
    return dict(name=name, split=split, errors=errors)

if __name__ == '__main__':
    jobs = [('train', n) for n in TRAIN] + [('test', n) for n in TEST]
    assert len(TRAIN) == 25 and len(TEST) == 66
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(download, jobs))
    (ROOT / 'datasets' / 'download_manifest.json').write_text(json.dumps(rows, indent=2))
    missing = [r['name'] for r in rows if 'sha256' not in r]
    print('Downloaded', len(rows)-len(missing), '/', len(rows), 'Missing:', missing)
    raise SystemExit(bool(missing))
