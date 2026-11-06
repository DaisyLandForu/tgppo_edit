"""Restore oversized text logs from lossless gzip without overwriting files."""
from pathlib import Path
import gzip,hashlib,json,shutil
root=Path(__file__).resolve().parents[1]
for item in json.loads((root/'snapshot/COMPRESSED_LARGE_LOGS.json').read_text()):
    src=root/item['compressed_path'];dst=root/item['original_path']
    if dst.exists():raise SystemExit('Destination exists: '+str(dst))
    if hashlib.sha256(src.read_bytes()).hexdigest()!=item['compressed_sha256']:raise SystemExit('Archive SHA mismatch')
    dst.parent.mkdir(exist_ok=True,parents=True)
    with gzip.open(src,'rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g)
    if hashlib.sha256(dst.read_bytes()).hexdigest()!=item['original_sha256']:raise SystemExit('Restored SHA mismatch')
    print(dst.relative_to(root))
