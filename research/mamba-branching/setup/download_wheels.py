"""Cache exact public release wheels with resumable HTTP range downloads."""
import hashlib
import json
import math
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from download_http import open_url

ROOT = Path(__file__).resolve().parent / "wheels"
URLS = [
    ("torch-2.2.2+cu118-cp310-cp310-linux_x86_64.whl", "https://download.pytorch.org/whl/cu118/torch-2.2.2%2Bcu118-cp310-cp310-linux_x86_64.whl"),
    ("causal_conv1d-1.4.0+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"),
    ("mamba_ssm-2.2.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"),
]


def main():
    ROOT.mkdir(exist_ok=True)
    records = []
    for filename, url in URLS:
        with open_url(urllib.request.Request(url, method="HEAD"), prefer_proxy="github.com" in url) as response:
            size, resolved = int(response.headers["Content-Length"]), response.url
        target = ROOT / filename
        if not (target.exists() and target.stat().st_size == size):
            pieces = ROOT / (filename + ".parts")
            pieces.mkdir(exist_ok=True)
            chunk = 16 * 1024 * 1024

            def fetch(index):
                first, last = index * chunk, min(size - 1, (index + 1) * chunk - 1)
                file = pieces / str(index)
                if file.exists() and file.stat().st_size == last - first + 1:
                    return
                for attempt in range(6):
                    try:
                        present = file.stat().st_size if file.exists() else 0
                        if present > last - first + 1:
                            file.unlink()
                            present = 0
                        resume = first + present
                        req = urllib.request.Request(resolved, headers={"Range": f"bytes={resume}-{last}"})
                        with open_url(req, timeout=30, prefer_proxy="github" in url) as response:
                            assert response.status == 206
                            assert response.headers["Content-Range"].startswith(f"bytes {resume}-{last}/")
                            with file.open("ab" if present else "wb") as stream:
                                shutil.copyfileobj(response, stream, 1024 * 1024)
                        assert file.stat().st_size == last - first + 1
                        return
                    except Exception:
                        if attempt == 5:
                            raise

            count = math.ceil(size / chunk)
            print("Downloading", filename, size, "bytes", flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(fetch, index) for index in range(count)]
                for done, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if done % 8 == 0 or done == count:
                        print(filename, done, "/", count, flush=True)
            tmp = target.with_suffix(".incomplete")
            with tmp.open("wb") as stream:
                for index in range(count):
                    with (pieces / str(index)).open("rb") as part:
                        shutil.copyfileobj(part, stream, 1024 * 1024)
            tmp.replace(target)
            shutil.rmtree(pieces)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"filename": filename, "url": url, "bytes": size, "sha256": digest.hexdigest()})
        (ROOT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("Ready", filename, flush=True)


if __name__ == "__main__":
    main()
