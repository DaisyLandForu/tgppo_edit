"""Cache the CUDA-11 wheels required by the pinned Torch wheel, verify PyPI hashes."""
import hashlib
import json
import math
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from download_http import open_url

ROOT = Path(__file__).resolve().parent / "wheels"
PACKAGES = {"nvidia-cuda-nvrtc-cu11": "11.8.89", "nvidia-cuda-runtime-cu11": "11.8.89",
            "nvidia-cuda-cupti-cu11": "11.8.87", "nvidia-cudnn-cu11": "8.7.0.84",
            "nvidia-cublas-cu11": "11.11.3.6", "nvidia-cufft-cu11": "10.9.0.58",
            "nvidia-curand-cu11": "10.3.0.86", "nvidia-cusolver-cu11": "11.4.1.48",
            "nvidia-cusparse-cu11": "11.7.5.86", "nvidia-nccl-cu11": "2.19.3",
            "nvidia-nvtx-cu11": "11.8.86", "triton": "2.2.0"}


def main():
    ROOT.mkdir(exist_ok=True)
    records, tasks = [], []
    chunk = 16 * 1024 * 1024
    metadata_cache = ROOT / "torch-dependencies-metadata.json"
    cached = json.loads(metadata_cache.read_text()) if metadata_cache.exists() else {}
    for name, version in PACKAGES.items():
        print("Metadata", name, version, flush=True)
        if name in cached:
            record = cached[name]
            assert record['version'] == version
        else:
            for attempt in range(3):
                try:
                    with open_url(f"https://pypi.org/pypi/{name}/{version}/json", timeout=20) as response:
                        metadata = json.load(response)
                    break
                except Exception:
                    if attempt == 2:
                        raise
            options = [item for item in metadata["urls"] if item["filename"].endswith(".whl")
                       and "x86_64" in item["filename"] and "manylinux" in item["filename"]
                       and ("py3-none" in item["filename"] or "cp310-cp310" in item["filename"])]
            assert options, name
            item = next((item for item in options if 'manylinux2014' in item['filename']), options[0])
            record = {"package": name, "version": version, "filename": item["filename"],
                      "url": item["url"], "bytes": item["size"], "sha256": item["digests"]["sha256"]}
            cached[name] = record
            metadata_cache.write_text(json.dumps(cached, indent=2) + '\n')
        records.append(record)
        if not (ROOT / record["filename"]).exists():
            pieces = ROOT / (record["filename"] + ".parts")
            pieces.mkdir(exist_ok=True)
            tasks.extend((record, index) for index in range(math.ceil(record["bytes"] / chunk)))

    def fetch(task):
        record, index = task
        first, last = index * chunk, min(record["bytes"] - 1, (index + 1) * chunk - 1)
        file = ROOT / (record["filename"] + ".parts") / str(index)
        if file.exists() and file.stat().st_size == last - first + 1:
            return
        for attempt in range(6):
            try:
                present = file.stat().st_size if file.exists() else 0
                if present == last - first + 1:
                    return
                if present > last - first + 1:
                    file.unlink()
                    present = 0
                resume = first + present
                req = urllib.request.Request(record["url"], headers={"Range": f"bytes={resume}-{last}"})
                with open_url(req, timeout=30, prefer_proxy=True) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"].startswith(f"bytes {resume}-{last}/")
                    with file.open("ab" if present else "wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                assert file.stat().st_size == last - first + 1
                return
            except Exception as error:
                print('Retry', record['package'], 'part', index, 'attempt', attempt + 1,
                      type(error).__name__, flush=True)
                if attempt == 5:
                    raise

    print("Caching", len(records), "CUDA dependency wheels,", len(tasks), "parts", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, task) for task in tasks]), 1):
            try:
                future.result()
            except Exception as error:
                failures.append(type(error).__name__)
            if done % 16 == 0 or done == len(tasks):
                print(done, "/", len(tasks), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} download parts failed; preserve cache and rerun: {failures}')
    for record in records:
        target = ROOT / record["filename"]
        if not target.exists():
            pieces = ROOT / (record["filename"] + ".parts")
            with target.with_suffix(".incomplete").open("wb") as stream:
                for index in range(math.ceil(record["bytes"] / chunk)):
                    with (pieces / str(index)).open("rb") as part:
                        shutil.copyfileobj(part, stream, 1024 * 1024)
            target.with_suffix(".incomplete").replace(target)
            shutil.rmtree(pieces)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == record["sha256"], record["filename"]
        print("Verified", record["filename"], flush=True)
    (ROOT / "torch-dependencies-manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
