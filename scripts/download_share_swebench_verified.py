from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
EXPECTED_SHA256 = "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"
RELATIVE_FILE = "data/test-00000-of-00001.parquet"
URLS = (
    f"https://hf-mirror.com/datasets/SWE-bench/SWE-bench_Verified/resolve/{REVISION}/{RELATIVE_FILE}",
    f"https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/{REVISION}/{RELATIVE_FILE}",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "SHARE-SWE-bench-fetch/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            if total:
                print(f"downloaded {downloaded}/{total} bytes", file=sys.stderr)
            else:
                print(f"downloaded {downloaded} bytes", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the pinned SWE-bench Verified Parquet file.")
    parser.add_argument("--out", type=Path, default=Path("/tmp/swebench_verified.parquet"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if not 10 <= args.timeout <= 1800:
        parser.error("--timeout must be between 10 and 1800 seconds")
    if not 1 <= args.retries <= 10:
        parser.error("--retries must be between 1 and 10")
    return args


def main() -> None:
    args = parse_args()
    if args.out.exists():
        if args.out.is_file() and sha256_file(args.out) == EXPECTED_SHA256:
            print(f"already verified: {args.out}")
            return
        raise SystemExit(f"refusing to overwrite existing unverified path: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".part")
    failures: list[str] = []
    try:
        for url in URLS:
            for attempt in range(1, args.retries + 1):
                print(f"trying source {URLS.index(url) + 1}/{len(URLS)}, attempt {attempt}/{args.retries}", file=sys.stderr)
                try:
                    if partial.exists():
                        partial.unlink()
                    download(url, partial, args.timeout)
                    observed = sha256_file(partial)
                    if observed != EXPECTED_SHA256:
                        raise ValueError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {observed}")
                    partial.replace(args.out)
                    print(f"downloaded and verified: {args.out}")
                    return
                except (OSError, ValueError, urllib.error.URLError) as error:
                    failures.append(f"{url}: {error}")
                    if partial.exists():
                        partial.unlink()
                    if attempt < args.retries:
                        time.sleep(min(2**attempt, 8))
    finally:
        if partial.exists():
            partial.unlink()
    detail = "\n".join(f"- {failure}" for failure in failures)
    raise SystemExit(
        "all pinned download sources failed; this server cannot reach the dataset hosts. "
        "Download the Parquet file on another machine and place it at "
        f"{args.out}.\n{detail}"
    )


if __name__ == "__main__":
    main()
