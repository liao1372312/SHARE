from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently scheduled SHARE-Guard seed screens without evaluating data.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    seen: set[tuple[str, int]] = set()
    sources = {}
    warnings = set()
    for root in args.inputs:
        path = root / "results.json" if root.is_dir() else root
        payload = json.loads(path.read_text(encoding="utf-8"))
        warnings.add(payload.get("evidence_warning"))
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for run in payload.get("runs", []):
            key = (str(run["method"]), int(run["seed"]))
            if key in seen:
                raise ValueError(f"duplicate run {key}")
            seen.add(key)
            runs.append(run)
    if len(warnings) != 1:
        raise ValueError("inputs have inconsistent evidence warnings")
    output = {
        "evidence_warning": warnings.pop(),
        "merge_warning": "Scheduling-only merge. Each source retains its own protocol manifest; no new data were evaluated.",
        "source_results_sha256": sources,
        "runs": sorted(runs, key=lambda row: (int(row["seed"]), str(row["method"]))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "runs": len(runs), "seeds": sorted({seed for _, seed in seen})}, indent=2))


if __name__ == "__main__":
    main()
