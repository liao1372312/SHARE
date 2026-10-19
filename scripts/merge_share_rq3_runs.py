from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently scheduled SHARE RQ3 seed outputs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[str, list] = {"summary.json": [], "stratified.json": [], "longitudinal.json": []}
    manifests = []
    source_hashes = {}
    seen_seeds: set[int] = set()
    for root in args.inputs:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seeds = {int(value) for value in manifest["seeds"]}
        if seen_seeds & seeds:
            raise ValueError(f"duplicate seeds: {sorted(seen_seeds & seeds)}")
        seen_seeds |= seeds
        manifests.append(manifest)
        for name in rows:
            path = root / name
            rows[name].extend(json.loads(path.read_text(encoding="utf-8")))
            source_hashes[str(path.resolve())] = sha256(path)
        source_hashes[str(manifest_path.resolve())] = sha256(manifest_path)
    invariant_keys = ("episode_sha256", "replay_protocol", "constraint_confidence_z", "synthetic_evidence", "epochs", "updates_per_epoch")
    for key in invariant_keys:
        if len({json.dumps(manifest.get(key), sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"inconsistent manifest field: {key}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        (args.out / name).write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
    merged = {key: manifests[0].get(key) for key in invariant_keys}
    merged.update({
        "evidence_warning": "Synthetic results are engineering evidence only and are not practitioner outcomes.",
        "longitudinal_warning": manifests[0]["longitudinal_warning"],
        "seeds": sorted(seen_seeds),
        "device": sorted({manifest.get("device") for manifest in manifests}),
        "source_hashes": source_hashes,
    })
    (args.out / "manifest.json").write_text(json.dumps(merged, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "seeds": sorted(seen_seeds), "summary_rows": len(rows["summary.json"]), "stratified_rows": len(rows["stratified.json"]), "longitudinal_rows": len(rows["longitudinal.json"])}, indent=2))


if __name__ == "__main__":
    main()
