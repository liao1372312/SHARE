from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


METRICS = (
    "repair_sr", "hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate",
    "normalized_return", "human_minutes_per_100", "escalation_f1",
    "collaboration_regret", "human_budget_violation", "unsafe_budget_violation",
)


def aggregate(rows: Iterable[Mapping], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group_key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        record = {key: value for key, value in zip(keys, group_key)}
        record["seeds"] = len({int(item["seed"]) for item in items if "seed" in item})
        record["rows"] = len(items)
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items if item.get(metric) is not None])
            if values.size:
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_p2_5"] = float(np.percentile(values, 2.5))
                record[f"{metric}_p97_5"] = float(np.percentile(values, 97.5))
        record["synthetic_evidence"] = all(bool(item.get("synthetic_evidence", False)) for item in items)
        output.append(record)
    return output


def corruption_deltas(rows: list[Mapping]) -> list[dict]:
    clean = {int(row["seed"]): row for row in rows if row.get("experiment") == "ablation" and row.get("policy") == "full_share"}
    output = []
    for row in rows:
        if row.get("experiment") != "robustness":
            continue
        seed = int(row["seed"])
        if seed not in clean:
            raise ValueError(f"missing clean full_share row for seed {seed}")
        delta = {"seed": seed, "corruption_rate": float(row["corruption_rate"]), "synthetic_evidence": bool(row.get("synthetic_evidence", False))}
        for metric in ("repair_sr", "uar", "unsafe_autonomy_cost_rate", "hir", "normalized_return"):
            delta[f"delta_{metric}"] = float(row[metric]) - float(clean[seed][metric])
        output.append(delta)
    return output


def aggregate_deltas(rows: list[Mapping]) -> list[dict]:
    groups: dict[float, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[float(row["corruption_rate"])].append(row)
    output = []
    for rate, items in sorted(groups.items()):
        record = {"corruption_rate": rate, "seeds": len(items), "synthetic_evidence": all(bool(item["synthetic_evidence"]) for item in items)}
        for metric in ("delta_repair_sr", "delta_uar", "delta_unsafe_autonomy_cost_rate", "delta_hir", "delta_normalized_return"):
            values = np.asarray([float(item[metric]) for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_p2_5"] = float(np.percentile(values, 2.5))
            record[f"{metric}_p97_5"] = float(np.percentile(values, 97.5))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[Mapping]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five-seed synthetic SHARE RQ3 diagnostics.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = {name: args.input / name for name in ("summary.json", "stratified.json", "longitudinal.json", "manifest.json")}
    if not all(path.is_file() for path in sources.values()):
        raise SystemExit("input must contain summary.json, stratified.json, longitudinal.json, and manifest.json")
    raw = json.loads(sources["summary.json"].read_text(encoding="utf-8"))
    stratified = json.loads(sources["stratified.json"].read_text(encoding="utf-8"))
    longitudinal = json.loads(sources["longitudinal.json"].read_text(encoding="utf-8"))
    deltas = corruption_deltas(raw)
    payload = {
        "evidence_warning": "All rows are synthetic development diagnostics, not practitioner, deployment, or independent held-out evidence.",
        "ablation": aggregate([row for row in raw if row.get("experiment") == "ablation"], ("policy",)),
        "corruption": aggregate_deltas(deltas),
        "stratified": aggregate(stratified, ("dimension", "stratum")),
        "longitudinal": aggregate(longitudinal, ("mode", "round")),
        "source_hashes": {name: sha256(path) for name, path in sources.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rq3_aggregate.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.out / "rq3_ablation.csv", payload["ablation"])
    write_csv(args.out / "rq3_corruption.csv", payload["corruption"])
    write_csv(args.out / "rq3_stratified.csv", payload["stratified"])
    write_csv(args.out / "rq3_longitudinal.csv", payload["longitudinal"])
    print(json.dumps({"out": str(args.out), "ablation_rows": len(payload["ablation"]), "strata": len(payload["stratified"]), "longitudinal_rows": len(payload["longitudinal"])}, indent=2))


if __name__ == "__main__":
    main()
