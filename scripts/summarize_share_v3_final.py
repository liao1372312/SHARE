"""Create publication-neutral tables from the sealed V3 final result."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/share_v3_final_evaluation_20260827/FINAL_RESULTS.json"
OUT = ROOT / "results/share_v3_final_analysis_20260827"
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("refusing to overwrite V3 final analysis")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True)
    write_csv(OUT / "aggregate_budget_results.csv", data["aggregate"])
    write_csv(OUT / "per_method_per_seed_metrics.csv", data["per_seed"])
    paired = [{key: value for key, value in row.items() if key != "per_seed_differences"} for row in data["paired_differences"]]
    write_csv(OUT / "paired_seed_differences.csv", paired)
    lines = [
        "# SHARE-V3 third-holdout analysis",
        "",
        "Evidence: one-shot evaluation on 400 new workflow-group-disjoint real-source replayable synthetic episodes; no human-participant execution.",
        "",
        "| Budget | Method | Repair.SR | HIR | UAR | Stop | Return | Regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["aggregate"]:
        values = [f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" for metric in METRICS]
        lines.append(f"| {row['budget']:.0%} | {row['policy']} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Findings",
        "",
        "1. V3 removes abstention (Stop=0 at every budget) and obtains Repair.SR around 0.91.",
        "2. V3 consumes only 6--7% Human attention despite nominal caps of 10--30%; its UAR remains about 0.164--0.169.",
        "3. At 20%, Conf weakly dominates V3 on the declared five objectives; at 30%, Static dominates V3.",
        "4. V3 therefore does not preserve the low-UAR advantage of V2 and should not replace V2 in the manuscript.",
        "5. Exact five-seed sign-flip tests have minimum two-sided p=0.0625; no superiority claim is made.",
        "",
        "## Evidence boundary",
        "",
        "The old original 400 was reclassified as V3 training and the second 400 as V3 validation. Only this third 400 is V3 final evidence. All branch outcomes remain deterministic replay simulations.",
    ])
    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = sorted(path for path in OUT.iterdir() if path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "aggregate_rows": len(data["aggregate"]), "runs": data["runs"], "all_caps_respected": data["all_caps_respected"]}, indent=2))


if __name__ == "__main__":
    main()
