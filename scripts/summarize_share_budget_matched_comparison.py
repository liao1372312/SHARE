from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    aggregate = json.loads((args.run_dir / "aggregate.json").read_text(encoding="utf-8"))
    paired = json.loads((args.run_dir / "paired_differences.json").read_text(encoding="utf-8"))
    pareto = json.loads((args.run_dir / "pareto.json").read_text(encoding="utf-8"))
    seeds = sorted(args.run_dir.glob("seed_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in seeds]

    lines = [
        "# Strict 20% HIR-matched controller comparison",
        "",
        "Evidence class: synthetic development evaluation already inspected during controller rescue. "
        "No locked holdout was accessed.",
        "",
        "Comparator thresholds/penalties were selected using validation episode-level HIR only. "
        "No development-evaluation metric was fed back into calibration.",
        "",
        "## Mean ± sample SD over five seeds",
        "",
        "| Policy | Repair.SR | HIR | UAR | Unsafe cost | Stop | Return | Regret | Mean HIR eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate:
        value = lambda metric: f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}"
        lines.append(
            f"| {row['policy']} | {value('repair_sr')} | {value('hir')} | {value('uar')} | "
            f"{value('unsafe_autonomy_cost_rate')} | {value('stopped_rate')} | "
            f"{value('normalized_return')} | {value('collaboration_regret')} | "
            f"{'yes' if row['mean_hir_in_band'] else 'no'} |"
        )

    lines.extend(["", "## Per-seed evaluation", "", "| Seed | Policy | Repair.SR | HIR | UAR | Stop | Return | Regret | HIR eligible |", "|---:|---|---:|---:|---:|---:|---:|---:|:---:|"])
    for payload in payloads:
        for row in payload["results"]:
            lines.append(
                f"| {payload['seed']} | {row['policy']} | {fmt(row['repair_sr'])} | {fmt(row['hir'])} | "
                f"{fmt(row['uar'])} | {fmt(row['stopped_rate'])} | {fmt(row['normalized_return'])} | "
                f"{fmt(row['collaboration_regret'])} | {'yes' if row['evaluation_hir_in_band'] else 'no'} |"
            )

    lines.extend([
        "",
        "## Paired differences against SHARE-H",
        "",
        "All differences are comparator minus SHARE-H. CIs resample the five seed pairs; exact sign-flip tests "
        "have low power at n=5. These differences are descriptive when the comparator misses the HIR band.",
        "",
        "| Comparator | Metric | Mean ± SD | 95% bootstrap CI | p (sign flip) | p (Holm, primary only) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in paired:
        comparator = row["comparison"].removesuffix("-minus-share_h")
        ci = row["ci95_seed_bootstrap"]
        holm = "—" if "p_holm" not in row else fmt(row["p_holm"])
        lines.append(
            f"| {comparator} | {row['metric']} | {fmt(row['mean'])} ± {fmt(row['std'])} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row['p_exact_sign_flip'])} | {holm} |"
        )

    lines.extend([
        "",
        "## Pareto decision",
        "",
        f"Result: `{pareto['conclusion']}`.",
        "",
        "No comparator achieved five-seed mean evaluation HIR in [0.18, 0.22]. Consequently, this run does "
        "not establish that SHARE-H is on the Pareto frontier; it establishes that validation-only scalar "
        "calibration does not transport to the development evaluation split for the current comparator set. "
        "Using evaluation HIR to retune these thresholds after seeing this result would invalidate the comparison.",
        "",
    ])
    (args.run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    source_files = [
        ROOT / "scripts/run_share_budget_matched_comparison.py",
        ROOT / "scripts/summarize_share_budget_matched_comparison.py",
        ROOT / "share/dqn.py",
        ROOT / "share/evaluation.py",
        ROOT / "share/policies.py",
    ]
    audit: dict[str, Any] = {
        "seed_files": [path.name for path in seeds],
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "locked_holdout_accessed": False,
    }
    (args.run_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.run_dir / 'report.md'), "seeds": len(seeds)}))


if __name__ == "__main__":
    main()
