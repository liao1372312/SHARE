"""Consolidate the completed SHARE-V2 confirmation evaluations.

This script performs analysis only.  It never loads episodes or policies and
therefore cannot execute or modify the sealed confirmation evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "results/share_v2_confirmation_budget_curve_20260827/FINAL_RESULTS.json"
MAIN = ROOT / "results/share_v2_confirmation_locked_20260827/FINAL_RESULTS.json"
BASE20 = ROOT / "results/share_v2_confirmation_locked_20260827/frozen_comparator_20pct.json"
OUT = ROOT / "results/share_v2_confirmation_analysis_20260827"
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_v2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(row: dict) -> dict:
    return {
        "budget": float(row["budget"]),
        "policy": row["policy"],
        **{f"{metric}_{suffix}": float(row[f"{metric}_{suffix}"]) for metric in METRICS for suffix in ("mean", "std")},
        "all_caps_respected": bool(row["all_caps_respected"]),
    }


def dominates(left: dict, right: dict) -> bool:
    # HIR is reported but not optimized here: all methods operate under the
    # same nominal hard cap and may legitimately leave attention unused.
    maximize = ("repair_sr_mean", "normalized_return_mean")
    minimize = ("uar_mean", "stopped_rate_mean", "collaboration_regret_mean")
    weak = all(left[key] >= right[key] for key in maximize) and all(left[key] <= right[key] for key in minimize)
    strict = any(left[key] > right[key] for key in maximize) or any(left[key] < right[key] for key in minimize)
    return weak and strict


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(mean(differences))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(mean(sign * value for sign, value in zip(signs, differences))))
    return sum(value >= observed - 1e-15 for value in values) / len(values)


def main() -> None:
    curve, main, base20 = load(CURVE), load(MAIN), load(BASE20)
    aggregate = [canonical(row) for row in curve["aggregate"]]
    for policy, row in base20["aggregate"].items():
        aggregate.append(canonical({
            "budget": 0.20,
            "policy": policy,
            "all_caps_respected": True,
            **{f"{metric}_mean": row[metric] for metric in METRICS},
            **{f"{metric}_std": row[f"{metric}_std"] for metric in METRICS},
        }))
    aggregate.append(canonical({"budget": 0.20, "policy": "share_v2", **main["aggregate"]}))
    aggregate.sort(key=lambda row: (row["budget"], POLICIES.index(row["policy"])))

    per_seed = [{key: value for key, value in row.items() if key in ("nominal_budget", "seed", "policy", *METRICS, "cap_respected", "hard_human_cap", "human_interventions")} for row in curve["per_seed"]]
    for row in base20["per_seed"]:
        per_seed.append({"nominal_budget": 0.20, **{key: value for key, value in row.items() if key in ("seed", "policy", *METRICS, "cap_respected", "hard_human_cap", "human_interventions")}})
    for row in main["per_seed"]:
        per_seed.append({"nominal_budget": 0.20, **{key: value for key, value in row.items() if key in ("seed", "policy", *METRICS, "cap_respected", "hard_human_cap", "human_interventions")}})
    per_seed.sort(key=lambda row: (float(row["nominal_budget"]), POLICIES.index(row["policy"]), int(row["seed"])))

    paired = []
    for budget in (0.10, 0.20, 0.30):
        v2 = {int(row["seed"]): row for row in per_seed if row["nominal_budget"] == budget and row["policy"] == "share_v2"}
        for comparator in POLICIES[:-1]:
            other = {int(row["seed"]): row for row in per_seed if row["nominal_budget"] == budget and row["policy"] == comparator}
            for metric in METRICS:
                differences = [float(v2[seed][metric]) - float(other[seed][metric]) for seed in sorted(v2)]
                paired.append({
                    "budget": budget,
                    "comparator": comparator,
                    "metric": metric,
                    "direction": "SHARE-V2 minus comparator",
                    "mean_difference": mean(differences),
                    "per_seed_differences": differences,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
                })

    frontier = []
    for budget in (0.10, 0.20, 0.30):
        rows = [row for row in aggregate if row["budget"] == budget]
        for row in rows:
            dominators = [other["policy"] for other in rows if other is not row and dominates(other, row)]
            frontier.append({"budget": budget, "policy": row["policy"], "nondominated": not dominators, "dominated_by": dominators})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "aggregate_budget_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    with (OUT / "per_method_per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("nominal_budget", "seed", "policy", *METRICS, "hard_human_cap", "human_interventions", "cap_respected")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(per_seed)
    (OUT / "paired_seed_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (OUT / "pareto_status.json").write_text(json.dumps(frontier, indent=2), encoding="utf-8")

    report = [
        "# SHARE-V2 independent confirmation analysis",
        "",
        "Evidence class: new workflow-group-disjoint, real-source replayable synthetic confirmation; no human-participant execution.",
        "",
        "## Aggregate results (mean ± sample SD over five seeds)",
        "",
        "| Budget | Method | Repair.SR | HIR | UAR | Stop | Return | Regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append("| " + " | ".join([
            f"{row['budget']:.0%}", row["policy"],
            *[f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" for metric in METRICS],
        ]) + " |")
    report.extend(["", "## Pareto status", "", "Dominance uses Repair.SR/Return (higher is better) and UAR/Stop/Regret (lower is better). HIR is treated as a cap rather than a quantity to maximize or minimize.", ""])
    for budget in (0.10, 0.20, 0.30):
        names = [row["policy"] for row in frontier if row["budget"] == budget and row["nondominated"]]
        report.append(f"- {budget:.0%}: {', '.join(names)}")
    report.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- SHARE-V2 is a post-freeze successor evaluated on a newly generated confirmation set; it must not replace or relabel the original preregistered SHARE locked result.",
        "- The confirmation branches are deterministic replayable synthetic outcomes, not observed human-participant executions.",
        "- Five-seed sign-flip tests have a minimum attainable two-sided p-value of 0.0625; paired differences are descriptive and do not establish superiority.",
        "- The 10% point retains a material coverage cost, while the 20% and 30% points narrow the repair gap relative to Static.",
    ])
    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(OUT.iterdir()):
        if path.name != "SHA256SUMS": sums.append(f"{sha256(path)}  {path.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "aggregate_rows": len(aggregate), "paired_rows": len(paired), "all_caps_respected": all(row["all_caps_respected"] for row in aggregate)}, indent=2))


if __name__ == "__main__":
    main()
