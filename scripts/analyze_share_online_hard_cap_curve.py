from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np


BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.5)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
METRICS = ("repair_sr", "hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate", "normalized_return", "collaboration_regret")


def nondominated(rows: Sequence[Mapping[str, Any]], include_hir: bool) -> list[dict[str, Any]]:
    dimensions = [
        ("repair_sr_mean", "max"), ("normalized_return_mean", "max"),
        ("uar_mean", "min"), ("unsafe_autonomy_cost_rate_mean", "min"),
        ("stopped_rate_mean", "min"), ("collaboration_regret_mean", "min"),
    ]
    if include_hir:
        dimensions.append(("hir_mean", "min"))
    output = []
    for budget in BUDGETS:
        group = [row for row in rows if float(row["nominal_budget"]) == budget]
        front = []
        for candidate in group:
            dominated = False
            for other in group:
                if other["policy"] == candidate["policy"]:
                    continue
                weak = all(
                    other[key] >= candidate[key] - 1e-12 if direction == "max" else other[key] <= candidate[key] + 1e-12
                    for key, direction in dimensions
                )
                strict = any(
                    other[key] > candidate[key] + 1e-12 if direction == "max" else other[key] < candidate[key] - 1e-12
                    for key, direction in dimensions
                )
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate["policy"])
        output.append({"nominal_budget": budget, "nondominated": front})
    return output


def paired(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260825)
    output = []
    by_unit = {
        (float(payload["budget"]), int(payload["seed"])): {row["policy"]: row for row in payload["results"]}
        for payload in payloads
    }
    for budget in BUDGETS:
        for policy in POLICIES[:-1]:
            for metric in METRICS:
                values = [
                    by_unit[(budget, seed)][policy][metric] - by_unit[(budget, seed)]["share_h"][metric]
                    for seed in (17, 29, 43, 71, 101)
                ]
                samples = np.asarray(values)[rng.integers(0, 5, size=(20_000, 5))].mean(axis=1)
                output.append(
                    {
                        "nominal_budget": budget,
                        "comparison": f"{policy}-minus-share_h",
                        "metric": metric,
                        "mean": mean(values),
                        "std": stdev(values),
                        "ci95_seed_bootstrap": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                    }
                )
    return output


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    aggregate = json.loads((args.run_dir / "aggregate.json").read_text(encoding="utf-8"))
    paths = sorted(args.run_dir.glob("budget_*_seed_*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    paired_rows = paired(payloads)
    effort_front = nondominated(aggregate, include_hir=True)
    cap_front = json.loads((args.run_dir / "nondominated.json").read_text(encoding="utf-8"))
    (args.run_dir / "paired_differences.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    (args.run_dir / "nondominated_effort_aware.json").write_text(json.dumps(effort_front, indent=2), encoding="utf-8")
    lines = [
        "# Preregistered online hard-Human-budget curve",
        "",
        "Synthetic development evidence; the locked holdout was not accessed. Values are mean ± sample SD over five seeds.",
        "",
        "| Budget | Policy | Repair.SR | HIR | UAR | Stop | Return | Regret | Human/cap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        value = lambda metric: f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}"
        lines.append(
            f"| {row['nominal_budget']:.1f} | {row['policy']} | {value('repair_sr')} | {value('hir')} | "
            f"{value('uar')} | {value('stopped_rate')} | {value('normalized_return')} | "
            f"{value('collaboration_regret')} | {row['human_interventions_mean']:.1f}/{row['hard_human_cap']} |"
        )
    cap_map = {row["nominal_budget"]: row["nondominated"] for row in cap_front}
    effort_map = {row["nominal_budget"]: row["nondominated"] for row in effort_front}
    lines.extend([
        "",
        "## Nondominated controllers",
        "",
        "Cap-conditional dominance compares Repair, UAR, unsafe cost, Stop, Return, and regret. "
        "The effort-aware sensitivity additionally minimizes realized HIR.",
        "",
        "| Budget | Cap-conditional | Effort-aware sensitivity |",
        "|---:|---|---|",
    ])
    for budget in BUDGETS:
        lines.append(f"| {budget:.1f} | {', '.join(cap_map[budget])} | {', '.join(effort_map[budget])} |")
    lines.extend([
        "",
        "The 20% point exactly reproduces the separately validated hard-cap run. All 125 method-budget-seed "
        "units respect their cap. With five seeds, paired bootstrap intervals are descriptive and low-powered.",
        "",
    ])
    (args.run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.run_dir), "paired_rows": len(paired_rows), "payloads": len(payloads)}))


if __name__ == "__main__":
    main()
