from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)


def sign_flip_p(values: list[float]) -> float:
    observed = abs(mean(values))
    permuted = [
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def paired(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260825)
    by_seed = {
        payload["seed"]: {row["policy"]: row for row in payload["results"]}
        for payload in payloads
    }
    output = []
    for policy in POLICIES[:-1]:
        for metric in METRICS:
            values = [
                rows[policy][metric] - rows["share_h"][metric]
                for _, rows in sorted(by_seed.items())
            ]
            indices = rng.integers(0, len(values), size=(20_000, len(values)))
            boot = np.asarray(values)[indices].mean(axis=1)
            output.append(
                {
                    "comparison": f"{policy}-minus-share_h",
                    "metric": metric,
                    "per_seed": {str(seed): rows[policy][metric] - rows["share_h"][metric] for seed, rows in sorted(by_seed.items())},
                    "mean": mean(values),
                    "std": stdev(values),
                    "ci95_seed_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "p_exact_sign_flip": sign_flip_p(values),
                }
            )
    return output


def pareto(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {row["policy"]: row for row in aggregate}
    reference = rows["share_h"]
    dimensions = (
        ("repair_sr_mean", "max"),
        ("hir_mean", "min"),
        ("uar_mean", "min"),
        ("unsafe_autonomy_cost_rate_mean", "min"),
        ("stopped_rate_mean", "min"),
        ("normalized_return_mean", "max"),
        ("collaboration_regret_mean", "min"),
    )
    comparisons = []
    for policy in POLICIES[:-1]:
        row = rows[policy]
        weak = all(
            row[key] >= reference[key] - 1e-12 if direction == "max" else row[key] <= reference[key] + 1e-12
            for key, direction in dimensions
        )
        strict = any(
            row[key] > reference[key] + 1e-12 if direction == "max" else row[key] < reference[key] - 1e-12
            for key, direction in dimensions
        )
        comparisons.append({"comparator": policy, "dominates_share_h": weak and strict})
    return {
        "scope": "five-seed means under the identical online 24-episode Human cap",
        "dimensions": {"maximize": ["repair_sr", "normalized_return"], "minimize": ["hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate", "collaboration_regret"]},
        "comparisons": comparisons,
        "share_h_nondominated": not any(row["dominates_share_h"] for row in comparisons),
        "warning": "Synthetic development evidence; nondominance here is not an independent locked-test claim.",
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("seed_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    aggregate = json.loads((args.run_dir / "aggregate.json").read_text(encoding="utf-8"))
    paired_rows = paired(payloads)
    pareto_result = pareto(aggregate)
    (args.run_dir / "paired_differences.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    (args.run_dir / "pareto.json").write_text(json.dumps(pareto_result, indent=2), encoding="utf-8")
    lines = [
        "# Hard-cap detailed analysis",
        "",
        "All methods used the same frozen lexicographic episode order and a maximum of 24 Human-intervened episodes.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Policy | Repair.SR | HIR | UAR | Stop | Return | Regret | Human |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        for row in payload["results"]:
            lines.append(
                f"| {payload['seed']} | {row['policy']} | {fmt(row['repair_sr'])} | {fmt(row['hir'])} | "
                f"{fmt(row['uar'])} | {fmt(row['stopped_rate'])} | {fmt(row['normalized_return'])} | "
                f"{fmt(row['collaboration_regret'])} | {row['human_interventions']}/24 |"
            )
    lines.extend([
        "",
        "## Paired differences versus SHARE-H",
        "",
        "Differences are comparator minus SHARE-H; inference resamples only five seed pairs and is therefore low-powered.",
        "",
        "| Comparator | Metric | Mean ± SD | 95% bootstrap CI | Exact sign-flip p |",
        "|---|---|---:|---:|---:|",
    ])
    for row in paired_rows:
        ci = row["ci95_seed_bootstrap"]
        lines.append(
            f"| {row['comparison'].removesuffix('-minus-share_h')} | {row['metric']} | "
            f"{fmt(row['mean'])} ± {fmt(row['std'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row['p_exact_sign_flip'])} |"
        )
    lines.extend([
        "",
        "## Pareto assessment",
        "",
        f"SHARE-H nondominated under the declared seven-dimensional hard-cap comparison: `{pareto_result['share_h_nondominated']}`.",
        "No comparator is weakly better on every dimension and strictly better on at least one. This is development-only evidence.",
        "",
    ])
    (args.run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.run_dir), "seeds": len(payloads), "paired_rows": len(paired_rows)}))


if __name__ == "__main__":
    main()
