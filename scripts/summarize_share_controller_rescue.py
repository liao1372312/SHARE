from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
)
DIAGNOSTICS = (
    "repair_sr",
    "hir",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "p_ai",
    "p_human",
    "p_stop",
    "q_ai_raw_mean",
    "q_human_raw_mean",
    "q_stop_raw_mean",
    "q_ai_adjusted_mean",
    "q_human_adjusted_mean",
    "q_stop_adjusted_mean",
    "beta_h",
    "beta_u",
    "human_calibration_penalty",
    "ai_calibration_penalty",
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit(f"No seed_*.json files found in {root}")
    return runs


def aggregate_phases(runs: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for policy, rows in run["diagnostics"].items():
            for row in rows:
                if row["phase"] in {"selected_checkpoint_pre_calibration", "post_calibration"}:
                    buckets[(policy, row["phase"])].append(row)
    output: dict[str, Any] = defaultdict(dict)
    for (policy, phase), rows in buckets.items():
        output[policy][phase] = {
            key: stats([float(row[key]) for row in rows if row.get(key) is not None])
            for key in DIAGNOSTICS
            if any(row.get(key) is not None for row in rows)
        }
    return dict(output)


def write_trajectory_csv(root: Path, runs: list[dict[str, Any]]) -> None:
    fields = ["seed", "policy", "epoch", "phase", "episodes", "decision_states", *DIAGNOSTICS]
    with (root / "diagnostic_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            for policy, rows in run["diagnostics"].items():
                for row in rows:
                    writer.writerow({"seed": run["seed"], "policy": policy, **row})


def write_epoch_means(root: Path, runs: list[dict[str, Any]]) -> None:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for policy, rows in run["diagnostics"].items():
            for row in rows:
                if row["phase"] == "epoch_end":
                    buckets[(policy, int(row["epoch"]))].append(row)
    fields = ["policy", "epoch", *[f"{key}_mean" for key in DIAGNOSTICS]]
    with (root / "diagnostic_epoch_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (policy, epoch), rows in sorted(buckets.items()):
            values: dict[str, Any] = {"policy": policy, "epoch": epoch}
            for key in DIAGNOSTICS:
                present = [float(row[key]) for row in rows if row.get(key) is not None]
                values[f"{key}_mean"] = mean(present) if present else ""
            writer.writerow(values)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_report(root: Path, aggregate: list[dict[str, Any]], phases: dict[str, Any]) -> None:
    by_policy = {row["policy"]: row for row in aggregate}
    h = by_policy["share_h"]
    full = by_policy["full_share"]
    full_pre = phases["full_share"]["selected_checkpoint_pre_calibration"]
    full_post = phases["full_share"]["post_calibration"]
    h_post = phases["share_h"]["post_calibration"]
    lines = [
        "# SHARE controller rescue: frozen 20% diagnostic",
        "",
        "Evidence class: synthetic development diagnostic (not an independent locked test).",
        "",
        "## Five-seed evaluation",
        "",
        "| Controller | Repair.SR | HIR | UAR | Unsafe cost | Stop | Return | Absolute gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name in ("share_bandit", "share_ddqn", "share_h", "full_share"):
        row = by_policy[name]
        lines.append(
            f"| {name} | {fmt(row['repair_sr_mean'])} | {fmt(row['hir_mean'])} | "
            f"{fmt(row['uar_mean'])} | {fmt(row['unsafe_autonomy_cost_rate_mean'])} | "
            f"{fmt(row['stopped_rate_mean'])} | {fmt(row['normalized_return_mean'])} | "
            f"{'PASS' if row['passes_absolute_gate_on_means'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Gate: Repair.SR >= 0.92; 0.18 <= HIR <= 0.22; UAR < 0.12; Stop < 0.05.",
            "",
            "## Mechanism diagnosis",
            "",
            f"- SHARE-H restores Repair.SR by {h['repair_sr_mean'] - full['repair_sr_mean']:+.3f} "
            f"and reduces Stop by {h['stopped_rate_mean'] - full['stopped_rate_mean']:+.3f} relative to Full SHARE.",
            f"- Full SHARE's selected checkpoint before calibration has mean Stop={fmt(full_pre['stopped_rate']['mean'])}; "
            f"after joint calibration it is Stop={fmt(full_post['stopped_rate']['mean'])}.",
            f"- At that Full SHARE checkpoint, mean beta_U={fmt(full_pre['beta_u']['mean'])}. "
            f"After calibration, the explicit AI calibration penalty is {fmt(full_post['ai_calibration_penalty']['mean'])}.",
            f"- Full SHARE post-calibration adjusted Q means are AI={fmt(full_post['q_ai_adjusted_mean']['mean'])}, "
            f"Human={fmt(full_post['q_human_adjusted_mean']['mean'])}, Stop={fmt(full_post['q_stop_adjusted_mean']['mean'])}. "
            "The calibration penalties push both continuation actions below Stop.",
            f"- SHARE-H post-calibration keeps AI unpenalized and reaches validation Stop={fmt(h_post['stopped_rate']['mean'])}; "
            f"its final evaluation UAR={fmt(h['uar_mean'])}, which misses the strict UAR gate by "
            f"{h['uar_mean'] - 0.12:+.3f}.",
            "",
            "## Decision",
            "",
            "SHARE-H passes Repair.SR, mean HIR, and Stop gates but fails the strict UAR gate. "
            "Therefore do not run the budget curve or locked test from this checkpoint. "
            "The next experiment should tune only the reward-side unsafe-autonomy penalty on train/validation, "
            "with the architecture, state, seeds, split, and human-only constraint frozen.",
            "",
        ]
    )
    (root / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed SHARE controller rescue run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.run_dir)
    aggregate = json.loads((args.run_dir / "aggregate.json").read_text(encoding="utf-8"))
    phases = aggregate_phases(runs)
    write_trajectory_csv(args.run_dir, runs)
    write_epoch_means(args.run_dir, runs)
    (args.run_dir / "phase_summary.json").write_text(
        json.dumps(phases, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report(args.run_dir, aggregate, phases)
    print(json.dumps({"run_dir": str(args.run_dir), "seeds": len(runs), "complete": True}))


if __name__ == "__main__":
    main()
