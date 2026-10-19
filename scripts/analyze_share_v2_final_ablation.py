from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/share_v2_final_development_ablation_20pct_20260827"
SEEDS = (17, 29, 43, 71, 101)
VARIANTS = (
    "share_v2",
    "share_v2_myopic",
    "share_v2_wo_risk",
    "share_v2_wo_budget",
    "share_v2_wo_paired_replay",
    "share_v2_rollback_025",
)
LABELS = {
    "share_v2": "SHARE (V2)",
    "share_v2_myopic": "V2-Myopic",
    "share_v2_wo_risk": "V2-w/o-Risk",
    "share_v2_wo_budget": "V2-w/o-Budget",
    "share_v2_wo_paired_replay": "V2-w/o-Paired-Replay",
    "share_v2_rollback_025": "V2-Rollback-0.25",
}
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    aggregate = {row["policy"]: row for row in json.loads((RESULTS / "aggregate.json").read_text(encoding="utf-8"))}
    paired = json.loads((RESULTS / "paired_comparisons.json").read_text(encoding="utf-8"))
    payloads = [json.loads((RESULTS / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in SEEDS]

    audit_rows = []
    for variant in VARIANTS:
        budgets = [payload["budget_audit"][variant] for payload in payloads]
        trainings = [payload["training_audit"][variant] for payload in payloads]
        row: dict[str, Any] = {
            "policy": variant,
            "cap_respected_runs": sum(int(item["cap_respected"]) for item in budgets),
            "runs_reaching_cap": sum(int(item["human_interventions"] == item["hard_human_cap"]) for item in budgets),
        }
        for key in (
            "human_interventions", "attempted_human_after_exhaustion", "value_policy_replans",
            "stop_decisions_before_exhaustion", "stop_decisions_after_exhaustion",
        ):
            values = [float(item[key]) for item in budgets]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_std"] = stdev(values)
        for key in ("gradient_steps", "replay_size"):
            values = [float(item[key]) for item in trainings]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_std"] = stdev(values)
        audit_rows.append(row)
    write_csv(RESULTS / "audit_aggregate.csv", audit_rows)

    interpretations = {
        "share_v2_myopic": {
            "verdict": "supported_for_repair_safety_and_stop_on_posthoc_development_data",
            "evidence": "Repair.SR -0.068, UAR +0.036, Stop +0.048 versus Full V2; all three group-cluster intervals exclude zero.",
            "boundary": "Regret interval crosses zero and all five-seed Holm-adjusted p-values equal 1.0.",
        },
        "share_v2_wo_risk": {
            "verdict": "independent_contribution_not_supported",
            "evidence": "Repair.SR -0.002, UAR +0.006, Stop -0.010; all group-cluster intervals cross zero.",
            "boundary": "Only three direct risk coordinates are zeroed; correlated proxies remain.",
        },
        "share_v2_wo_budget": {
            "verdict": "independent_contribution_not_supported_in_final_v2",
            "evidence": "Repair.SR +0.012, UAR +0.008, Stop -0.010, Return -0.025, Regret +0.003; all group-cluster intervals cross zero.",
            "boundary": "Validation-only Human calibration and the deployment hard cap remain, so this isolates training-time budget awareness rather than all budget control.",
        },
        "share_v2_wo_paired_replay": {
            "verdict": "supported_for_safety_and_regret_on_posthoc_development_data",
            "evidence": "UAR +0.029 and Regret +0.159; both group-cluster intervals exclude zero. Repair.SR -0.022 has an interval crossing zero.",
            "boundary": "Exploratory replay produces fewer gradient steps than the fixed paired-replay update schedule, so the result combines branch coverage and training-signal volume.",
        },
        "share_v2_rollback_025": {
            "verdict": "supported_for_repair_and_stop_on_posthoc_development_data",
            "evidence": "Repair.SR -0.027 and Stop +0.023; both group-cluster intervals exclude zero.",
            "boundary": "UAR and Regret intervals cross zero; evaluation always uses the common rollback reward 0.25.",
        },
    }
    atomic_json(RESULTS / "component_interpretation.json", interpretations)

    lines = [
        "# Final SHARE (V2) post-hoc development ablation",
        "",
        "This is already-inspected synthetic development evidence produced after the final confirmation had been opened. It is not preregistered, confirmatory, or a basis for further tuning. Full V2 is loaded from frozen checkpoints; no locked or confirmation data is accessed.",
        "",
        "## Aggregate diagnostic results (mean ± SD, five seeds)",
        "",
        "| Variant | Repair.SR | HIR | UAR | Stop | Return | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = aggregate[variant]
        values = [f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}" for metric in METRICS]
        lines.append(f"| {LABELS[variant]} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Component interpretation",
        "",
        "- **Sequential value:** Myopic loses 6.8 Repair.SR points, raises UAR by 3.6 points, and raises Stop by 4.8 points. The workflow-group intervals for these three changes exclude zero.",
        "- **Risk coordinates:** the direct risk-feature ablation is near Full V2 and all intervals cross zero; their independent contribution remains unsupported.",
        "- **Training-time budget awareness:** w/o-Budget has slightly higher Repair.SR but also slightly higher UAR and lower Return; every interval crosses zero. Final-V2 evidence does not independently support this component, although validation calibration and the online cap remain essential protocol mechanisms.",
        "- **Paired replay:** removing paired replay raises UAR by 2.9 points and Regret by 0.159, with workflow-group intervals excluding zero. The Repair.SR difference is uncertain. Standard replay also uses fewer optimization steps, which limits causal attribution to branch coverage alone.",
        "- **Rollback training cost:** reducing the training coefficient from 0.60 to 0.25 lowers Repair.SR by 2.7 points and raises Stop by 2.3 points; both workflow-group intervals exclude zero under the common evaluation reward.",
        "",
        "All exact tests are descriptive: five seeds imply a minimum two-sided sign-flip p-value of 0.0625, and all 20 primary Holm-adjusted values equal 1.0.",
        "",
        "## Budget and training audit",
        "",
        "| Variant | Human interventions | Requests after exhaustion | Replans | Stop before/after exhaustion | Gradient steps |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    audits = {row["policy"]: row for row in audit_rows}
    for variant in VARIANTS:
        row = audits[variant]
        gradient = "not serialized" if variant == "share_v2" else f"{row['gradient_steps_mean']:.0f}"
        lines.append(
            f"| {LABELS[variant]} | {row['human_interventions_mean']:.1f} | "
            f"{row['attempted_human_after_exhaustion_mean']:.1f} | {row['value_policy_replans_mean']:.1f} | "
            f"{row['stop_decisions_before_exhaustion_mean']:.1f}/{row['stop_decisions_after_exhaustion_mean']:.1f} | {gradient} |"
        )
    lines.extend([
        "",
        "All 30 method-seed runs respect the global cap of 24 Human-intervened episodes. Full-checkpoint optimizer/replay state was intentionally not serialized and is therefore not reconstructed in the audit.",
        "",
    ])
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    hashes = []
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.relative_to(RESULTS)}")
    (RESULTS / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(RESULTS / "REPORT.md"), "files_hashed": len(hashes), "protocol": manifest["protocol_version"]}, indent=2))


if __name__ == "__main__":
    main()
