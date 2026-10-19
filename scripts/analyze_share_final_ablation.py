from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


POLICIES = ("share", "share_myopic", "share_wo_risk", "share_wo_budget", "share_dual")
LABELS = {
    "share": "SHARE",
    "share_myopic": "SHARE-Myopic",
    "share_wo_risk": "SHARE-w/o-Risk",
    "share_wo_budget": "SHARE-w/o-Budget",
    "share_dual": "SHARE-Dual",
}
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads((args.results / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in (17, 29, 43, 71, 101)]
    aggregate = {row["policy"]: row for row in json.loads((args.results / "aggregate.json").read_text(encoding="utf-8"))}
    paired = json.loads((args.results / "paired_comparisons.json").read_text(encoding="utf-8"))
    audit_rows = []
    for policy in POLICIES:
        budget = [payload["budget_audit"][policy] for payload in payloads]
        training = [payload["training_audit"][policy] for payload in payloads]
        row: dict[str, Any] = {
            "policy": policy,
            "seeds": len(payloads),
            "cap_respected_runs": sum(int(item["cap_respected"]) for item in budget),
            "runs_reaching_cap": sum(int(item["human_interventions"] == item["hard_human_cap"]) for item in budget),
        }
        for key in ("human_interventions", "attempted_human_after_exhaustion", "value_policy_replans"):
            values = [float(item[key]) for item in budget]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_std"] = stdev(values)
        for key in ("beta_h", "beta_u", "human_calibration_penalty", "ai_calibration_penalty", "calibration_validation_hir"):
            values = [float(item[key]) for item in training]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_std"] = stdev(values)
        audit_rows.append(row)
    write_csv(args.results / "audit_aggregate.csv", audit_rows)

    interpretations = {
        "share_myopic": {
            "tested_component": "sequential return in the learned action value",
            "observation": "Repair.SR decreases by 0.078 and Stop increases by 0.080; both seed- and group-bootstrap CIs exclude zero.",
            "verdict": "supported_for_repair_and_stop",
            "qualification": "Regret increases only 0.032 and its confidence intervals cross zero.",
        },
        "share_wo_risk": {
            "tested_component": "drift_risk, uncertainty, and evidence_conflict input coordinates",
            "observation": "Repair.SR is unchanged (+0.003); UAR is lower by 0.014 and Regret lower by 0.069, with group-bootstrap intervals crossing zero.",
            "verdict": "not_supported",
            "qualification": "The ablation removes only the declared risk coordinates; correlated proxies remain in the state.",
        },
        "share_wo_budget": {
            "tested_component": "training-time Human dual under retained validation calibration and online hard cap",
            "observation": "Repair.SR decreases by 0.048, Stop increases by 0.050, and post-exhaustion replans rise from 2.4 to 6.6 per run on average.",
            "verdict": "supported_for_budget_aware_routing",
            "qualification": "Executed HIR is capped at 0.20, so proposed Human requests reveal the mechanism more directly than realized HIR.",
        },
        "share_dual": {
            "tested_component": "joint Human and unsafe dual/calibration versus reward-side safety",
            "observation": "Repair.SR decreases by 0.565, Stop rises by 0.587 to 0.608, and Regret increases by 0.848; UAR is lower because the controller stops frequently.",
            "verdict": "supports_rejection_of_dual_formulation",
            "qualification": "beta_U is zero, but the mean AI calibration penalty is positive, identifying calibration-driven conservative collapse.",
        },
    }
    atomic_json(args.results / "component_interpretation.json", interpretations)
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    share_audit = {str(payload["seed"]): payload["training_audit"]["share"] for payload in payloads}
    atomic_json(args.results / "method_freeze.json", {
        "method_name": "SHARE",
        "former_development_name": "SHARE-H",
        "freeze_status": "frozen_before_final_development_ablation",
        "formulation": manifest["definitions"]["share"],
        "seeds": manifest["seeds"],
        "episodes_sha256": manifest["episodes_sha256"],
        "episode_order_sha256": manifest["episode_order_sha256"],
        "base_encoder_sha256": manifest["base_encoder_sha256"],
        "reward_config": manifest["reward_config"],
        "hard_human_episode_cap": manifest["hard_human_episode_cap"],
        "checkpoint_sha256": manifest["frozen_checkpoint_sha256"]["share"],
        "checkpoint_config_by_seed": {seed: value["config"] for seed, value in share_audit.items()},
        "calibration_by_seed": {
            seed: {
                key: value[key] for key in (
                    "beta_h", "beta_u", "human_calibration_penalty", "ai_calibration_penalty",
                    "calibration_human_target", "calibration_unsafe_target",
                    "calibration_validation_hir", "calibration_validation_unsafe",
                )
            } for seed, value in share_audit.items()
        },
        "hard_cap_evaluator_sha256": manifest["source_sha256"]["scripts/run_share_online_hard_cap.py"],
        "locked_holdout_accessed": False,
        "further_development_tuning_permitted": False,
    })

    lines = [
        "# Final preregistered 20% development ablation",
        "",
        "All methods use the same train/validation split, five seeds, lexicographic development-evaluation order, and global online cap of 24 Human-intervened episodes. SHARE and the reused comparator checkpoints are frozen; only Myopic and w/o-Risk were retrained on train/validation. No evaluation outcome was used for calibration and the locked holdout was not accessed.",
        "",
        "## Aggregate results (mean ± SD across five seeds)",
        "",
        "| Method | Repair.SR | HIR | UAR | Stop | Return | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = aggregate[policy]
        values = [f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}" for metric in METRICS]
        lines.append(f"| {LABELS[policy]} | " + " | ".join(values) + " |")
    lines.extend(["", "## Per-seed results", "", "| Seed | Method | Repair.SR | HIR | UAR | Stop | Return | Regret |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for payload in payloads:
        by = {row["policy"]: row for row in payload["results"]}
        for policy in POLICIES:
            row = by[policy]
            lines.append(f"| {payload['seed']} | {LABELS[policy]} | " + " | ".join(fmt(float(row[m])) for m in METRICS) + " |")
    lines.extend(["", "## Paired differences against SHARE", "", "Differences are ablation minus SHARE. Seed bootstrap and workflow-group cluster bootstrap intervals are reported separately.", "", "| Ablation | Metric | Mean Δ | Seed 95% CI | Group-cluster 95% CI | Exact p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in paired:
        seed_ci = row["seed_bootstrap_ci95"]
        group_ci = row["workflow_group_cluster_bootstrap_ci95"]
        holm = f"{row['holm_p_across_16_primary_tests']:.4f}" if "holm_p_across_16_primary_tests" in row else "—"
        lines.append(
            f"| {LABELS[row['comparison'].removesuffix('-minus-share')]} | {row['metric']} | {fmt(row['mean_difference'])} | "
            f"[{fmt(seed_ci[0])}, {fmt(seed_ci[1])}] | [{fmt(group_ci[0])}, {fmt(group_ci[1])}] | "
            f"{row['exact_sign_flip_p_two_sided']:.4f} | {holm} |"
        )
    lines.extend([
        "",
        "With five seeds, the smallest attainable two-sided exact sign-flip p-value is 0.0625; consequently no comparison survives the planned 16-test Holm correction. Bootstrap intervals are descriptive robustness summaries rather than substitutes for locked-test inference.",
        "",
        "## Component-level interpretation",
        "",
        "1. **Sequential value is supported for repair routing.** Myopic loses 7.8 Repair.SR points and adds 8.0 Stop points. Its Regret increase is uncertain, so the claim should focus on Repair.SR and Stop.",
        "2. **The declared risk inputs are not supported by this ablation.** Removing them does not increase UAR or reduce Repair.SR. Correlated risk proxies remain, but the paper cannot claim an independent benefit for these three coordinates from this experiment.",
        "3. **Training-time budget awareness is supported.** w/o-Budget loses 4.8 Repair.SR points, adds 5.0 Stop points, and requests Human after exhaustion 6.6 times per run versus 2.4 for SHARE, despite identical executed caps.",
        "4. **The original dual formulation collapses conservatively.** It reaches Stop=0.608, Repair.SR=0.348, and Regret=1.605. UAR falls to 0.067 primarily because it stops rather than repairs. Across seeds beta_U remains zero while the AI calibration penalty is strongly positive, locating the collapse in unsafe-action calibration rather than dual ascent alone.",
        "",
        "## Evidence boundary",
        "",
        "This is synthetic, already-inspected development evidence. It justifies freezing the selected SHARE formulation and carrying both the positive sequential/budget findings and the negative risk-feature result into the paper. It is not independent locked-holdout evidence and must not be used for further tuning.",
        "",
    ])
    (args.results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.results / 'report.md'), "audit_rows": len(audit_rows)}))


if __name__ == "__main__":
    main()
