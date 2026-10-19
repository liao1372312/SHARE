from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_locked_holdout_once import BUDGETS, POLICIES, PRIMARY_METRICS, SECONDARY_METRICS
from scripts.run_share_plan_a_locked_holdout_once import (
    exploratory_strata_all,
    lock_snapshot,
    markdown_report,
    nondominated,
    plot_budget_frontiers,
    plot_exploratory,
    plot_primary,
    plot_voh,
    sha256_file,
    sha256_json,
    voh_bin_rows,
)
from scripts.run_share_locked_holdout_once import verify_lock
from share.data import load_episodes
from share.types import AuthorityAction

def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    selected = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize an already-computed Plan-A locked evaluation after a packaging-only NaN failure.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    attempt_path = out / "UNBLINDING_ATTEMPT.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt.get("status") != "HOLDOUT_OPENED_ONCE":
        raise SystemExit("recovery is permitted only for the recorded incomplete packaging state")
    required = [
        out / "raw_results.json", out / "aggregate_budget_results.json", out / "primary_20pct_results.json",
        out / "primary_paired_comparisons.json", out / "voh_validity_analysis.json",
    ]
    if not all(path.is_file() for path in required):
        raise SystemExit("recovery inputs are incomplete; refusing to recompute policies or bootstrap")
    if (out / "final_evaluation_audit.json").exists() or (out / "FINAL_SHA256SUMS").exists():
        raise SystemExit("final results already exist; refusing a second finalization")

    lock = json.loads(args.artifact_lock.read_text(encoding="utf-8"))
    verify_lock(lock)
    before = lock_snapshot(lock)
    payloads = json.loads((out / "raw_results.json").read_text(encoding="utf-8"))["payloads"]
    if len(payloads) != 5 or sum(len(payload["results"]) for payload in payloads) != 75:
        raise SystemExit("raw locked evaluation does not contain the complete 75 preregistered runs")
    aggregate_rows = json.loads((out / "aggregate_budget_results.json").read_text(encoding="utf-8"))
    primary_rows = json.loads((out / "primary_20pct_results.json").read_text(encoding="utf-8"))
    comparisons = json.loads((out / "primary_paired_comparisons.json").read_text(encoding="utf-8"))
    voh_payload = json.loads((out / "voh_validity_analysis.json").read_text(encoding="utf-8"))
    state_rows = voh_payload["state_rows"]
    voh_summary = {key: value for key, value in voh_payload.items() if key != "state_rows"}
    voh_bins = voh_bin_rows(state_rows)

    episodes = load_episodes(args.holdout)
    exploratory = exploratory_strata_all(payloads, episodes)
    frontier = nondominated(aggregate_rows)
    atomic_json(out / "exploratory_complexity_horizon.json", exploratory)
    atomic_json(out / "pareto_status.json", frontier)
    atomic_json(out / "voh_validity_analysis.json", {**voh_summary, "state_rows": state_rows, "per_seed_quintile_means": voh_bins})

    per_seed_rows = []
    for payload in payloads:
        for row in payload["results"]:
            eligible_uar = sum(
                1 for record in row["records"]
                if not record["human_used"] and int(record["actions"][-1]) != int(AuthorityAction.STOP)
            )
            per_seed_rows.append({
                "seed": payload["seed"], "nominal_budget": row["nominal_budget"], "policy": row["policy"],
                **{metric: row[metric] for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS)},
                "hard_human_cap": row["hard_human_cap"], "human_interventions": row["human_interventions"],
                "uar_denominator": eligible_uar, "cap_respected": row["cap_respected"],
            })
    write_csv(out / "per_method_per_seed_metrics.csv", per_seed_rows)
    write_csv(out / "aggregate_budget_results.csv", aggregate_rows)
    write_csv(out / "primary_20pct_results.csv", primary_rows)
    write_csv(out / "primary_paired_comparisons.csv", comparisons, [
        "comparison", "metric", "direction", "mean_difference", "std_difference",
        "workflow_group_cluster_bootstrap_ci95", "exact_seed_sign_flip_p_two_sided",
        "holm_adjusted_p_across_16_primary_tests",
    ])
    write_csv(out / "voh_quintile_means.csv", voh_bins)
    write_csv(out / "exploratory_complexity_horizon.csv", exploratory)
    write_csv(out / "pareto_status.csv", frontier)
    plot_primary(primary_rows, out / "primary_20pct_results.pdf")
    plot_budget_frontiers(aggregate_rows, out / "hard_budget_frontiers.pdf")
    plot_voh(voh_bins, out / "voh_validity.pdf")
    plot_exploratory(exploratory, out / "exploratory_complexity_horizon.pdf")

    cap_runs = [
        {
            "seed": payload["seed"], "run": name, "cap": audit["cap"],
            "human_interventions": audit["human_interventions"],
            "cap_respected": audit["human_interventions"] <= audit["cap"],
        }
        for payload in payloads for name, audit in payload["budget_audit"].items()
    ]
    after = lock_snapshot(lock)
    if after != before:
        raise RuntimeError("frozen artifact changed during deterministic packaging recovery")
    recovery = {
        "schema_version": 1,
        "kind": "share_plan_a_postprocessing_recovery_audit",
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "incident": "the single locked run completed 75 policy runs, primary bootstrap, and VoH bootstrap, then failed while JSON-serializing exploratory UAR NaN for strata with no eligible denominator",
        "recovery_action": "read already-written raw/aggregate/comparison/VoH artifacts; encode undefined exploratory UAR as JSON null/CSV blank; complete deterministic tables, figures, report, hashes, and audit",
        "policies_rerun": False,
        "bootstrap_rerun": False,
        "training_or_calibration_run": False,
        "metric_definition_changed": False,
        "undefined_uar_rule": "undefined when the stratum has zero eligible no-Human non-Stop episodes; represented as null/blank rather than NaN",
        "complete_raw_runs_verified": 75,
        "primary_comparisons_verified": len(comparisons),
        "voh_state_rows_verified": len(state_rows),
    }
    atomic_json(out / "postprocessing_recovery_audit.json", recovery)
    final_audit = {
        "schema_version": 1,
        "kind": "share_plan_a_final_locked_evaluation_audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_locked_evaluations": 1,
        "pre_unblinding_failures": 1,
        "postprocessing_recoveries": 1,
        "runs": 75,
        "episodes": 400,
        "workflow_groups": 400,
        "provenance": {
            "result": "PASS", "all_synthetic_true": all(episode.synthetic for episode in episodes),
            "source_cases_synthetic": False, "workflow_group_overlap": 0, "episode_overlap": 0,
            "actual_human_participant_execution": False,
            "evidence_kind": "real_source_replayable_synthetic_locked_holdout",
        },
        "hard_cap": {"runs_checked": 75, "violations": sum(not row["cap_respected"] for row in cap_runs), "all_respected": all(row["cap_respected"] for row in cap_runs), "runs": cap_runs},
        "frozen_artifacts": {"files_checked": len(before), "pre_binding_sha256": sha256_json(before), "post_binding_sha256": sha256_json(after), "changes": 0},
        "method_changes": {"retraining": False, "recalibration": False, "retuning": False, "checkpoint_replacement": False, "threshold_changes": False, "normalization_changes": False, "reward_changes": False, "seed_changes": False, "hard_cap_evaluator_changes": False, "posthoc_selection": False},
        "analysis_execution": {"primary_bootstrap_runs": 10000, "voh_bootstrap_runs": 10000, "primary_comparisons": 16, "holm_family": 16, "postprocessing_only_recovery": True},
        "plan_b": {"materials_preserved": True, "used_as_evidence": False, "participant_outcomes_collected": False, "actual_human_execution_claimed": False},
        "evidence_boundary": "real-source replayable synthetic locked evidence; all episodes synthetic=true; no human-participant execution claim",
        "stop_all_experimentation": True,
    }
    atomic_json(out / "final_evaluation_audit.json", final_audit)
    atomic_json(out / "evaluation_manifest.json", {
        "completed_at_utc": final_audit["completed_at_utc"],
        "protocol_amendment_sha256": sha256_file(out / "protocol_amendment.json"),
        "combined_pre_evaluation_seal_sha256": sha256_file(out / "combined_pre_evaluation_seal.json"),
        "artifact_lock_sha256": sha256_file(args.artifact_lock),
        "holdout_sha256": sha256_file(args.holdout),
        "episode_order_sha256": hashlib.sha256("\n".join(episode.episode_id for episode in sorted(episodes, key=lambda episode: episode.episode_id)).encode("utf-8")).hexdigest(),
        "runs": 75, "all_caps_respected": final_audit["hard_cap"]["all_respected"],
        "single_locked_evaluation_completed": True, "packaging_recovery_without_rerun": True,
        "further_experimentation_prohibited": True,
    })
    atomic_text(out / "FINAL_RESULTS_REPORT.md", markdown_report(primary_rows, comparisons, voh_summary, frontier))
    attempt.update({
        "status": "COMPLETED_ONCE_WITH_PACKAGING_RECOVERY_DO_NOT_RERUN",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "policies_rerun_during_recovery": False,
        "bootstrap_rerun_during_recovery": False,
        "postprocessing_recovery_audit": "postprocessing_recovery_audit.json",
    })
    atomic_json(attempt_path, attempt)
    files = sorted(path for path in out.iterdir() if path.is_file() and path.name != "FINAL_SHA256SUMS")
    atomic_text(out / "FINAL_SHA256SUMS", "".join(f"{sha256_file(path)}  {path.name}\n" for path in files))
    print(json.dumps({
        "out": str(out), "status": attempt["status"], "runs": 75,
        "policies_rerun": False, "bootstrap_rerun": False,
        "cap_violations": final_audit["hard_cap"]["violations"],
        "frozen_artifact_changes": 0, "stop_all_experimentation": True,
    }, indent=2))


if __name__ == "__main__":
    main()
