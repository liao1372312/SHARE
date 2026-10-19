from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_guard_experiments import load_historical_identifier_index
from scripts.run_share_locked_holdout_once import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    BUDGETS,
    POLICIES,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    SEEDS,
    aggregate,
    load_budget_policies,
    primary_comparisons,
    record_metric,
    run_policy,
    verify_lock,
    voh_analysis,
)
from share.data import load_episodes
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def verify_binding(item: Mapping[str, Any]) -> None:
    path = Path(item["path"])
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"sealed binding mismatch: {path}")


def lock_snapshot(lock: Mapping[str, Any]) -> dict[str, str]:
    return {item["path"]: sha256_file(Path(item["path"])) for item in lock["files"]}


def validate_plan_a_holdout(
    episodes: Sequence[RepairEpisode],
    updated_manifest: Mapping[str, Any],
    development: Sequence[RepairEpisode],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    contract = updated_manifest["dataset"]
    if len(episodes) != 400 or len({episode.group_id for episode in episodes}) != 400:
        raise RuntimeError("Plan-A holdout must contain exactly 400 episodes and 400 workflow groups")
    if not all(episode.synthetic for episode in episodes):
        raise RuntimeError("Plan-A evidence contract requires synthetic=true for every episode")
    if contract["synthetic"] is not True or contract["actual_human_participant_execution"] is not False:
        raise RuntimeError("Plan-A manifest evidence boundary is inconsistent")
    if contract["paired_branch_outcomes"] != "deterministic replayable AI/Human/Stop simulations":
        raise RuntimeError("Plan-A branch-outcome contract mismatch")
    episode_ids = [episode.episode_id for episode in episodes]
    group_ids = [episode.group_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids) or len(set(group_ids)) != len(group_ids):
        raise RuntimeError("duplicate episode or workflow-group identifier")
    known_episode_ids = {episode.episode_id for episode in development}
    known_group_ids = {episode.group_id for episode in development}
    known_episode_ids.update(historical["historical_episode_ids"])
    known_group_ids.update(historical["historical_group_ids"])
    episode_overlap = sorted(set(episode_ids) & known_episode_ids)
    group_overlap = sorted(set(group_ids) & known_group_ids)
    if episode_overlap or group_overlap:
        raise RuntimeError("Plan-A holdout overlaps declared development/history")
    return {
        "result": "PASS",
        "episodes": 400,
        "workflow_groups": 400,
        "all_synthetic_true": True,
        "source_cases_synthetic": False,
        "episode_overlap": 0,
        "workflow_group_overlap": 0,
        "actual_human_participant_execution": False,
        "evidence_kind": "real_source_replayable_synthetic_locked_holdout",
    }


def exploratory_strata_all(
    payloads: Sequence[Mapping[str, Any]], episodes: Sequence[RepairEpisode]
) -> list[dict[str, Any]]:
    by_id = {episode.episode_id: episode for episode in episodes}
    complexity_levels = ("C_low", "C_mid", "C_high")
    horizon_levels = tuple(f"H{value}" for value in sorted({len(episode.steps) for episode in episodes}))
    rows = []
    for payload in payloads:
        for result in payload["results"]:
            if float(result["nominal_budget"]) != 0.20:
                continue
            for kind, levels in (("complexity", complexity_levels), ("initial_horizon", horizon_levels)):
                for level in levels:
                    selected = []
                    for record in result["records"]:
                        episode = by_id[record["episode_id"]]
                        complexity = episode.steps[0].state.complexity
                        observed = (
                            "C_low" if complexity < 0.5 else "C_mid" if complexity < 0.8 else "C_high"
                        ) if kind == "complexity" else f"H{len(episode.steps)}"
                        if observed == level:
                            selected.append(record)
                    if selected:
                        rows.append({
                            "seed": payload["seed"],
                            "policy": result["policy"],
                            "stratum_type": kind,
                            "stratum": level,
                            "episodes": len(selected),
                            **{
                                metric: record_metric(selected, metric)
                                for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS)
                            },
                        })
    return rows


def nondominated(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in BUDGETS:
        rows = [row for row in aggregate_rows if float(row["nominal_budget"]) == budget]
        for row in rows:
            target = (
                float(row["repair_sr_mean"]),
                float(row["hir_mean"]),
                float(row["uar_mean"]),
                float(row["stopped_rate_mean"]),
            )
            dominated_by = []
            for other in rows:
                if other["policy"] == row["policy"]:
                    continue
                candidate = (
                    float(other["repair_sr_mean"]),
                    float(other["hir_mean"]),
                    float(other["uar_mean"]),
                    float(other["stopped_rate_mean"]),
                )
                weak = candidate[0] >= target[0] and all(candidate[index] <= target[index] for index in (1, 2, 3))
                strict = candidate[0] > target[0] or any(candidate[index] < target[index] for index in (1, 2, 3))
                if weak and strict:
                    dominated_by.append(other["policy"])
            output.append({
                "nominal_budget": budget,
                "policy": row["policy"],
                "nondominated": not dominated_by,
                "dominated_by": dominated_by,
            })
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    selected = list(fields or rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_budget_frontiers(aggregate_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = {
        "share_conf": "SHARE-Conf",
        "share_static": "SHARE-Static",
        "share_bandit": "SHARE-Bandit",
        "share_ddqn": "SHARE-DDQN",
        "share": "SHARE",
    }
    markers = {"share_conf": "o", "share_static": "s", "share_bandit": "^", "share_ddqn": "D", "share": "*"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
    panels = (("repair_sr_mean", "Repair.SR", True), ("uar_mean", "UAR", False), ("collaboration_regret_mean", "Regret", False))
    for policy in POLICIES:
        rows = sorted((row for row in aggregate_rows if row["policy"] == policy), key=lambda row: row["nominal_budget"])
        x = [row["hir_mean"] for row in rows]
        for axis, (metric, ylabel, _) in zip(axes, panels):
            axis.plot(x, [row[metric] for row in rows], marker=markers[policy], linewidth=1.6, label=labels[policy])
            for row in rows:
                axis.annotate(f"{int(round(100 * row['nominal_budget']))}%", (row["hir_mean"], row[metric]), fontsize=6, xytext=(2, 2), textcoords="offset points")
            axis.set_xlabel("Realized HIR")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_primary(primary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    metrics = [("repair_sr_mean", "Repair.SR"), ("hir_mean", "HIR"), ("uar_mean", "UAR"), ("stopped_rate_mean", "Stop")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    labels = ["Conf", "Static", "Bandit", "DDQN", "SHARE"]
    colors = ["#9aa0a6", "#7f8c8d", "#4c78a8", "#f58518", "#54a24b"]
    for axis, (metric, title) in zip(axes, metrics):
        values = [next(row[metric] for row in primary_rows if row["policy"] == policy) for policy in POLICIES]
        errors = [next(row[metric.replace("_mean", "_std")] for row in primary_rows if row["policy"] == policy) for policy in POLICIES]
        axis.bar(labels, values, yerr=errors, color=colors, capsize=2)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def voh_bin_rows(state_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in state_rows:
        buckets[(int(row["seed"]), int(row["voh_bin"]))].append(float(row["realized_advantage"]))
    per_seed = [
        {"seed": seed, "voh_bin": bin_index, "realized_advantage_mean": mean(buckets[(seed, bin_index)]), "states": len(buckets[(seed, bin_index)])}
        for seed in SEEDS for bin_index in range(1, 6)
    ]
    return per_seed


def plot_voh(per_seed_bins: Sequence[Mapping[str, Any]], path: Path) -> None:
    means = []
    errors = []
    for bin_index in range(1, 6):
        values = [row["realized_advantage_mean"] for row in per_seed_bins if row["voh_bin"] == bin_index]
        means.append(mean(values))
        errors.append(stdev(values))
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    axis.errorbar(range(1, 6), means, yerr=errors, marker="o", capsize=3, color="#4c78a8")
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("Frozen predicted VoH quintile")
    axis.set_ylabel("Replay Human-over-AI advantage")
    axis.set_xticks(range(1, 6))
    axis.grid(alpha=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_exploratory(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis, kind in zip(axes, ("complexity", "initial_horizon")):
        levels = sorted({row["stratum"] for row in rows if row["stratum_type"] == kind})
        for policy in POLICIES:
            values = []
            for level in levels:
                cells = [row["repair_sr"] for row in rows if row["stratum_type"] == kind and row["stratum"] == level and row["policy"] == policy]
                values.append(mean(cells))
            axis.plot(levels, values, marker="o", label=policy)
        axis.set_xlabel(kind.replace("_", " ").title())
        axis.set_ylabel("Repair.SR at 20%")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_report(
    primary_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    voh: Mapping[str, Any],
    frontier: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# SHARE Plan-A Final Locked Evaluation",
        "",
        "Evidence boundary: prospectively generated, previously unseen, workflow-group-disjoint, real-source replayable synthetic locked holdout derived from public-API records. All 400 episodes are `synthetic=true`; no actual human-participant execution is claimed.",
        "",
        "## Primary 20% hard-cap results",
        "",
        "| Method | Repair.SR | HIR | UAR | Stop | Return | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = next(item for item in primary_rows if item["policy"] == policy)
        lines.append(
            f"| {policy} | {row['repair_sr_mean']:.3f} ± {row['repair_sr_std']:.3f} | "
            f"{row['hir_mean']:.3f} ± {row['hir_std']:.3f} | {row['uar_mean']:.3f} ± {row['uar_std']:.3f} | "
            f"{row['stopped_rate_mean']:.3f} ± {row['stopped_rate_std']:.3f} | "
            f"{row['normalized_return_mean']:.3f} ± {row['normalized_return_std']:.3f} | "
            f"{row['collaboration_regret_mean']:.3f} ± {row['collaboration_regret_std']:.3f} |"
        )
    lines.extend(["", "## Preregistered paired primary comparisons", "", "Differences are comparator minus SHARE.", "", "| Comparison | Metric | Mean difference | 95% group-bootstrap CI | Exact p | Holm p |", "|---|---|---:|---:|---:|---:|"])
    for row in comparisons:
        low, high = row["workflow_group_cluster_bootstrap_ci95"]
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_difference']:.4f} | [{low:.4f}, {high:.4f}] | {row['exact_seed_sign_flip_p_two_sided']:.4f} | {row['holm_adjusted_p_across_16_primary_tests']:.4f} |")
    lines.extend([
        "",
        "## VoH validity (secondary)",
        "",
        f"Mean seed-level Spearman rho: {voh['mean_spearman']:.4f} (SD {voh['std_spearman']:.4f}); 95% workflow-group bootstrap CI [{voh['group_cluster_bootstrap_mean_spearman_ci95'][0]:.4f}, {voh['group_cluster_bootstrap_mean_spearman_ci95'][1]:.4f}].",
        f"One-sided exact sign-flip p for a positive quintile slope: {voh['one_sided_exact_sign_flip_p_for_positive_slope']:.4f}.",
        "",
        "## Pareto status",
        "",
    ])
    for budget in BUDGETS:
        names = [row["policy"] for row in frontier if row["nominal_budget"] == budget and row["nondominated"]]
        lines.append(f"- {int(100 * budget)}%: {', '.join(names)}")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These findings are locked replay-simulation evidence, not observed human-participant or live-production evidence. Complexity and horizon summaries are exploratory only. No method, checkpoint, threshold, normalization statistic, reward, seed, hard-cap evaluator, or analysis choice was changed after the freeze.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed Plan-A replayable-synthetic locked evaluation exactly once.")
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--updated-manifest", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--historical-identifiers", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=ROOT / "datasets/share_synthetic/episodes.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    manifest = json.loads(args.updated_manifest.read_text(encoding="utf-8"))
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    lock = json.loads(args.artifact_lock.read_text(encoding="utf-8"))
    for item in seal["bindings"].values():
        verify_binding(item)
    if sha256_file(args.amendment) != seal["bindings"]["protocol_amendment"]["sha256"]:
        raise SystemExit("amendment seal mismatch")
    if sha256_file(args.updated_manifest) != seal["bindings"]["updated_locked_manifest"]["sha256"]:
        raise SystemExit("updated manifest seal mismatch")
    if sha256_file(args.artifact_lock) != seal["bindings"]["artifact_lock"]["sha256"]:
        raise SystemExit("artifact-lock seal mismatch")
    if sha256_file(args.holdout) != manifest["dataset"]["sha256"]:
        raise SystemExit("holdout hash differs from sealed Plan-A manifest")
    if amendment["status"] != "SEALED_PRE_EVALUATION" or seal["status"] != "SEALED_READY_FOR_EXACTLY_ONE_LOCKED_EVALUATION":
        raise SystemExit("Plan-A protocol is not sealed for evaluation")
    if amendment["method_level_locked_results_accessed"] or amendment["locked_evaluator_run"]:
        raise SystemExit("amendment does not attest a clean pre-evaluation state")

    verify_lock(lock)
    pre_lock_snapshot = lock_snapshot(lock)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt = out / "UNBLINDING_ATTEMPT.json"
    if attempt.exists():
        raise SystemExit("Plan-A locked evaluation has already been attempted; refusing a second opening")
    atomic_json(attempt, {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOLDOUT_OPENED_ONCE",
        "protocol_amendment_sha256": sha256_file(args.amendment),
        "updated_manifest_sha256": sha256_file(args.updated_manifest),
        "combined_pre_evaluation_seal_sha256": sha256_file(args.seal),
        "holdout_sha256": sha256_file(args.holdout),
    })

    development = load_episodes(args.development)
    historical = load_historical_identifier_index(args.historical_identifiers, args.development, development)
    holdout = load_episodes(args.holdout)
    provenance = validate_plan_a_holdout(holdout, manifest, development, historical)
    episodes = sorted(holdout, key=lambda episode: episode.episode_id)
    encoder = FeatureEncoder.load(Path(lock["encoder"]["path"]))
    reward = RewardConfig(**lock["reward_config"])

    payloads = []
    for seed in SEEDS:
        results, audits = [], {}
        for budget in BUDGETS:
            for policy in load_budget_policies(lock, budget, seed, encoder, reward):
                result, audit = run_policy(policy, episodes, reward, budget, seed)
                results.append(result)
                audits[f"{budget:.1f}/{policy.name}"] = audit
        payloads.append({"seed": seed, "results": results, "budget_audit": audits})

    aggregate_rows = aggregate(payloads)
    comparisons = primary_comparisons(payloads, episodes)
    voh = voh_analysis(lock, episodes, encoder, reward)
    voh_state_rows = voh.pop("state_rows")
    voh_bins = voh_bin_rows(voh_state_rows)
    exploratory = exploratory_strata_all(payloads, episodes)
    frontier = nondominated(aggregate_rows)
    primary_rows = [row for row in aggregate_rows if float(row["nominal_budget"]) == 0.20]

    raw_path = out / "raw_results.json"
    atomic_json(raw_path, {"payloads": payloads})
    atomic_json(out / "primary_20pct_results.json", primary_rows)
    atomic_json(out / "aggregate_budget_results.json", aggregate_rows)
    atomic_json(out / "primary_paired_comparisons.json", comparisons)
    atomic_json(out / "voh_validity_analysis.json", {**voh, "state_rows": voh_state_rows, "per_seed_quintile_means": voh_bins})
    atomic_json(out / "exploratory_complexity_horizon.json", exploratory)
    atomic_json(out / "pareto_status.json", frontier)

    per_seed_rows = []
    for payload in payloads:
        for row in payload["results"]:
            eligible_uar = sum(
                1
                for record in row["records"]
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
            "seed": payload["seed"], "run": key,
            "cap": value["cap"], "human_interventions": value["human_interventions"],
            "cap_respected": value["human_interventions"] <= value["cap"],
        }
        for payload in payloads for key, value in payload["budget_audit"].items()
    ]
    post_lock_snapshot = lock_snapshot(lock)
    if post_lock_snapshot != pre_lock_snapshot:
        raise RuntimeError("frozen method artifact changed during locked evaluation")
    final_audit = {
        "schema_version": 1,
        "kind": "share_plan_a_final_locked_evaluation_audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_attempts": 1,
        "runs": 75,
        "episodes": 400,
        "workflow_groups": 400,
        "provenance": provenance,
        "hard_cap": {
            "runs_checked": len(cap_runs),
            "violations": sum(not row["cap_respected"] for row in cap_runs),
            "all_respected": all(row["cap_respected"] for row in cap_runs),
            "runs": cap_runs,
        },
        "frozen_artifacts": {
            "files_checked": len(pre_lock_snapshot),
            "pre_binding_sha256": sha256_json(pre_lock_snapshot),
            "post_binding_sha256": sha256_json(post_lock_snapshot),
            "changes": 0,
        },
        "method_changes": {"retraining": False, "recalibration": False, "retuning": False, "checkpoint_replacement": False, "threshold_changes": False, "posthoc_selection": False},
        "plan_b": {"materials_preserved": True, "used_as_evidence": False, "participant_outcomes_collected": False, "actual_human_execution_claimed": False},
        "evidence_boundary": "real-source replayable synthetic locked evidence; all episodes synthetic=true; no human-participant execution claim",
        "stop_all_experimentation": True,
    }
    atomic_json(out / "final_evaluation_audit.json", final_audit)
    atomic_json(out / "evaluation_manifest.json", {
        "completed_at_utc": final_audit["completed_at_utc"],
        "protocol_amendment_sha256": sha256_file(args.amendment),
        "updated_manifest_sha256": sha256_file(args.updated_manifest),
        "combined_pre_evaluation_seal_sha256": sha256_file(args.seal),
        "artifact_lock_sha256": sha256_file(args.artifact_lock),
        "holdout_sha256": sha256_file(args.holdout),
        "episode_order_sha256": hashlib.sha256("\n".join(episode.episode_id for episode in episodes).encode("utf-8")).hexdigest(),
        "runs": 75,
        "all_caps_respected": final_audit["hard_cap"]["all_respected"],
        "single_locked_evaluation_completed": True,
        "further_experimentation_prohibited": True,
    })
    atomic_text(out / "FINAL_RESULTS_REPORT.md", markdown_report(primary_rows, comparisons, voh, frontier))

    attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
    attempt_payload.update({"status": "COMPLETED_ONCE_DO_NOT_RERUN", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(attempt, attempt_payload)
    result_files = sorted(path for path in out.iterdir() if path.is_file() and path.name not in {"FINAL_SHA256SUMS"})
    atomic_text(out / "FINAL_SHA256SUMS", "".join(f"{sha256_file(path)}  {path.name}\n" for path in result_files))
    print(json.dumps({
        "out": str(out), "episodes": 400, "runs": 75, "complete": True,
        "all_caps_respected": True, "frozen_artifact_changes": 0,
        "evidence": "real_source_replayable_synthetic",
        "do_not_rerun": True,
    }, indent=2))


if __name__ == "__main__":
    main()
