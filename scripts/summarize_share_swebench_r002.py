#!/usr/bin/env python3
"""Verify and summarize the frozen fixed-attempt R002 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


ALLOWED_STATUSES = {"resolved", "unresolved", "generation_error", "empty_patch", "patch_not_applied"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half_width, center + half_width


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    if a_only < 0 or b_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def verify_raw_files(artifact_path: Path, artifact: Mapping[str, Any]) -> None:
    raw_files = artifact.get("raw_files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise ValueError(f"archive lacks raw-file bindings: {artifact_path}")
    for name, binding in raw_files.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid raw-file binding {name}: {artifact_path}")
        raw_path = artifact_path.parent / str(binding.get("file"))
        if not raw_path.is_file():
            raise ValueError(f"missing raw artifact: {raw_path}")
        if sha256_file(raw_path) != binding.get("sha256"):
            raise ValueError(f"raw artifact digest mismatch: {raw_path}")
        if binding.get("bytes") != raw_path.stat().st_size:
            raise ValueError(f"raw artifact byte count mismatch: {raw_path}")


def summarize(pilot: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("kind") != "share_swebench_r002_controlled_protocol":
        raise ValueError("not an R002 controlled protocol manifest")
    protocol_id = protocol.get("protocol_id")
    bindings = protocol.get("bindings")
    if not isinstance(protocol_id, str) or not isinstance(bindings, Mapping):
        raise ValueError("protocol identity or bindings are missing")
    task_rows = bindings.get("tasks")
    profile_rows = bindings.get("profiles")
    if not isinstance(task_rows, list) or not isinstance(profile_rows, list):
        raise ValueError("protocol tasks or profiles are missing")
    expected_tasks = {row["instance_id"]: row for row in task_rows}
    expected_profiles = {row["name"]: row for row in profile_rows}
    if len(expected_tasks) != len(task_rows) or len(expected_profiles) != len(profile_rows):
        raise ValueError("duplicate task or profile in protocol")

    archive_root = pilot / "observed_r002_ai"
    artifacts = sorted(archive_root.rglob("artifact.json"))
    expected_branches = len(expected_tasks) * len(expected_profiles)
    if len(artifacts) != expected_branches:
        raise ValueError(f"expected {expected_branches} archives, found {len(artifacts)}")

    branches: dict[tuple[str, str], dict[str, Any]] = {}
    latest_archive = None
    for artifact_path in artifacts:
        artifact = read_json(artifact_path)
        instance_id = artifact.get("instance_id")
        profile = artifact.get("profile")
        key = (instance_id, profile)
        if instance_id not in expected_tasks or profile not in expected_profiles:
            raise ValueError(f"archive is outside frozen protocol: {artifact_path}")
        if key in branches:
            raise ValueError(f"duplicate archive for {instance_id}/{profile}")
        expected = {
            "schema_version": 1,
            "kind": "share_swebench_r002_fixed_attempt_branch",
            "protocol_id": protocol_id,
            "model": expected_profiles[profile]["model"],
            "analysis_rule": "fixed_single_attempt_no_retry_substitution",
        }
        for field, value in expected.items():
            if artifact.get(field) != value:
                raise ValueError(f"invalid {field} in {artifact_path}: expected {value!r}")
        status = artifact.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid branch status in {artifact_path}: {status!r}")
        if (artifact.get("resolved") is True) != (status == "resolved"):
            raise ValueError(f"resolved/status mismatch: {artifact_path}")
        turns = artifact.get("turns_used")
        if not isinstance(turns, int) or turns < 1:
            raise ValueError(f"invalid turns_used: {artifact_path}")
        usage = artifact.get("usage")
        if not isinstance(usage, list) or len(usage) != turns:
            raise ValueError(f"usage/turn count mismatch: {artifact_path}")
        verify_raw_files(artifact_path, artifact)
        prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage if isinstance(row, Mapping))
        completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage if isinstance(row, Mapping))
        branches[key] = {
            "instance_id": instance_id,
            "profile": profile,
            "model": artifact["model"],
            "status": status,
            "resolved": status == "resolved",
            "turns_used": turns,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "artifact_file": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
        }
        archived_at = artifact.get("archived_at_utc")
        if isinstance(archived_at, str) and (latest_archive is None or archived_at > latest_archive):
            latest_archive = archived_at

    profile_order = [row["name"] for row in profile_rows]
    rows = []
    for instance_id in expected_tasks:
        outcomes = {profile: branches[(instance_id, profile)] for profile in profile_order}
        rows.append({"instance_id": instance_id, "profiles": outcomes})

    profile_counts = {}
    for profile in profile_order:
        values = [branches[(instance_id, profile)] for instance_id in expected_tasks]
        counts = Counter(row["status"] for row in values)
        resolved = counts["resolved"]
        total = len(values)
        ci_low, ci_high = wilson_interval(resolved, total)
        profile_counts[profile] = {
            "model": expected_profiles[profile]["model"],
            "tasks": total,
            "resolved": resolved,
            "unsuccessful_fixed_attempts": total - resolved,
            "evaluated_unresolved": counts["unresolved"],
            "generation_error": counts["generation_error"],
            "empty_patch": counts["empty_patch"],
            "patch_not_applied": counts["patch_not_applied"],
            "repair_success_rate": resolved / total,
            "repair_success_wilson_95": [ci_low, ci_high],
            "mean_turns": mean(row["turns_used"] for row in values),
            "prompt_tokens": sum(row["prompt_tokens"] for row in values),
            "completion_tokens": sum(row["completion_tokens"] for row in values),
        }

    comparisons = []
    for index, profile_a in enumerate(profile_order):
        for profile_b in profile_order[index + 1 :]:
            outcomes_a = [branches[(instance_id, profile_a)]["resolved"] for instance_id in expected_tasks]
            outcomes_b = [branches[(instance_id, profile_b)]["resolved"] for instance_id in expected_tasks]
            a_only = sum(a and not b for a, b in zip(outcomes_a, outcomes_b))
            b_only = sum(b and not a for a, b in zip(outcomes_a, outcomes_b))
            both = sum(a and b for a, b in zip(outcomes_a, outcomes_b))
            neither = sum(not a and not b for a, b in zip(outcomes_a, outcomes_b))
            comparisons.append(
                {
                    "profile_a": profile_a,
                    "model_a": expected_profiles[profile_a]["model"],
                    "profile_b": profile_b,
                    "model_b": expected_profiles[profile_b]["model"],
                    "resolved_rate_difference_a_minus_b": (sum(outcomes_a) - sum(outcomes_b)) / len(outcomes_a),
                    "both_resolved": both,
                    "a_only_resolved": a_only,
                    "b_only_resolved": b_only,
                    "neither_resolved": neither,
                    "discordant_pairs": a_only + b_only,
                    "exact_mcnemar_two_sided_p": exact_mcnemar_pvalue(a_only, b_only),
                }
            )

    status_counts = Counter(branch["status"] for branch in branches.values())
    return {
        "schema_version": 1,
        "kind": "share_swebench_r002_acceptance_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol_id,
        "protocol_file": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "pilot_root": str(pilot),
        "tasks": len(expected_tasks),
        "repositories": len(expected_tasks),
        "profiles": len(expected_profiles),
        "fixed_attempt_branches": len(branches),
        "latest_archived_at_utc": latest_archive,
        "all_expected_archives_present": True,
        "all_artifact_and_raw_hashes_verified": True,
        "protocol_identity_and_model_bindings_verified": True,
        "analysis_rule": protocol.get("analysis_rule"),
        "overall_status_counts": dict(sorted(status_counts.items())),
        "profile_counts": profile_counts,
        "paired_comparisons": comparisons,
        "rows": rows,
        "ready_for_protocol_uniform_descriptive_ai_comparison": True,
        "ready_for_large_scale_backbone_ranking": False,
        "ready_for_human_time_claims": False,
        "ready_for_policy_training": False,
        "evidence_boundary": protocol.get("evidence_boundary"),
        "warning": (
            "This is an 11-task, one-fixed-attempt-per-model comparison on historical SWE-bench tasks. "
            "Generation errors and other unusable attempts are counted as unresolved. The paired exact tests are low-powered, "
            "the task sample is selected rather than population-representative, and historical gold patches are not observed "
            "contemporaneous human repairs or human-time measurements."
        ),
    }


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    profiles = list(report["profile_counts"])
    fields = ["instance_id"]
    for profile in profiles:
        fields.extend([f"{profile}_model", f"{profile}_status", f"{profile}_resolved", f"{profile}_turns"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in report["rows"]:
            row = {"instance_id": task["instance_id"]}
            for profile in profiles:
                branch = task["profiles"][profile]
                row.update(
                    {
                        f"{profile}_model": branch["model"],
                        f"{profile}_status": branch["status"],
                        f"{profile}_resolved": int(branch["resolved"]),
                        f"{profile}_turns": branch["turns_used"],
                    }
                )
            writer.writerow(row)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_latex(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"Model & Resolved & Eval. unres. & Gen. error & Tasks & Repair.SR [95\% CI] \\",
        r"\hline",
    ]
    for profile in report["profile_counts"]:
        row = report["profile_counts"][profile]
        low, high = row["repair_success_wilson_95"]
        lines.append(
            rf"\texttt{{{latex_escape(row['model'])}}} & {row['resolved']} & {row['evaluated_unresolved']} & "
            rf"{row['generation_error']} & {row['tasks']} & {row['repair_success_rate']:.3f} [{low:.3f}, {high:.3f}] \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and summarize the frozen R002 SWE-bench comparison.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--tex-out", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.pilot, args.protocol)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.csv_out:
            write_csv(report, args.csv_out)
        if args.tex_out:
            write_latex(report, args.tex_out)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "out": str(args.out),
                "tasks": report["tasks"],
                "branches": report["fixed_attempt_branches"],
                "profile_counts": report["profile_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
