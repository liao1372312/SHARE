from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_share_swebench_pilots import read_json, verify_triplet


EXPECTED_BACKBONE_PROFILES = ("backbone_2", "backbone_3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_files(artifact_path: Path, artifact: Mapping[str, Any]) -> None:
    raw_files = artifact.get("raw_files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise ValueError(f"execution artifact lacks raw-file bindings: {artifact_path}")
    for binding in raw_files.values():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid raw-file binding: {artifact_path}")
        raw_path = artifact_path.parent / str(binding.get("file"))
        if not raw_path.is_file() or sha256_file(raw_path) != binding.get("sha256"):
            raise ValueError(f"raw artifact digest mismatch: {raw_path}")


def transcript_action_counts(artifact_path: Path, artifact: Mapping[str, Any]) -> dict[str, int]:
    binding = artifact.get("raw_files", {}).get("generation_transcript.jsonl")
    if not isinstance(binding, Mapping):
        raise ValueError(f"execution artifact lacks generation transcript: {artifact_path}")
    transcript_path = artifact_path.parent / str(binding.get("file"))
    counts: Counter[str] = Counter()
    with transcript_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            action = value.get("response", {}).get("action") if isinstance(value, Mapping) else None
            if not isinstance(action, str):
                raise ValueError(f"transcript line {line_number} lacks an action: {transcript_path}")
            counts[action] += 1
    return dict(sorted(counts.items()))


def verify_backbone_archive(
    pilot: Path,
    instance_id: str,
    profile: str,
    expected_image_id: str,
) -> dict[str, Any]:
    artifact_path = pilot / "observed_ai_backbones" / instance_id / profile / "artifact.json"
    if not artifact_path.is_file():
        raise ValueError(f"missing {profile} execution artifact: {artifact_path}")
    artifact = read_json(artifact_path)
    expected = {
        "schema_version": 1,
        "kind": "share_swebench_ai_execution_artifact",
        "instance_id": instance_id,
        "branch": "ai",
        "profile": profile,
        "patch_successfully_applied": True,
        "evaluated_patch_exactly_matches_model_patch": True,
        "clean_tracked_generation_baseline_verified": True,
        "gold_patch_withheld_during_generation": True,
        "test_patch_withheld_during_generation": True,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"invalid {profile} {key} in {artifact_path}: expected {value!r}")
    model = artifact.get("model")
    run_id = artifact.get("run_id")
    image = artifact.get("image")
    tests = artifact.get("tests")
    if not isinstance(model, str) or not model or not isinstance(run_id, str) or not run_id:
        raise ValueError(f"invalid model or run ID: {artifact_path}")
    if not isinstance(image, Mapping) or image.get("id") != expected_image_id:
        raise ValueError(f"cross-backbone image differs from primary triplet: {artifact_path}")
    if not isinstance(tests, Mapping):
        raise ValueError(f"execution artifact lacks test counts: {artifact_path}")
    for key in ("fail_to_pass_success", "fail_to_pass_failure", "pass_to_pass_success", "pass_to_pass_failure"):
        if not isinstance(tests.get(key), int) or tests[key] < 0:
            raise ValueError(f"invalid test count {key}: {artifact_path}")
    verify_raw_files(artifact_path, artifact)
    return {
        "profile": profile,
        "model": model,
        "run_id": run_id,
        "resolved": artifact.get("resolved") is True,
        "artifact_file": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "tests": dict(tests),
        "transcript_action_counts": transcript_action_counts(artifact_path, artifact),
    }


def summarize(pilots: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pilot in pilots:
        for manifest_path in sorted(pilot.glob("observed_branches/*/triplet_manifest.json")):
            primary = verify_triplet(pilot, manifest_path)
            triplet = read_json(manifest_path)
            instance_id = primary["instance_id"]
            image_id = triplet.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"triplet lacks image identity: {manifest_path}")
            profiles = {
                "primary": {
                    "profile": "primary",
                    "model": primary.get("ai_model"),
                    "resolved": primary["ai"],
                    "artifact_file": str(pilot / triplet["branches"]["ai"]["artifact_file"]),
                    "artifact_sha256": triplet["branches"]["ai"]["artifact_sha256"],
                }
            }
            primary_artifact_path = Path(profiles["primary"]["artifact_file"])
            primary_artifact = read_json(primary_artifact_path)
            profiles["primary"]["transcript_action_counts"] = transcript_action_counts(primary_artifact_path, primary_artifact)
            for profile in EXPECTED_BACKBONE_PROFILES:
                profiles[profile] = verify_backbone_archive(pilot, instance_id, profile, image_id)
            rows.append(
                {
                    "instance_id": instance_id,
                    "pilot": str(pilot),
                    "human_resolved": primary["human"],
                    "stop_resolved": primary["stop"],
                    "human_time_observed": primary["human_time_observed"],
                    "ai_profiles": profiles,
                }
            )
    if not rows:
        raise ValueError("no valid executed triplets found")
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate instance ID across pilot roots")

    profile_counts: dict[str, dict[str, Any]] = {}
    for profile in ("primary", *EXPECTED_BACKBONE_PROFILES):
        values = [row["ai_profiles"][profile] for row in rows]
        model_counts = Counter(str(value["model"]) for value in values)
        if len(model_counts) != 1:
            raise ValueError(f"profile {profile} maps to multiple models: {dict(model_counts)}")
        resolved = sum(bool(value["resolved"]) for value in values)
        profile_counts[profile] = {
            "model": next(iter(model_counts)),
            "tasks": len(values),
            "resolved": resolved,
            "unresolved": len(values) - resolved,
            "repair_success_rate": resolved / len(values),
        }

    patterns = Counter(
        "".join("1" if row["ai_profiles"][profile]["resolved"] else "0" for profile in ("primary", *EXPECTED_BACKBONE_PROFILES))
        for row in rows
    )
    structured_replace_branches = sum(
        row["ai_profiles"][profile]["transcript_action_counts"].get("replace", 0) > 0
        for row in rows
        for profile in ("primary", *EXPECTED_BACKBONE_PROFILES)
    )
    return {
        "schema_version": 1,
        "kind": "share_swebench_cross_backbone_pilot_acceptance_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_roots": [str(path) for path in pilots],
        "valid_tasks": len(rows),
        "executed_ai_branches": len(rows) * 3,
        "additional_backbone_archives": len(rows) * len(EXPECTED_BACKBONE_PROFILES),
        "all_artifact_and_raw_hashes_verified": True,
        "same_task_and_image_binding_verified": True,
        "generation_protocol_hash_bound": False,
        "generation_protocol_uniform_verified": False,
        "branches_with_observed_structured_replace_action": structured_replace_branches,
        "profile_counts": profile_counts,
        "task_outcome_patterns_primary_backbone2_backbone3": dict(sorted(patterns.items())),
        "human_time_observed_count": sum(row["human_time_observed"] for row in rows),
        "ready_for_cross_backbone_pilot_descriptive_analysis": True,
        "ready_for_controlled_cross_backbone_performance_comparison": False,
        "ready_for_large_scale_cross_backbone_claims": False,
        "ready_for_human_time_claims": False,
        "ready_for_policy_training": False,
        "next_gate": "expand_group_disjoint_executed_task_sample_and_collect_observed_human_time",
        "rows": rows,
        "warning": (
            "This is a five-task executed SWE-bench pilot. It verifies same-task cross-backbone repair outcomes, "
            "but it is not the planned large-scale paired holdout, a practitioner study, or human-time evidence. "
            "The generation prompt/source revision was not bound into the artifacts and the scaffold changed during collection; "
            "model success rates are therefore descriptive outcomes, not a controlled backbone effect estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and summarize same-task SWE-bench cross-backbone pilot archives.")
    parser.add_argument("--pilot", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.pilot)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "valid_tasks": report["valid_tasks"],
                "executed_ai_branches": report["executed_ai_branches"],
                "next_gate": report["next_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
