from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_triplet(pilot: Path, manifest_path: Path) -> dict[str, Any]:
    triplet = read_json(manifest_path)
    instance_id = triplet.get("instance_id")
    if triplet.get("kind") != "share_swebench_executed_branch_triplet_manifest" or not isinstance(instance_id, str):
        raise ValueError(f"invalid triplet manifest: {manifest_path}")
    row: dict[str, Any] = {"instance_id": instance_id, "pilot": str(pilot), "human_time_observed": triplet.get("human_time_observed") is True}
    for branch in ("human", "stop", "ai"):
        binding = triplet.get("branches", {}).get(branch)
        if not isinstance(binding, dict):
            raise ValueError(f"triplet lacks {branch} binding: {manifest_path}")
        artifact_path = pilot / str(binding.get("artifact_file"))
        if not artifact_path.is_file() or sha256_file(artifact_path) != binding.get("artifact_sha256"):
            raise ValueError(f"triplet {branch} artifact digest mismatch: {manifest_path}")
        artifact = read_json(artifact_path)
        if artifact.get("instance_id") != instance_id or artifact.get("branch") != branch:
            raise ValueError(f"triplet {branch} artifact identity mismatch: {manifest_path}")
        for raw in artifact.get("raw_files", {}).values():
            raw_path = artifact_path.parent / str(raw.get("file"))
            if not raw_path.is_file() or sha256_file(raw_path) != raw.get("sha256"):
                raise ValueError(f"triplet raw artifact digest mismatch: {raw_path}")
        row[branch] = bool(artifact.get("resolved"))
        if branch == "ai":
            row["ai_model"] = artifact.get("model")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and summarize one or more executed SWE-bench pilot roots.")
    parser.add_argument("--pilot", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    exclusions = []
    try:
        for pilot in args.pilot:
            rows.extend(verify_triplet(pilot, path) for path in sorted(pilot.glob("observed_branches/*/triplet_manifest.json")))
            for path in sorted(pilot.glob("exclusions/*.json")):
                exclusion = read_json(path)
                exclusions.append({"instance_id": exclusion.get("instance_id"), "reason": exclusion.get("reason"), "file": str(path), "sha256": sha256_file(path)})
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if not rows:
        raise SystemExit("no valid triplet manifests found")
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise SystemExit("duplicate instance ID across pilot roots")
    branch_counts = {branch: dict(sorted(Counter(str(row[branch]).lower() for row in rows).items())) for branch in ("human", "stop", "ai")}
    ai_models = Counter(str(row.get("ai_model")) for row in rows)
    ai_values = {row["ai"] for row in rows}
    report = {
        "schema_version": 1,
        "kind": "share_swebench_pilot_acceptance_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_roots": [str(path) for path in args.pilot],
        "valid_triplets": len(rows),
        "excluded_instances": exclusions,
        "all_artifact_and_raw_hashes_verified": True,
        "branch_outcome_counts": branch_counts,
        "ai_model_counts": dict(sorted(ai_models.items())),
        "ai_outcome_has_variation": len(ai_values) > 1,
        "human_time_observed_count": sum(row["human_time_observed"] for row in rows),
        "ready_for_repair_success_pilot_analysis": True,
        "ready_for_human_time_claims": all(row["human_time_observed"] for row in rows),
        "ready_for_policy_training": len(ai_values) > 1 and len(rows) >= 30,
        "next_gate": "collect_same_task_additional_ai_backbones" if len(ai_values) == 1 else "expand_group_disjoint_task_sample",
        "rows": rows,
        "warning": "Historical Human execution time is unavailable; this report does not create RepairEpisode records or authorize human-time claims.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "valid_triplets": len(rows), "ai_variation": report["ai_outcome_has_variation"], "next_gate": report["next_gate"]}, indent=2))


if __name__ == "__main__":
    main()
