"""Check whether the proposed 250-group component confirmation can run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_share_locked_holdout_replayable as base
from scripts.build_share_public_api_workflow_input import iter_operations, workflow_record


HISTORICAL = [
    ROOT / f"datasets/{name}/episodes.jsonl"
    for name in (
        "share_synthetic",
        "share_guard_development",
        "share_synthetic_augmented_guard_development",
        "share_locked_holdout_replayable_20260826",
        "share_v2_confirmation_holdout_20260827",
        "share_v3_final_holdout_20260827",
        "share_v4_ensemble_holdout_20260829",
        "share_v5_expected_voh_holdout_20260829",
    )
]
SOURCES = [
    ROOT / "datasets/share_public_api_workflow_input.jsonl",
    *(
        ROOT / f"datasets/{name}/source_input.jsonl"
        for name in (
            "share_v2_confirmation_holdout_20260827",
            "share_v3_final_holdout_20260827",
            "share_v4_ensemble_holdout_20260829",
            "share_v5_expected_voh_holdout_20260829",
        )
    ),
]
OUT = ROOT / "results/share_component_confirmation_readiness_20260930.json"


def main() -> None:
    historical, _ = base.historical_identifiers(
        HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json"
    )
    blocked = {str(row["group_id"]) for path in SOURCES for row in base.read_jsonl(path)}
    eligible_groups: set[str] = set()
    for path, document, operation in iter_operations(ROOT / "dataset/api"):
        row = workflow_record(path, document, operation)
        if row is None or str(row["group_id"]) in blocked:
            continue
        features = base.source_features(row)
        ids = base.candidate_identifiers(row, features)
        if not base.overlaps(ids, historical):
            eligible_groups.add(ids["group_id"])
    group_digest = hashlib.sha256("\n".join(sorted(eligible_groups)).encode()).hexdigest()
    result = {
        "target_new_workflow_groups": 250,
        "eligible_new_groups_in_current_public_api_corpus": len(eligible_groups),
        "shortfall": max(0, 250 - len(eligible_groups)),
        "eligible_group_ids_sha256": group_digest,
        "historical_episode_files": [str(path.relative_to(ROOT)) for path in HISTORICAL],
        "blocked_source_files": [str(path.relative_to(ROOT)) for path in SOURCES],
        "e2_protocol_manifest_found": bool(list((ROOT / "results").glob("**/matched_e2*manifest*.json"))),
        "component_confirmation_dataset_found": bool(
            list((ROOT / "datasets").glob("*component*confirmation*/episodes.jsonl"))
        ),
        "can_run_requested_250_group_table": len(eligible_groups) >= 250,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
