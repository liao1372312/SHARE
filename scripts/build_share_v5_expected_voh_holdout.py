"""Build the final 150-group holdout for the frozen expected-VoH filter."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_share_locked_holdout_replayable as base
from scripts.build_share_public_api_workflow_input import iter_operations, workflow_record
from share.data import load_episodes, write_episodes

OUT = ROOT / "datasets/share_v5_expected_voh_holdout_20260829"
PROTOCOL = "share-v5-expected-voh-replay-150-v1"
SELECTION_SEED, FAILURE_SEED, BRANCH_SEED = 2026082911, 2026082912, 2026082913
SOURCE_FILES = (
    ROOT / "datasets/share_public_api_workflow_input.jsonl",
    ROOT / "datasets/share_v2_confirmation_holdout_20260827/source_input.jsonl",
    ROOT / "datasets/share_v3_final_holdout_20260827/source_input.jsonl",
    ROOT / "datasets/share_v4_ensemble_holdout_20260829/source_input.jsonl",
)
HISTORICAL = (
    ROOT / "datasets/share_synthetic/episodes.jsonl",
    ROOT / "datasets/share_guard_development/episodes.jsonl",
    ROOT / "datasets/share_synthetic_augmented_guard_development/episodes.jsonl",
    ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl",
    ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl",
    ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl",
    ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl",
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("refusing to overwrite V5 holdout")
    OUT.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    dump(OUT / "generation_preregistration.json", {
        "protocol": PROTOCOL, "created_at_utc": created,
        "target": {"episodes": 150, "groups": 150},
        "complexity_counts": {"low": 50, "medium": 50, "high": 50},
        "failure_counts": {name: 25 for name in (
            "planning_failure", "service_selection_failure", "parameter_binding_failure",
            "tool_execution_failure", "supervisor_failure", "answer_grounding_failure")},
        "seeds": {"selection": SELECTION_SEED, "failure": FAILURE_SEED, "branch": BRANCH_SEED},
        "candidate_artifact_frozen_before_generation": "results/share_expected_voh_filter_development_20260829/expected_voh_filter.joblib",
        "method_artifacts_accessed_by_generator": False,
    })
    historical, sources = base.historical_identifiers(
        HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json"
    )
    blocked = {str(row["group_id"]) for path in SOURCE_FILES for row in base.read_jsonl(path)}
    representatives = {}
    for path, document, operation in iter_operations(ROOT / "dataset/api"):
        row = workflow_record(path, document, operation)
        if row is None or str(row["group_id"]) in blocked:
            continue
        features = base.source_features(row)
        ids = base.candidate_identifiers(row, features)
        if base.overlaps(ids, historical):
            continue
        rank = base.sha256_text(f"{SELECTION_SEED}:{base.canonical_json(row)}")
        item = (rank, row, {**features, "source_record_sha256": base.sha256_text(base.canonical_json(row))}, ids)
        if ids["group_id"] not in representatives or rank < representatives[ids["group_id"]][0]:
            representatives[ids["group_id"]] = item
    if len(representatives) < 150:
        raise SystemExit(f"only {len(representatives)} eligible groups")
    ranked = sorted(representatives.values(), key=lambda x: (x[2]["complexity_score"], base.sha256_text(f"{SELECTION_SEED}:{x[3]['group_id']}")))
    parts = {"low": ranked[:50], "medium": ranked[50:100], "high": ranked[-50:]}
    selected = [
        {"record": row, "features": features, "identifiers": ids, "complexity_bin": label}
        for label in ("low", "medium", "high") for _, row, features, ids in parts[label]
    ]
    base.PROTOCOL_ID = PROTOCOL
    base.SELECTION_SEED, base.FAILURE_SEED, base.BRANCH_SEED = SELECTION_SEED, FAILURE_SEED, BRANCH_SEED
    base.GENERATED_AT = created
    base.FAILURE_COUNTS = [(name, 25) for name in (
        "planning_failure", "service_selection_failure", "parameter_binding_failure",
        "tool_execution_failure", "supervisor_failure", "answer_grounding_failure")]
    base.COMPLEXITY_COUNTS = [("low", 50), ("medium", 50), ("high", 50)]
    base.assign_failures(selected)
    episodes = sorted((base.build_episode(item) for item in selected), key=lambda e: e.episode_id)
    audit = base.overlap_audit(episodes, historical)
    duplicates = base.ensure_unique(episodes)
    source_path = OUT / "source_input.jsonl"
    source_path.write_text("".join(base.canonical_json(item["record"]) + "\n" for item in selected), encoding="utf-8")
    episode_path = OUT / "episodes.jsonl"
    write_episodes(episode_path, episodes)
    if len(load_episodes(episode_path)) != 150:
        raise SystemExit("reload failure")
    dump(OUT / "import_report.json", {
        "protocol": PROTOCOL, "created_at_utc": created, "eligible_groups": len(representatives),
        "output": {"path": str(episode_path.relative_to(ROOT)), "sha256": base.sha256_file(episode_path), "episodes": 150, "groups": 150},
        "overlap_audit": audit, "duplicate_audit": duplicates, "historical_sources": sources,
        "method_artifacts_accessed": False, "evaluator_run": False,
    })
    dump(OUT / "manifest.json", {"protocol": PROTOCOL, "status": "generated_audited_unbound", "episodes_sha256": base.sha256_file(episode_path), "audit_result": audit["result"]})
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{base.sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"episodes": 150, "eligible_groups": len(representatives), "overlap": audit["result"]}, indent=2))


if __name__ == "__main__":
    main()
