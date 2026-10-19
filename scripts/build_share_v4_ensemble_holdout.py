"""Build a fourth, 250-group holdout for the frozen SHARE Q ensemble."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_share_locked_holdout_replayable as base
from scripts.build_share_public_api_workflow_input import iter_operations, workflow_record
from share.data import load_episodes, write_episodes

OUT = ROOT / "datasets/share_v4_ensemble_holdout_20260829"
PROTOCOL = "share-v4-ensemble-replay-250-v1"
SELECTION_SEED, FAILURE_SEED, BRANCH_SEED = 2026082901, 2026082902, 2026082903
TARGET = 250
COMPLEXITY_COUNTS = (("low", 84), ("medium", 83), ("high", 83))
FAILURE_COUNTS = (
    ("planning_failure", 42), ("service_selection_failure", 42),
    ("parameter_binding_failure", 42), ("tool_execution_failure", 42),
    ("supervisor_failure", 41), ("answer_grounding_failure", 41),
)
SOURCE_FILES = (
    ROOT / "datasets/share_public_api_workflow_input.jsonl",
    ROOT / "datasets/share_v2_confirmation_holdout_20260827/source_input.jsonl",
    ROOT / "datasets/share_v3_final_holdout_20260827/source_input.jsonl",
)
HISTORICAL = (
    ROOT / "datasets/share_synthetic/episodes.jsonl",
    ROOT / "datasets/share_guard_development/episodes.jsonl",
    ROOT / "datasets/share_synthetic_augmented_guard_development/episodes.jsonl",
    ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl",
    ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl",
    ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("refusing to overwrite V4 holdout")
    OUT.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prereg = {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "target": {"episodes": TARGET, "groups": TARGET, "synthetic_branches": True},
        "complexity_counts": dict(COMPLEXITY_COUNTS),
        "failure_counts": dict(FAILURE_COUNTS),
        "seeds": {"selection": SELECTION_SEED, "failure": FAILURE_SEED, "branch": BRANCH_SEED},
        "source_rule": "all remaining public-API groups after excluding V1, V2, and V3 source pools",
        "required_zero_overlap": ["episode_id", "workflow_id", "group_id", "source_document_sha256", "normalized_source_fingerprint"],
        "method_artifacts_accessed": False,
    }
    write_json(OUT / "generation_preregistration.json", prereg)

    historical, historical_sources = base.historical_identifiers(
        HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json"
    )
    blocked_groups = {
        str(row["group_id"]) for path in SOURCE_FILES for row in base.read_jsonl(path)
    }
    representatives: dict[str, tuple[str, dict, dict, dict]] = {}
    rejected: Counter[str] = Counter()
    for path, document, operation in iter_operations(ROOT / "dataset/api"):
        row = workflow_record(path, document, operation)
        if row is None or str(row["group_id"]) in blocked_groups:
            continue
        features = base.source_features(row)
        ids = base.candidate_identifiers(row, features)
        overlap = base.overlaps(ids, historical)
        if overlap:
            rejected.update(f"historical_overlap:{key}" for key in overlap)
            continue
        rank = base.sha256_text(f"{SELECTION_SEED}:{base.canonical_json(row)}")
        group = ids["group_id"]
        item = (rank, row, {**features, "source_record_sha256": base.sha256_text(base.canonical_json(row))}, ids)
        if group not in representatives or rank < representatives[group][0]:
            representatives[group] = item
    if len(representatives) < TARGET:
        raise SystemExit(f"only {len(representatives)} eligible groups remain")

    ranked = sorted(
        representatives.values(),
        key=lambda item: (item[2]["complexity_score"], base.sha256_text(f"{SELECTION_SEED}:{item[3]['group_id']}")),
    )
    parts = {"low": ranked[:84], "medium": ranked[84:167], "high": ranked[-83:]}
    selected = [
        {"record": row, "features": features, "identifiers": ids, "complexity_bin": label}
        for label, expected in COMPLEXITY_COUNTS
        for _, row, features, ids in parts[label]
    ]
    if len(selected) != TARGET or len({x["identifiers"]["group_id"] for x in selected}) != TARGET:
        raise SystemExit("selection quota or uniqueness failure")

    base.PROTOCOL_ID = PROTOCOL
    base.SELECTION_SEED, base.FAILURE_SEED, base.BRANCH_SEED = SELECTION_SEED, FAILURE_SEED, BRANCH_SEED
    base.GENERATED_AT = created
    base.FAILURE_COUNTS = list(FAILURE_COUNTS)
    base.COMPLEXITY_COUNTS = list(COMPLEXITY_COUNTS)
    base.assign_failures(selected)
    episodes = sorted((base.build_episode(item) for item in selected), key=lambda episode: episode.episode_id)
    duplicates = base.ensure_unique(episodes)
    audit = base.overlap_audit(episodes, historical)

    source_path = OUT / "source_input.jsonl"
    source_path.write_text(
        "".join(base.canonical_json(item["record"]) + "\n" for item in selected), encoding="utf-8"
    )
    episode_path = OUT / "episodes.jsonl"
    write_episodes(episode_path, episodes)
    if len(load_episodes(episode_path)) != TARGET:
        raise SystemExit("canonical reload failed")
    selection_path = OUT / "source_selection.jsonl"
    selection_path.write_text(
        "".join(base.canonical_json({
            "episode_id": episode.episode_id,
            "workflow_id": episode.metadata["source_workflow_id"],
            "group_id": episode.group_id,
            "failure_type": episode.metadata["failure_type"],
            "complexity_bin": episode.metadata["complexity_bin"],
        }) + "\n" for episode in episodes), encoding="utf-8"
    )
    report = {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "eligible_groups_before_selection": len(representatives),
        "output": {"path": str(episode_path.relative_to(ROOT)), "sha256": base.sha256_file(episode_path), "episodes": TARGET, "groups": TARGET},
        "overlap_audit": audit,
        "duplicate_audit": duplicates,
        "rejections": dict(rejected),
        "historical_sources": historical_sources,
        "method_artifacts_accessed": False,
        "evaluator_run": False,
    }
    write_json(OUT / "import_report.json", report)
    write_json(OUT / "manifest.json", {"protocol": PROTOCOL, "status": "generated_audited_unbound", "dataset": report["output"], "audit_result": audit["result"], "evaluator_bound": False})
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{base.sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "groups": len({e.group_id for e in episodes}), "eligible_groups": len(representatives), "overlap": audit["result"]}, indent=2))


if __name__ == "__main__":
    main()
