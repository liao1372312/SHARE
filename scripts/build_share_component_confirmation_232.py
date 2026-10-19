"""Freeze all remaining group-disjoint public-API replay cases for E2 comparison."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_share_locked_holdout_replayable as base
from scripts.audit_component_confirmation_readiness import HISTORICAL, SOURCES
from scripts.build_share_public_api_workflow_input import iter_operations, workflow_record
from share.data import load_episodes, write_episodes


OUT = ROOT / "datasets/share_component_confirmation_232_20260930"
PROTOCOL = "share-component-confirmation-public-api-replay-232-v1"
SELECTION_SEED, FAILURE_SEED, BRANCH_SEED = 2026093001, 2026093002, 2026093003
FAILURE_NAMES = (
    "planning_failure", "service_selection_failure", "parameter_binding_failure",
    "tool_execution_failure", "supervisor_failure", "answer_grounding_failure",
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    OUT.mkdir(parents=True)
    dump(OUT / "generation_preregistration.json", {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "target": "all eligible new public-API document groups remaining after V5; expected 232",
        "source_rule": "same public-API document parser and exclusions as V5",
        "representative_rule": "minimum fixed-seed SHA256 rank per document group",
        "complexity_rule": "rank selected operations by complexity score; thirds are low/medium/high",
        "failure_rule": "deterministic shuffled balanced assignment across six categories",
        "seeds": {"selection": SELECTION_SEED, "failure": FAILURE_SEED, "branch": BRANCH_SEED},
        "human_ai_stop_outcomes": "deterministic simulator branches, not observed repairs",
        "method_artifacts_accessed_by_generator": False,
    })
    historical, historical_sources = base.historical_identifiers(
        HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json"
    )
    blocked = {str(row["group_id"]) for path in SOURCES for row in base.read_jsonl(path)}
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
    count = len(representatives)
    if count != 232:
        raise SystemExit(f"expected 232 eligible groups from readiness audit, found {count}")
    ranked = sorted(
        representatives.values(),
        key=lambda x: (x[2]["complexity_score"], base.sha256_text(f"{SELECTION_SEED}:{x[3]['group_id']}")),
    )
    low, medium = count // 3, count // 3
    parts = {"low": ranked[:low], "medium": ranked[low:low + medium], "high": ranked[low + medium:]}
    selected = [
        {"record": row, "features": features, "identifiers": ids, "complexity_bin": label}
        for label in ("low", "medium", "high") for _, row, features, ids in parts[label]
    ]
    base.PROTOCOL_ID = PROTOCOL
    base.SELECTION_SEED, base.FAILURE_SEED, base.BRANCH_SEED = SELECTION_SEED, FAILURE_SEED, BRANCH_SEED
    base.GENERATED_AT = created
    quotient, remainder = divmod(count, len(FAILURE_NAMES))
    base.FAILURE_COUNTS = [(name, quotient + (index < remainder)) for index, name in enumerate(FAILURE_NAMES)]
    base.COMPLEXITY_COUNTS = [(label, len(parts[label])) for label in ("low", "medium", "high")]
    base.assign_failures(selected)
    episodes = sorted((base.build_episode(item) for item in selected), key=lambda e: e.episode_id)
    duplicates = base.ensure_unique(episodes)
    audit = base.overlap_audit(episodes, historical)
    if audit["result"] != "PASS":
        raise SystemExit(f"holdout overlap audit failed: {audit}")
    source_path = OUT / "source_input.jsonl"
    source_path.write_text("".join(base.canonical_json(item["record"]) + "\n" for item in selected), encoding="utf-8")
    episode_path = OUT / "episodes.jsonl"
    write_episodes(episode_path, episodes)
    if len(load_episodes(episode_path)) != count:
        raise SystemExit("episode reload count mismatch")
    dump(OUT / "manifest.json", {
        "protocol": PROTOCOL,
        "status": "generated_audited_unbound",
        "created_at_utc": created,
        "episodes": count,
        "groups": count,
        "states": sum(len(ep.steps) for ep in episodes),
        "complexity_counts": dict(base.COMPLEXITY_COUNTS),
        "failure_counts": dict(base.FAILURE_COUNTS),
        "episodes_sha256": base.sha256_file(episode_path),
        "source_input_sha256": base.sha256_file(source_path),
        "overlap_audit": audit,
        "duplicate_audit": duplicates,
        "historical_sources": historical_sources,
        "model_artifacts_accessed": False,
        "evaluator_run": False,
    })
    print(json.dumps({"episodes": count, "states": sum(len(ep.steps) for ep in episodes), "overlap": audit["result"]}))


if __name__ == "__main__":
    main()
