"""Generate the third, unbound SHARE-V3 final replay holdout.

The generator excludes both earlier 400-workflow source pools and all declared
development identifiers. It never loads controller artifacts or outcomes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_share_locked_holdout_replayable as base
from scripts.build_share_public_api_workflow_input import iter_operations, workflow_record
from share.data import load_episodes, write_episodes

PROTOCOL = "share-v3-final-replay-400-v1"
SELECTION_SEED, FAILURE_SEED, BRANCH_SEED = 2026082704, 2026082705, 2026082706
OLD_SOURCE = ROOT / "datasets/share_public_api_workflow_input.jsonl"
V2_SOURCE = ROOT / "datasets/share_v2_confirmation_holdout_20260827/source_input.jsonl"
OLD_HOLDOUT = ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl"
V2_VALIDATION = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
HISTORICAL = [
    ROOT / "datasets/share_synthetic/episodes.jsonl",
    ROOT / "datasets/share_guard_development/episodes.jsonl",
    ROOT / "datasets/share_synthetic_augmented_guard_development/episodes.jsonl",
    OLD_HOLDOUT,
    V2_VALIDATION,
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    out = ROOT / "datasets/share_v3_final_holdout_20260827"
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite V3 final holdout")
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prereg = {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "target": {"episodes": 400, "groups": 400, "synthetic": True},
        "seeds": {"selection": SELECTION_SEED, "failure": FAILURE_SEED, "branch": BRANCH_SEED},
        "source_rule": "previously unused local public-API document groups after excluding both earlier 884-group source pools",
        "required_zero_overlap": ["episode_id", "workflow_id", "group_id", "source_document_sha256", "normalized_source_fingerprint"],
        "prohibitions": ["no_method_artifacts", "no_method_outcomes", "no_evaluator_binding"],
    }
    write_json(out / "generation_preregistration.json", prereg)

    historical, sources = base.historical_identifiers(HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json")
    blocked_groups = {
        str(row["group_id"])
        for path in (OLD_SOURCE, V2_SOURCE)
        for row in base.read_jsonl(path)
    }
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for path, document, operation in iter_operations(ROOT / "dataset/api"):
        row = workflow_record(path, document, operation)
        if row is not None and str(row["group_id"]) not in blocked_groups:
            rows_by_group[str(row["group_id"])].append(row)
    if len(rows_by_group) < 884:
        raise SystemExit(f"only {len(rows_by_group)} third-wave source groups; need 884")
    groups = sorted(rows_by_group, key=lambda group: base.sha256_text(f"{SELECTION_SEED}:{group}"))[:884]
    source_rows = [min(rows_by_group[group], key=base.canonical_json) for group in groups]
    extras = sorted(
        (row for group in groups for row in rows_by_group[group] if row not in source_rows),
        key=lambda row: base.sha256_text(f"{SELECTION_SEED}:{base.canonical_json(row)}"),
    )
    source_rows.extend(extras[: 3000 - len(source_rows)])
    if len(source_rows) != 3000 or len({row["group_id"] for row in source_rows}) != 884:
        raise SystemExit("unable to construct third fixed 3,000-record/884-group source pool")
    source_path = out / "source_input.jsonl"
    source_path.write_text("".join(base.canonical_json(row) + "\n" for row in source_rows), encoding="utf-8")

    base.PROTOCOL_ID, base.SELECTION_SEED, base.FAILURE_SEED, base.BRANCH_SEED = PROTOCOL, SELECTION_SEED, FAILURE_SEED, BRANCH_SEED
    base.GENERATED_AT = created
    selected, rejected = base.select_records(source_rows, historical)
    base.assign_failures(selected)
    episodes = sorted((base.build_episode(item) for item in selected), key=lambda item: item.episode_id)
    if len(episodes) != 400 or len({episode.group_id for episode in episodes}) != 400:
        raise SystemExit("V3 final target quota failed")
    duplicates = base.ensure_unique(episodes)
    audit = base.overlap_audit(episodes, historical)
    episode_path = out / "episodes.jsonl"
    write_episodes(episode_path, episodes)
    if len(load_episodes(episode_path)) != 400:
        raise SystemExit("V3 final canonical reload failed")
    selection_path = out / "source_selection.jsonl"
    selection_path.write_text("".join(base.canonical_json({
        "episode_id": episode.episode_id,
        "workflow_id": episode.metadata["source_workflow_id"],
        "group_id": episode.group_id,
        "failure_type": episode.metadata["failure_type"],
        "complexity_bin": episode.metadata["complexity_bin"],
    }) + "\n" for episode in episodes), encoding="utf-8")
    distributions = {
        "failure_type": dict(Counter(episode.metadata["failure_type"] for episode in episodes)),
        "complexity": dict(Counter(episode.metadata["complexity_bin"] for episode in episodes)),
    }
    digest = base.sha256_file
    report = {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "source": {"path": str(source_path.relative_to(ROOT)), "sha256": digest(source_path), "records": 3000, "groups": 884, "synthetic": False},
        "output": {"path": str(episode_path.relative_to(ROOT)), "sha256": digest(episode_path), "episodes": 400, "groups": 400, "synthetic": True},
        "excluded_source_groups": len(blocked_groups),
        "historical_sources": sources,
        "overlap_audit": audit,
        "duplicate_audit": duplicates,
        "distributions": distributions,
        "method_artifacts_accessed": False,
        "method_level_outcomes_accessed": False,
        "evaluator_run": False,
    }
    write_json(out / "import_report.json", report)
    write_json(out / "prior_unseen_attestation.json", {
        "protocol": PROTOCOL,
        "created_at_utc": created,
        "episodes_sha256": digest(episode_path),
        "audit": audit,
        "all_source_records_non_synthetic": True,
        "all_branch_outcomes_synthetic": True,
        "method_outcomes_not_accessed": True,
        "evaluator_not_run": True,
        "limitation": "real-source public API documentation with deterministic replayed AI/Human/Stop outcomes; not observed execution",
    })
    write_json(out / "manifest.json", {
        "protocol": PROTOCOL,
        "status": "generated_audited_unbound",
        "dataset": report["output"],
        "source": report["source"],
        "files": {name: digest(out / name) for name in ("generation_preregistration.json", "source_selection.jsonl", "import_report.json", "prior_unseen_attestation.json")},
        "audit_result": audit["result"],
        "evaluator_bound": False,
    })
    paths = sorted(path for path in out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in paths), encoding="utf-8")
    print(json.dumps({"out": str(out), "episodes": 400, "groups": 400, "overlap": audit["result"]}, indent=2))


if __name__ == "__main__":
    main()
