"""Create an unbound, V2-confirmation replay holdout from unused local API documents.

No controller artifact or outcome is loaded.  The source pool excludes every
document group in the original API input and every declared development or
previous-holdout identifier before deterministic selection.
"""
from __future__ import annotations

import hashlib
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

PROTOCOL = "share-v2-confirmation-replay-400-v1"
SELECTION_SEED, FAILURE_SEED, BRANCH_SEED = 2026082701, 2026082702, 2026082703
OLD_SOURCE = ROOT / "datasets/share_public_api_workflow_input.jsonl"
OLD_HOLDOUT = ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl"
HISTORICAL = [
    ROOT / "datasets/share_synthetic/episodes.jsonl",
    ROOT / "datasets/share_guard_development/episodes.jsonl",
    ROOT / "datasets/share_synthetic_augmented_guard_development/episodes.jsonl",
    OLD_HOLDOUT,
]


def digest(path: Path) -> str:
    return base.sha256_file(path)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    out = ROOT / "datasets/share_v2_confirmation_holdout_20260827"
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite a confirmation-holdout ledger")
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prereg = {
        "protocol": PROTOCOL, "created_at_utc": created,
        "prohibitions": ["no_method_artifacts", "no_method_outcomes", "no_evaluator_binding"],
        "target": {"episodes": 400, "groups": 400, "synthetic": True},
        "seeds": {"selection": SELECTION_SEED, "failure": FAILURE_SEED, "branch": BRANCH_SEED},
        "source_rule": "unused local public-API document groups only; one representative per selected group",
        "required_zero_overlap": ["episode_id", "workflow_id", "group_id", "source_document_sha256", "normalized_source_fingerprint"],
    }
    write(out / "generation_preregistration.json", prereg)

    historical, sources = base.historical_identifiers(HISTORICAL, ROOT / "datasets/share_guard_development/historical_identifiers.json")
    old_rows = base.read_jsonl(OLD_SOURCE)
    blocked_groups = {str(row["group_id"]) for row in old_rows}
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for path, document, operation in iter_operations(ROOT / "dataset/api"):
        row = workflow_record(path, document, operation)
        if row is not None and row["group_id"] not in blocked_groups:
            rows_by_group[str(row["group_id"])].append(row)
    if len(rows_by_group) < 884:
        raise SystemExit(f"only {len(rows_by_group)} unused source groups; need 884")
    groups = sorted(rows_by_group, key=lambda group: base.sha256_text(f"{SELECTION_SEED}:{group}"))[:884]
    # Guarantee every selected group is represented, then fill to exactly 3,000
    # records deterministically; the imported base selector subsequently chooses
    # one representative/group and applies the predeclared strata.
    source_rows = [min(rows_by_group[group], key=base.canonical_json) for group in groups]
    extras = sorted(
        (row for group in groups for row in rows_by_group[group] if row not in source_rows),
        key=lambda row: base.sha256_text(f"{SELECTION_SEED}:{base.canonical_json(row)}"),
    )
    source_rows.extend(extras[: 3000 - len(source_rows)])
    if len(source_rows) != 3000 or len({row["group_id"] for row in source_rows}) != 884:
        raise SystemExit("unable to construct fixed 3,000-record/884-group source pool")
    source_path = out / "source_input.jsonl"
    source_path.write_text("".join(base.canonical_json(row) + "\n" for row in source_rows), encoding="utf-8")

    # The imported pure data-generation functions resolve these globals at call
    # time.  This leaves the historical generator and its sealed old hashes intact.
    base.PROTOCOL_ID, base.SELECTION_SEED, base.FAILURE_SEED, base.BRANCH_SEED = PROTOCOL, SELECTION_SEED, FAILURE_SEED, BRANCH_SEED
    base.GENERATED_AT = created
    selected, rejected = base.select_records(source_rows, historical)
    base.assign_failures(selected)
    episodes = sorted((base.build_episode(item) for item in selected), key=lambda item: item.episode_id)
    if len(episodes) != 400 or len({item.group_id for item in episodes}) != 400:
        raise SystemExit("confirmation target quota failed")
    duplicates = base.ensure_unique(episodes)
    audit = base.overlap_audit(episodes, historical)
    episode_path = out / "episodes.jsonl"
    write_episodes(episode_path, episodes)
    if len(load_episodes(episode_path)) != 400:
        raise SystemExit("canonical episode reload failed")
    selection_path = out / "source_selection.jsonl"
    selection_path.write_text("".join(base.canonical_json({"episode_id": ep.episode_id, "workflow_id": ep.metadata["source_workflow_id"], "group_id": ep.group_id, "failure_type": ep.metadata["failure_type"], "complexity_bin": ep.metadata["complexity_bin"]}) + "\n" for ep in episodes), encoding="utf-8")
    counts = {"failure_type": dict(Counter(ep.metadata["failure_type"] for ep in episodes)), "complexity": dict(Counter(ep.metadata["complexity_bin"] for ep in episodes))}
    report = {"protocol": PROTOCOL, "created_at_utc": created, "source": {"path": str(source_path.relative_to(ROOT)), "sha256": digest(source_path), "records": 3000, "groups": 884, "synthetic": False}, "output": {"path": str(episode_path.relative_to(ROOT)), "sha256": digest(episode_path), "episodes": 400, "groups": 400, "synthetic": True}, "historical_sources": sources, "overlap_audit": audit, "duplicate_audit": duplicates, "distributions": counts, "method_artifacts_accessed": False, "method_level_outcomes_accessed": False, "evaluator_run": False}
    write(out / "import_report.json", report)
    attestation = {"protocol": PROTOCOL, "created_at_utc": created, "episodes_sha256": digest(episode_path), "audit": audit, "all_source_records_non_synthetic": True, "all_branch_outcomes_synthetic": True, "method_outcomes_not_accessed": True, "evaluator_not_run": True, "limitation": "real-source public API documentation with deterministic replayed AI/Human/Stop outcomes; not observed execution"}
    write(out / "prior_unseen_attestation.json", attestation)
    manifest = {"protocol": PROTOCOL, "status": "generated_audited_unbound", "dataset": report["output"], "source": report["source"], "files": {name: digest(out / name) for name in ("generation_preregistration.json", "source_selection.jsonl", "import_report.json", "prior_unseen_attestation.json")}, "audit_result": audit["result"], "evaluator_bound": False}
    write(out / "manifest.json", manifest)
    paths = sorted(path for path in out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in paths), encoding="utf-8")
    print(json.dumps({"out": str(out), "episodes": 400, "groups": 400, "overlap": audit["result"]}, indent=2))

if __name__ == "__main__":
    main()
