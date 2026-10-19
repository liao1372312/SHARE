from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import label_human_benefit, load_episodes, write_episodes
from share.types import BranchOutcome, PairedRepairStep, RepairEpisode, RepairState, RewardConfig


PROTOCOL_ID = "share-locked-replayable-400-v3"
SIMULATOR_VERSION = "share-replay-sim-v2"
SOURCE_SHA256 = "64125bb8750c07a0c2d53e555c1a7070aff4fdc77be395c93ee056d05f3f133f"
PROTOCOL_V2_SHA256 = "8d50a3415dae9895d3ff5f21c4e86647f6fbf5854d7955523eec7497662a4337"
PROTOCOL_V3_SHA256 = "3c3fd19198521c02d7b5c2f28b330030fdeb69e954848fb2ad9d3a5838153536"
SELECTION_SEED = 2026082601
FAILURE_SEED = 2026082602
BRANCH_SEED = 2026082603
GENERATED_AT = "2026-08-26T00:08:00Z"

FAILURE_COUNTS = [
    ("planning_failure", 67),
    ("service_selection_failure", 67),
    ("parameter_binding_failure", 67),
    ("tool_execution_failure", 67),
    ("supervisor_failure", 66),
    ("answer_grounding_failure", 66),
]
COMPLEXITY_COUNTS = [("low", 134), ("medium", 133), ("high", 133)]
COMPLEXITY_VALUE = {"low": 0.30, "medium": 0.65, "high": 0.90}
HORIZON = {"low": 2, "medium": 3, "high": 5}
AI_OFFSET = {
    "planning_failure": -0.08,
    "service_selection_failure": -0.02,
    "parameter_binding_failure": 0.04,
    "tool_execution_failure": 0.00,
    "supervisor_failure": -0.12,
    "answer_grounding_failure": -0.06,
}
UNSAFE_OFFSET = {
    "planning_failure": 0.03,
    "service_selection_failure": 0.02,
    "parameter_binding_failure": 0.01,
    "tool_execution_failure": 0.05,
    "supervisor_failure": 0.06,
    "answer_grounding_failure": 0.04,
}
RISK_TERMS = (
    "delete", "remove", "revoke", "disable", "terminate", "write", "update",
    "create", "payment", "secret", "token", "credential", "admin",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uniform(key: str) -> float:
    return int(sha256_text(f"{BRANCH_SEED}:{key}")[:13], 16) / float(16**13)


def clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def operation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    context = record.get("context")
    value = context.get("operation") if isinstance(context, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"workflow {record.get('workflow_id')} lacks context.operation")
    return value


def method_mutability(record: Mapping[str, Any]) -> float:
    method = str(operation(record).get("method", "")).upper()
    if method in {"PATCH", "DELETE"}:
        return 1.0
    if method in {"POST", "PUT"}:
        return 0.5
    return 0.0


def source_features(record: Mapping[str, Any]) -> dict[str, Any]:
    op = operation(record)
    method = str(op.get("method", "")).upper()
    params = op.get("required_parameters", [])
    if not isinstance(params, list):
        raise ValueError(f"workflow {record.get('workflow_id')} has invalid required_parameters")
    text = " ".join(str(op.get(key, "")) for key in ("operation_name", "description", "endpoint_path")).lower()
    risk_hits = min(3, sum(term in text for term in RISK_TERMS))
    description_length = len(str(op.get("description", "")))
    mutating = int(method in {"POST", "PUT"})
    destructive = int(method in {"PATCH", "DELETE"})
    score = 4 * destructive + 2 * mutating + min(len(params), 6) + min(description_length / 300.0, 2.0) + risk_hits
    normalized = {
        "source": record.get("source"),
        "group_id": record.get("group_id"),
        "operation_name": op.get("operation_name"),
        "method": method,
        "endpoint_path": op.get("endpoint_path"),
        "required_parameters": params,
    }
    return {
        "method": method,
        "required_parameter_count": len(params),
        "risk_lexicon_hits": risk_hits,
        "description_character_count": description_length,
        "complexity_score": score,
        "normalized_source_fingerprint": sha256_text(canonical_json(normalized)),
    }


def identifiers_from_episode_file(path: Path) -> dict[str, set[str]]:
    found = {key: set() for key in ("episode_id", "workflow_id", "group_id", "source_document_sha256", "normalized_source_fingerprint")}
    if not path.is_file():
        raise ValueError(f"declared historical source is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), Mapping) else {}
            mappings = {
                "episode_id": row.get("episode_id"),
                "workflow_id": metadata.get("source_workflow_id") or metadata.get("workflow_id"),
                "group_id": row.get("group_id"),
                "source_document_sha256": metadata.get("source_document_sha256"),
                "normalized_source_fingerprint": metadata.get("normalized_source_fingerprint"),
            }
            for key, value in mappings.items():
                if isinstance(value, str) and value:
                    found[key].add(value)
    return found


def merge_identifiers(target: dict[str, set[str]], source: Mapping[str, Iterable[str]]) -> None:
    for key in target:
        target[key].update(str(value) for value in source.get(key, []) if value)


def historical_identifiers(paths: list[Path], index_path: Path) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    identifiers = {key: set() for key in ("episode_id", "workflow_id", "group_id", "source_document_sha256", "normalized_source_fingerprint")}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    merge_identifiers(identifiers, {
        "episode_id": index.get("historical_episode_ids", []),
        "group_id": index.get("historical_group_ids", []),
    })
    sources = [{"path": str(index_path), "sha256": sha256_file(index_path), "kind": "metadata_only_identifier_index"}]
    for path in paths:
        merge_identifiers(identifiers, identifiers_from_episode_file(path))
        sources.append({"path": str(path), "sha256": sha256_file(path), "kind": "episode_identifiers_and_metadata_only"})
    return identifiers, sources


def candidate_identifiers(record: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, str]:
    metadata = record.get("metadata", {})
    workflow_id = str(record["workflow_id"])
    return {
        "episode_id": "locked-replay-" + sha256_text(f"{SELECTION_SEED}:{workflow_id}")[:24],
        "workflow_id": workflow_id,
        "group_id": str(record["group_id"]),
        "source_document_sha256": str(metadata["source_document_sha256"]),
        "normalized_source_fingerprint": str(features["normalized_source_fingerprint"]),
    }


def overlaps(values: Mapping[str, str], historical: Mapping[str, set[str]]) -> list[str]:
    return [key for key, value in values.items() if value in historical[key]]


def select_records(rows: list[dict[str, Any]], historical: Mapping[str, set[str]]) -> tuple[list[dict[str, Any]], Counter]:
    if len(rows) != 3000:
        raise ValueError(f"source count mismatch: expected 3000, observed {len(rows)}")
    if len({row.get("group_id") for row in rows}) != 884:
        raise ValueError("source group count mismatch")
    representatives: dict[str, tuple[str, dict[str, Any], dict[str, Any], dict[str, str]]] = {}
    rejected: Counter = Counter()
    for row in rows:
        if row.get("synthetic") is not False:
            raise ValueError("source record is not explicitly synthetic=false")
        features = source_features(row)
        ids = candidate_identifiers(row, features)
        dimensions = overlaps(ids, historical)
        if dimensions:
            for dimension in dimensions:
                rejected[f"historical_overlap:{dimension}"] += 1
            continue
        digest = sha256_text(canonical_json(row))
        rank = sha256_text(f"{SELECTION_SEED}:{canonical_json(row)}")
        group_id = ids["group_id"]
        candidate = (rank, row, {**features, "source_record_sha256": digest}, ids)
        if group_id not in representatives or rank < representatives[group_id][0]:
            representatives[group_id] = candidate
    if len(representatives) < 400:
        raise ValueError(f"only {len(representatives)} eligible groups remain")
    ranked = sorted(representatives.values(), key=lambda item: (item[2]["complexity_score"], sha256_text(f"{SELECTION_SEED}:{item[3]['group_id']}")))
    chosen_parts = {
        "low": ranked[:134],
        "medium": ranked[134:267],
        "high": ranked[-133:],
    }
    selected: list[dict[str, Any]] = []
    for label, expected in COMPLEXITY_COUNTS:
        part = chosen_parts[label]
        if len(part) != expected:
            raise ValueError(f"complexity quota mismatch for {label}")
        for _, row, features, ids in part:
            selected.append({"record": row, "features": features, "identifiers": ids, "complexity_bin": label})
    return selected, rejected


def assign_failures(selected: list[dict[str, Any]]) -> None:
    ordered = sorted(selected, key=lambda item: sha256_text(f"{FAILURE_SEED}:{item['identifiers']['group_id']}"))
    assignments = [failure for failure, count in FAILURE_COUNTS for _ in range(count)]
    if len(assignments) != len(ordered):
        raise ValueError("failure quota does not match selected episodes")
    for item, failure in zip(ordered, assignments):
        item["failure_type"] = failure


def branch_metadata(item: Mapping[str, Any], step: int, branch: str, probability: float | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source": SIMULATOR_VERSION,
        "evidence_kind": "deterministic_simulation_not_observed_execution",
        "source_record_sha256": item["features"]["source_record_sha256"],
        "protocol_id": PROTOCOL_ID,
        "branch_seed": BRANCH_SEED,
        "step": step,
        "branch": branch,
    }
    if probability is not None:
        value["success_probability"] = probability
    return value


def build_episode(item: Mapping[str, Any]) -> RepairEpisode:
    record = item["record"]
    features = item["features"]
    ids = item["identifiers"]
    complexity_bin = str(item["complexity_bin"])
    failure = str(item["failure_type"])
    episode_id = ids["episode_id"]
    c = COMPLEXITY_VALUE[complexity_bin]
    horizon = HORIZON[complexity_bin]
    mutability = method_mutability(record)
    parameter_norm = min(float(features["required_parameter_count"]) / 6.0, 1.0)
    steps: list[PairedRepairStep] = []
    for step in range(horizon):
        affected = clip(0.15 + 0.35 * c + 0.10 * parameter_norm)
        state = RepairState(
            episode_id=episode_id,
            step=step,
            gap_type=failure,
            root_type=failure,
            localization_score=clip(0.82 - 0.12 * c - 0.05 * step + 0.08 * stable_uniform(f"{episode_id}:{step}:localization")),
            repair_utility=clip(0.72 - 0.08 * c + 0.10 * stable_uniform(f"{episode_id}:{step}:utility")),
            affected_ratio=affected,
            downstream_impact=clip(affected * (1 + 0.6 * c)),
            drift_risk=clip(0.10 + 0.35 * c + 0.20 * mutability + 0.05 * step),
            previous_failures=step / max(horizon - 1, 1),
            uncertainty=clip(0.15 + 0.25 * c + 0.12 * step + 0.15 * stable_uniform(f"{episode_id}:{step}:uncertainty")),
            evidence_conflict=0.55 if failure == "answer_grounding_failure" else 0.10 * stable_uniform(f"{episode_id}:{step}:conflict"),
            remaining_budget=(horizon - step) / horizon,
            complexity=c,
            domain="public_api_documentation_replay",
        )
        ai_p = clip(0.70 + AI_OFFSET[failure] - 0.18 * c - 0.07 * step - 0.10 * mutability + 0.08 * stable_uniform(f"{episode_id}:{step}:ai:p"))
        ai_success = stable_uniform(f"{episode_id}:{step}:ai:success") < ai_p
        ai_unsafe_p = clip(0.10 + UNSAFE_OFFSET[failure] + 0.25 * c + 0.20 * mutability + 0.05 * step)
        ai_unsafe = (not ai_success) and stable_uniform(f"{episode_id}:{step}:ai:unsafe") < ai_unsafe_p
        ai_terminal = ai_success or step == horizon - 1
        human_p = clip(0.93 - 0.08 * c - 0.03 * step + 0.04 * stable_uniform(f"{episode_id}:{step}:human:p"))
        human_success = stable_uniform(f"{episode_id}:{step}:human:success") < human_p
        human_unsafe_p = clip(0.01 + 0.03 * c + 0.02 * mutability)
        human_unsafe = (not human_success) and stable_uniform(f"{episode_id}:{step}:human:unsafe") < human_unsafe_p
        outcomes = {
            "ai": BranchOutcome(
                success=ai_success, unsafe=ai_unsafe, quality_delta=0.8 if ai_success else -0.35,
                latency=0.15 + 0.25 * affected, execution_cost=0.12 + 0.35 * affected,
                next_step=None if ai_terminal else step + 1, terminal=ai_terminal,
                metadata={**branch_metadata(item, step, "ai", ai_p), "unsafe_probability_given_failure": ai_unsafe_p},
            ),
            "human": BranchOutcome(
                success=human_success, unsafe=human_unsafe, quality_delta=0.9 if human_success else -0.15,
                latency=0.55 + 0.35 * c, execution_cost=0.18, human_minutes=4 + 8 * c + 2 * step,
                terminal=True,
                metadata={**branch_metadata(item, step, "human", human_p), "unsafe_probability_given_failure": human_unsafe_p, "human_branch_disclosure": "simulated_expert_reference_not_observed_human"},
            ),
            "stop": BranchOutcome(
                success=False, unsafe=False, quality_delta=-0.1, latency=0.05, execution_cost=0.02,
                terminal=True, metadata=branch_metadata(item, step, "stop"),
            ),
        }
        steps.append(PairedRepairStep(state=state, outcomes=outcomes, human_beneficial=False))
    labeled = label_human_benefit(steps, RewardConfig(), margin=0.1, gamma=0.95)
    metadata = record.get("metadata", {})
    return RepairEpisode(
        episode_id=episode_id,
        group_id=ids["group_id"],
        split="test",
        source="repository_public_api_document_corpus_replay_simulation",
        synthetic=True,
        steps=labeled,
        metadata={
            "holdout_status": "locked_holdout_unbound",
            "evidence_kind": "real_source_replayable_simulated_paired_outcomes",
            "simulation_disclosure": "source workflow is non-synthetic; all AI/Human/Stop outcomes are deterministic simulations",
            "simulator_version": SIMULATOR_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_workflow_id": ids["workflow_id"],
            "source_group_id": ids["group_id"],
            "source_document_sha256": ids["source_document_sha256"],
            "source_operation_sha256": metadata.get("operation_sha256"),
            "source_record_sha256": features["source_record_sha256"],
            "normalized_source_fingerprint": ids["normalized_source_fingerprint"],
            "failure_type": failure,
            "complexity_bin": complexity_bin,
            "complexity_score": features["complexity_score"],
            "source_features": {key: features[key] for key in ("method", "required_parameter_count", "risk_lexicon_hits", "description_character_count")},
            "generation_seeds": {"selection": SELECTION_SEED, "failure_assignment": FAILURE_SEED, "branch_construction": BRANCH_SEED, "labeling": 2026082604},
        },
    )


def ensure_unique(episodes: list[RepairEpisode]) -> dict[str, int]:
    fields: dict[str, list[str]] = {
        "episode_id": [ep.episode_id for ep in episodes],
        "workflow_id": [str(ep.metadata["source_workflow_id"]) for ep in episodes],
        "group_id": [ep.group_id for ep in episodes],
        "source_record_sha256": [str(ep.metadata["source_record_sha256"]) for ep in episodes],
        "normalized_source_fingerprint": [str(ep.metadata["normalized_source_fingerprint"]) for ep in episodes],
    }
    duplicates = {key: len(values) - len(set(values)) for key, values in fields.items()}
    if any(duplicates.values()):
        raise ValueError(f"duplicate audit failed: {duplicates}")
    return duplicates


def overlap_audit(episodes: list[RepairEpisode], historical: Mapping[str, set[str]]) -> dict[str, Any]:
    current = {
        "episode_id": {ep.episode_id for ep in episodes},
        "workflow_id": {str(ep.metadata["source_workflow_id"]) for ep in episodes},
        "group_id": {ep.group_id for ep in episodes},
        "source_document_sha256": {str(ep.metadata["source_document_sha256"]) for ep in episodes},
        "normalized_source_fingerprint": {str(ep.metadata["normalized_source_fingerprint"]) for ep in episodes},
    }
    dimensions = {}
    for key in current:
        shared = sorted(current[key] & historical[key])
        dimensions[key] = {"holdout_unique": len(current[key]), "historical_unique": len(historical[key]), "overlap_count": len(shared), "overlap_sha256": sha256_text("\n".join(shared))}
    if any(value["overlap_count"] for value in dimensions.values()):
        raise ValueError("historical overlap audit failed")
    return {"result": "PASS", "zero_overlap": True, "dimensions": dimensions}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit an unbound replayable SHARE holdout without running any method.")
    parser.add_argument("--source", type=Path, default=Path("datasets/share_public_api_workflow_input.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("datasets/share_locked_holdout_replayable_20260826"))
    parser.add_argument("--historical-index", type=Path, default=Path("datasets/share_guard_development/historical_identifiers.json"))
    parser.add_argument("--historical-episodes", type=Path, nargs="*", default=[Path("datasets/share_synthetic/episodes.jsonl"), Path("datasets/share_guard_development/episodes.jsonl"), Path("datasets/share_synthetic_augmented_guard_development/episodes.jsonl")])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.source) != SOURCE_SHA256:
        raise SystemExit("source hash mismatch; aborting before generation")
    for name, digest in (("generation_protocol_v2.json", PROTOCOL_V2_SHA256), ("generation_protocol_v3.json", PROTOCOL_V3_SHA256)):
        if sha256_file(args.out / name) != digest:
            raise SystemExit(f"sealed protocol hash mismatch: {name}")

    historical, historical_sources = historical_identifiers(args.historical_episodes, args.historical_index)
    selected, rejected = select_records(read_jsonl(args.source), historical)
    assign_failures(selected)
    episodes = [build_episode(item) for item in selected]
    episodes.sort(key=lambda episode: episode.episode_id)
    if len(episodes) != 400 or len({episode.group_id for episode in episodes}) != 400:
        raise SystemExit("episode/group target mismatch")
    if any(not episode.synthetic for episode in episodes):
        raise SystemExit("simulation disclosure failure: every episode must be synthetic=true")
    failure_counts = Counter(str(episode.metadata["failure_type"]) for episode in episodes)
    complexity_counts = Counter(str(episode.metadata["complexity_bin"]) for episode in episodes)
    if failure_counts != Counter(dict(FAILURE_COUNTS)) or complexity_counts != Counter(dict(COMPLEXITY_COUNTS)):
        raise SystemExit("preregistered distribution mismatch")
    duplicate_counts = ensure_unique(episodes)
    audit = overlap_audit(episodes, historical)

    episodes_path = args.out / "episodes.jsonl"
    write_episodes(episodes_path, episodes)
    # Reloading validates the canonical RepairEpisode schema without evaluating a policy.
    validated = load_episodes(episodes_path)
    if len(validated) != 400:
        raise SystemExit("canonical reload count mismatch")
    episodes_sha256 = sha256_file(episodes_path)

    selection_path = args.out / "source_selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(canonical_json({
                "episode_id": episode.episode_id,
                "workflow_id": episode.metadata["source_workflow_id"],
                "group_id": episode.group_id,
                "source_document_sha256": episode.metadata["source_document_sha256"],
                "source_operation_sha256": episode.metadata["source_operation_sha256"],
                "source_record_sha256": episode.metadata["source_record_sha256"],
                "normalized_source_fingerprint": episode.metadata["normalized_source_fingerprint"],
                "failure_type": episode.metadata["failure_type"],
                "complexity_bin": episode.metadata["complexity_bin"],
                "complexity_score": episode.metadata["complexity_score"],
            }) + "\n")

    report = {
        "schema_version": 1,
        "kind": "share_replayable_holdout_import_report",
        "created_at_utc": GENERATED_AT,
        "protocol_id": PROTOCOL_ID,
        "evidence_kind": "real_source_replayable_simulated_paired_outcomes",
        "source": {"path": str(args.source), "sha256": SOURCE_SHA256, "records": 3000, "groups": 884, "synthetic": False},
        "output": {"path": str(episodes_path), "sha256": episodes_sha256, "episodes": 400, "groups": 400, "split": "test", "holdout_status": "locked_holdout_unbound", "synthetic": True},
        "selection": {"unit": "workflow_group", "episodes_per_group": 1, "rejected_candidate_counts": dict(sorted(rejected.items())), "selection_index_path": str(selection_path), "selection_index_sha256": sha256_file(selection_path)},
        "distributions": {"failure_type": dict(sorted(failure_counts.items())), "complexity": dict(sorted(complexity_counts.items())), "horizon": dict(sorted(Counter(len(episode.steps) for episode in episodes).items()))},
        "duplicate_audit": {"result": "PASS", "duplicate_counts": duplicate_counts},
        "historical_overlap_audit": audit,
        "canonical_schema_reload": "PASS",
        "method_artifacts_accessed": False,
        "method_level_outcomes_accessed": False,
        "locked_evaluator_bound_or_run": False,
        "eligibility_warning": "Not eligible for locked method evaluation as real paired evidence: all branch outcomes are simulated. Obtain immutable observed AI/Human/Stop branch artifacts and use the real-paired importer before binding.",
    }
    report_path = args.out / "import_report.json"
    write_json(report_path, report)

    attestation = {
        "schema_version": 1,
        "kind": "share_prior_unseen_attestation",
        "created_at_utc": GENERATED_AT,
        "scope": "identity/provenance attestation against the declared local historical corpus; not a claim of global novelty",
        "protocol_id": PROTOCOL_ID,
        "episodes_path": str(episodes_path),
        "episodes_sha256": episodes_sha256,
        "historical_sources": historical_sources,
        "audit": audit,
        "duplicate_audit": {"result": "PASS", "duplicate_counts": duplicate_counts},
        "attested_facts": {
            "episodes": 400,
            "unique_workflow_groups": 400,
            "one_episode_per_group": True,
            "zero_workflow_group_overlap_with_declared_development_data": True,
            "zero_overlap_on_all_preregistered_dimensions": True,
            "all_source_records_non_synthetic": True,
            "all_generated_paired_outcomes_explicitly_synthetic": True,
            "method_level_outcomes_not_accessed": True,
            "locked_evaluator_not_bound_or_run": True,
        },
        "limitation": "The source cases are real public-API documentation records, but the paired outcomes are deterministic simulations. This attestation does not convert them into observed executions.",
    }
    attestation_path = args.out / "prior_unseen_attestation.json"
    write_json(attestation_path, attestation)

    manifest = {
        "schema_version": 1,
        "kind": "share_locked_holdout_data_manifest",
        "created_at_utc": GENERATED_AT,
        "status": "generated_audited_unbound",
        "protocol": {
            "active": str(args.out / "generation_protocol_v3.json"),
            "active_sha256": PROTOCOL_V3_SHA256,
            "inherited": str(args.out / "generation_protocol_v2.json"),
            "inherited_sha256": PROTOCOL_V2_SHA256,
        },
        "dataset": {"path": str(episodes_path), "sha256": episodes_sha256, "episodes": 400, "workflow_groups": 400, "synthetic": True, "source_cases_synthetic": False},
        "files": {
            "import_report.json": sha256_file(report_path),
            "prior_unseen_attestation.json": sha256_file(attestation_path),
            "source_selection.jsonl": sha256_file(selection_path),
            "generation_protocol.json": sha256_file(args.out / "generation_protocol.json"),
            "protocol_seal.json": sha256_file(args.out / "protocol_seal.json"),
            "generation_protocol_v2.json": PROTOCOL_V2_SHA256,
            "protocol_seal_v2.json": sha256_file(args.out / "protocol_seal_v2.json"),
            "generation_protocol_v3.json": PROTOCOL_V3_SHA256,
            "protocol_seal_v3.json": sha256_file(args.out / "protocol_seal_v3.json"),
        },
        "distributions": report["distributions"],
        "audit_result": "PASS_ZERO_OVERLAP",
        "evaluator_binding": {"bound": False, "run": False, "eligible_as_real_paired_holdout": False},
        "evidence_boundary": "This is an audited, group-disjoint, real-source replayable simulation dataset. It is not observed paired service-workflow evidence and cannot be reported as such.",
    }
    manifest_path = args.out / "manifest.json"
    write_json(manifest_path, manifest)

    hash_paths = sorted(path for path in args.out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in hash_paths), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "episodes": 400, "groups": 400, "overlap": 0, "synthetic": True, "evaluator_bound": False}, indent=2))


if __name__ == "__main__":
    main()
