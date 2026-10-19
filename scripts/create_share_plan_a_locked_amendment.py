from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"required file missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def verify_artifact_lock(lock: dict[str, Any]) -> dict[str, Any]:
    checked = []
    for item in lock["files"]:
        path = Path(item["path"])
        got = sha256_file(path) if path.is_file() else None
        if got != item["sha256"]:
            raise SystemExit(f"frozen artifact mismatch before amendment: {path}")
        checked.append({"path": str(path), "sha256": got})
    return {
        "declared_files": len(lock["files"]),
        "verified_files": len(checked),
        "mismatches": 0,
        "binding_sha256": sha256_json(checked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and seal the Plan-A replayable-synthetic locked-evaluation amendment.")
    parser.add_argument("--out", type=Path, default=ROOT / "results/share_plan_a_locked_evaluation_20260826")
    parser.add_argument("--artifact-lock", type=Path, default=ROOT / "results/share_locked_evaluation_prereg_20260826/artifact_lock.json")
    parser.add_argument("--original-prereg", type=Path, default=ROOT / "results/share_locked_evaluation_prereg_20260826/preregistration.json")
    parser.add_argument("--original-seal", type=Path, default=ROOT / "results/share_locked_evaluation_prereg_20260826/seal.json")
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "datasets/share_locked_holdout_replayable_20260826/manifest.json")
    parser.add_argument("--episodes", type=Path, default=ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl")
    parser.add_argument("--import-report", type=Path, default=ROOT / "datasets/share_locked_holdout_replayable_20260826/import_report.json")
    parser.add_argument("--attestation", type=Path, default=ROOT / "datasets/share_locked_holdout_replayable_20260826/prior_unseen_attestation.json")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "UNBLINDING_ATTEMPT.json").exists():
        raise SystemExit("Plan-A locked evaluation has already been attempted; amendment cannot be regenerated")

    lock = json.loads(args.artifact_lock.read_text(encoding="utf-8"))
    original_prereg = json.loads(args.original_prereg.read_text(encoding="utf-8"))
    original_seal = json.loads(args.original_seal.read_text(encoding="utf-8"))
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    import_report = json.loads(args.import_report.read_text(encoding="utf-8"))
    attestation = json.loads(args.attestation.read_text(encoding="utf-8"))

    artifact_audit = verify_artifact_lock(lock)
    if sha256_file(args.artifact_lock) != original_seal["artifact_lock_sha256"]:
        raise SystemExit("original artifact lock no longer matches its seal")
    if sha256_file(args.original_prereg) != original_seal["preregistration_sha256"]:
        raise SystemExit("original preregistration no longer matches its seal")
    if data_manifest["dataset"] != {
        "path": "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl",
        "sha256": "a7e66906538a419d1be18e78120f998e3a70a8776d84897b34ffa7cdd301b74f",
        "episodes": 400,
        "workflow_groups": 400,
        "synthetic": True,
        "source_cases_synthetic": False,
    }:
        raise SystemExit("locked replay dataset metadata differs from the sealed source manifest")
    if sha256_file(args.episodes) != data_manifest["dataset"]["sha256"]:
        raise SystemExit("locked episode payload hash mismatch")
    if data_manifest.get("audit_result") != "PASS_ZERO_OVERLAP":
        raise SystemExit("source manifest does not attest zero overlap")
    audit = attestation.get("audit", {})
    if audit.get("result") != "PASS" or not audit.get("zero_overlap"):
        raise SystemExit("prior-unseen attestation does not pass")
    if not attestation.get("attested_facts", {}).get("all_generated_paired_outcomes_explicitly_synthetic"):
        raise SystemExit("attestation does not preserve synthetic=true evidence boundary")

    now = datetime.now(timezone.utc).isoformat()
    updated_manifest = {
        "schema_version": 1,
        "kind": "share_plan_a_replayable_synthetic_locked_manifest",
        "protocol_id": "share-plan-a-replayable-locked-eval-v1",
        "created_at_utc": now,
        "created_before_method_level_locked_result_access": True,
        "status": "SEALED_READY_FOR_ONE_SHOT_EVALUATION",
        "formal_definition": "a prospectively generated, previously unseen, workflow-group-disjoint, real-source replayable synthetic locked holdout derived from public-API records.",
        "dataset": {
            "path": str(args.episodes.resolve()),
            "sha256": sha256_file(args.episodes),
            "episodes": 400,
            "workflow_groups": 400,
            "synthetic": True,
            "source_cases_synthetic": False,
            "paired_branch_outcomes": "deterministic replayable AI/Human/Stop simulations",
            "actual_human_participant_execution": False,
        },
        "provenance": {
            "public_api_source_path": import_report["source"]["path"],
            "public_api_source_sha256": import_report["source"]["sha256"],
            "source_records": import_report["source"]["records"],
            "source_groups": import_report["source"]["groups"],
            "original_data_manifest": binding(args.data_manifest),
            "import_report": binding(args.import_report),
            "prior_unseen_attestation": binding(args.attestation),
            "source_selection": binding(ROOT / "datasets/share_locked_holdout_replayable_20260826/source_selection.jsonl"),
            "generation_protocol": binding(ROOT / "datasets/share_locked_holdout_replayable_20260826/generation_protocol_v3.json"),
            "generation_protocol_seal": binding(ROOT / "datasets/share_locked_holdout_replayable_20260826/protocol_seal_v3.json"),
        },
        "distribution": data_manifest["distributions"],
        "overlap_audit": {
            "result": "PASS_ZERO_OVERLAP",
            "workflow_group_overlap": audit["dimensions"]["group_id"]["overlap_count"],
            "episode_id_overlap": audit["dimensions"]["episode_id"]["overlap_count"],
            "all_preregistered_dimensions": audit["dimensions"],
            "historical_sources": attestation["historical_sources"],
            "scope": attestation["scope"],
        },
        "duplicate_audit": attestation["duplicate_audit"],
        "evidence_boundary": {
            "allowed_claim": "locked evaluation on prospectively generated, previously unseen, workflow-group-disjoint, real-source replayable synthetic episodes",
            "prohibited_claims": [
                "actual human-participant execution",
                "observed paired Human/AI/Stop service outcomes",
                "live production deployment evidence",
            ],
            "all_episodes_must_remain_synthetic_true": True,
        },
        "evaluation_binding": {"bound": True, "run": False, "one_shot": True},
    }
    manifest_path = out / "updated_locked_manifest.json"
    atomic_json(manifest_path, updated_manifest)

    amendment = {
        "schema_version": 1,
        "kind": "share_plan_a_locked_evaluation_preregistration_amendment",
        "protocol_id": "share-plan-a-replayable-locked-eval-v1",
        "amended_at_utc": now,
        "timing": "created and sealed before any method-level locked result was accessed",
        "status": "SEALED_PRE_EVALUATION",
        "reason": "Plan B human-participant collection is terminated for the current submission timeline; Plan A uses the existing replayable-synthetic locked holdout with an explicit restricted evidence claim.",
        "supersedes_only": [
            "the original requirement that locked paired outcomes have synthetic=false",
            "the original description of the holdout as externally collected real paired outcomes",
        ],
        "unchanged_frozen_plan": {
            "primary_operating_point": 0.20,
            "secondary_operating_points": [0.10, 0.30],
            "primary_methods": ["share_conf", "share_static", "share_bandit", "share_ddqn", "share"],
            "primary_metrics": original_prereg["primary_metrics"],
            "secondary_metrics": original_prereg["secondary_metrics"],
            "budget_rule": original_prereg["budget_rule"],
            "episode_order": original_prereg["episode_order"],
            "hard_cap_evaluator": original_prereg["hard_cap_evaluator"],
            "primary_comparisons": original_prereg["primary_comparisons"],
            "inference": original_prereg["inference"],
            "secondary_analyses": original_prereg["secondary_analyses"],
            "exploratory_only": original_prereg["exploratory_only"],
            "one_shot_rule": original_prereg["one_shot_rule"],
            "prohibited_actions": original_prereg["prohibited_actions"],
            "reporting_rule": original_prereg["reporting_rule"],
            "seeds": lock["seeds"],
        },
        "amended_data_contract": {
            "formal_definition": updated_manifest["formal_definition"],
            "synthetic": True,
            "source_cases_synthetic": False,
            "branch_outcomes": "deterministic replayable AI/Human/Stop simulations",
            "actual_human_participant_execution": False,
            "zero_workflow_group_overlap": True,
            "no_outcome_based_exclusions": True,
            "all_400_episodes_included": True,
        },
        "mechanism_estimand": "association between frozen predicted VoH and realized replay Human-over-AI downstream advantage",
        "plan_b_disposition": {
            "preserved": True,
            "used_in_current_evaluation": False,
            "human_outcomes_collected": False,
            "claim_of_real_human_execution": False,
            "status": "unused future-study artifacts",
        },
        "bindings": {
            "original_preregistration": binding(args.original_prereg),
            "original_seal": binding(args.original_seal),
            "artifact_lock": binding(args.artifact_lock),
            "updated_locked_manifest": binding(manifest_path),
            "plan_a_one_shot_evaluator": binding(ROOT / "scripts/run_share_plan_a_locked_holdout_once.py"),
            "plan_b_phase_b1_seal": binding(ROOT / "results/share_plan_b_real_execution_prereg_20260826/combined_precollection_seal.json"),
            "plan_b_phase_b2_readiness": binding(ROOT / "results/share_plan_b_phase_b2_readiness_20260826/collection_readiness.json"),
            "plan_b_final_blocked_seal": binding(ROOT / "results/share_plan_b_final_collection_gate_20260826/precollection_final_seal.json"),
        },
        "artifact_lock_audit": artifact_audit,
        "method_level_locked_results_accessed": False,
        "locked_evaluator_run": False,
    }
    amendment_path = out / "protocol_amendment.json"
    atomic_json(amendment_path, amendment)

    seal_bindings = {
        "protocol_amendment": binding(amendment_path),
        "updated_locked_manifest": binding(manifest_path),
        "artifact_lock": binding(args.artifact_lock),
        "original_preregistration": binding(args.original_prereg),
        "original_seal": binding(args.original_seal),
        "episodes": binding(args.episodes),
        "import_report": binding(args.import_report),
        "prior_unseen_attestation": binding(args.attestation),
        "plan_a_one_shot_evaluator": binding(ROOT / "scripts/run_share_plan_a_locked_holdout_once.py"),
    }
    combined = {
        "schema_version": 1,
        "kind": "share_plan_a_combined_pre_evaluation_seal",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEALED_READY_FOR_EXACTLY_ONE_LOCKED_EVALUATION",
        "bindings": seal_bindings,
        "combined_binding_sha256": sha256_json({key: value["sha256"] for key, value in seal_bindings.items()}),
        "artifact_lock_audit": artifact_audit,
        "data_contract": "real-source replayable synthetic; synthetic=true; no actual human-participant execution claim",
        "method_level_locked_results_accessed": False,
        "locked_evaluator_run": False,
        "collection_outcomes_used": False,
    }
    seal_path = out / "combined_pre_evaluation_seal.json"
    atomic_json(seal_path, combined)

    checksum_files = [amendment_path, manifest_path, seal_path]
    (out / "PRE_EVALUATION_SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files), encoding="utf-8"
    )
    print(json.dumps({
        "out": str(out),
        "status": combined["status"],
        "episodes": 400,
        "synthetic": True,
        "zero_overlap": True,
        "artifact_files_verified": artifact_audit["verified_files"],
    }, indent=2))


if __name__ == "__main__":
    main()
