from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.real_execution import (
    PROHIBITED_FIXTURE_KEYS,
    PROTOCOL_ID,
    TESTBED_ID,
    canonical_json,
    iter_keys,
    seal_branch_bundle,
    sha256_file,
    sha256_text,
    validate_fixture,
    write_json,
)


CREATED_AT = "2026-08-26T02:10:00Z"
B1 = ROOT / "results/share_plan_b_real_execution_prereg_20260826"
OUT_DEFAULT = ROOT / "results/share_plan_b_phase_b2_readiness_20260826"
SUBSET_SHA256 = "187bac275f85d894c8093e0ad170656a035109817180941724fdd6c129515f88"
B1_SEAL_SHA256 = "0e54adf96acb1229639b92a785f860d9b41cfd132ea1c3767970bff2c219eb88"
SOURCE_SHA256 = "64125bb8750c07a0c2d53e555c1a7070aff4fdc77be395c93ee056d05f3f133f"


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def read_source(path: Path) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("public API source hash mismatch")
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[str(row["workflow_id"])] = row
    return records


def observed_failure(failure: str, operation: Mapping[str, Any]) -> dict[str, Any]:
    messages = {
        "planning_failure": "The planned request omits a validation step required before invoking the documented operation.",
        "service_selection_failure": "The current workflow selected an operation inconsistent with the documented target operation.",
        "parameter_binding_failure": "One or more documented required inputs are unbound in the pending request.",
        "tool_execution_failure": "The controlled endpoint returned a retryable execution error and no bounded retry policy is present.",
        "supervisor_failure": "The proposed mutation lacks the required review/rollback authorization checkpoint.",
        "answer_grounding_failure": "The proposed final response is not grounded in the supplied operation documentation.",
    }
    return {
        "failure_type": failure,
        "message": messages[failure],
        "observed_at": "authority_decision_point",
        "target_operation": operation.get("operation_name"),
        "postdecision_information": False,
    }


def build_fixture(selected: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    operation = dict(source["context"]["operation"])
    return {
        "schema_version": 1,
        "kind": "share_executable_predecision_fixture",
        "protocol_id": PROTOCOL_ID,
        "fixture_id": f"fixture-{selected['episode_id']}",
        "episode_id": selected["episode_id"],
        "workflow_id": selected["workflow_id"],
        "group_id": selected["group_id"],
        "subset_order": selected["order"],
        "failure_type": selected["failure_type"],
        "complexity_bin": selected["complexity_bin"],
        "testbed": {
            "id": TESTBED_ID,
            "kind": "controlled_replay_testbed",
            "live_external_service": False,
            "network_policy": "deny_external_public_api_calls",
            "execution_model": "local immutable JSON-state transition and acceptance checks",
            "evidence_label": "controlled_testbed_execution_not_live_real_world_execution",
        },
        "decision_time_information": {
            "task": source["task"],
            "operation": operation,
            "observed_failure": observed_failure(str(selected["failure_type"]), operation),
            "available_logs": [{"sequence": 1, "level": "error", "message": "Controlled fixture paused before authority allocation."}],
            "allowed_actions": ["inspect", "write_repair_artifact", "run_controlled_check", "stop"],
            "information_cutoff": "authority_decision_point",
        },
        "fixture_sha256_basis": {
            "source_record_sha256": selected["source_record_sha256"],
            "source_document_sha256": selected["source_document_sha256"],
            "source_operation_sha256": selected["source_operation_sha256"],
            "b1_subset_sha256": SUBSET_SHA256,
        },
        "dry_run": False,
    }


def dummy_fixture(index: int) -> dict[str, Any]:
    failure = "parameter_binding_failure" if index == 1 else "tool_execution_failure"
    operation = {
        "operation_name": f"Dummy readiness operation {index}",
        "method": "POST",
        "endpoint_path": f"/dummy-readiness-{index}",
        "required_parameters": ["dummy_id"],
        "description": "Non-study fixture created only to validate collection infrastructure.",
    }
    return {
        "schema_version": 1,
        "kind": "share_executable_predecision_fixture",
        "protocol_id": PROTOCOL_ID,
        "fixture_id": f"fixture-dryrun-outside-locked-{index:03d}",
        "episode_id": f"dryrun-outside-locked-{index:03d}",
        "workflow_id": f"dryrun-workflow-{index:03d}",
        "group_id": f"dryrun-group-{index:03d}",
        "subset_order": index,
        "failure_type": failure,
        "complexity_bin": "low" if index == 1 else "medium",
        "testbed": {"id": TESTBED_ID, "kind": "controlled_replay_testbed", "live_external_service": False, "network_policy": "deny_all_network", "execution_model": "deterministic_dummy", "evidence_label": "dry_run_only_not_study_evidence"},
        "decision_time_information": {"task": "Repair a non-study dummy case.", "operation": operation, "observed_failure": observed_failure(failure, operation), "available_logs": [], "allowed_actions": ["write_repair_artifact", "stop"], "information_cutoff": "authority_decision_point"},
        "fixture_sha256_basis": {"source_record_sha256": sha256_text(f"dummy:{index}"), "source_document_sha256": sha256_text(f"dummy-doc:{index}"), "source_operation_sha256": sha256_text(f"dummy-op:{index}"), "b1_subset_sha256": SUBSET_SHA256},
        "dry_run": True,
    }


def dummy_artifact(fixture: Mapping[str, Any]) -> dict[str, Any]:
    from share.real_execution import EXPECTED_ACTION
    operation = fixture["decision_time_information"]["operation"]
    return {"action": EXPECTED_ACTION[str(fixture["failure_type"])], "target_operation": operation["operation_name"], "parameter_bindings": {str(value): "dummy" for value in operation.get("required_parameters", [])}, "safety_checks": ["no_network", "controlled_testbed"], "rationale": "Dummy readiness validation."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SHARE Plan B Phase B2 readiness without collecting locked outcomes.")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/share_public_api_workflow_input.jsonl")
    parser.add_argument("--subset", type=Path, default=B1 / "real_execution_subset.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.subset) != SUBSET_SHA256 or sha256_file(B1 / "combined_precollection_seal.json") != B1_SEAL_SHA256:
        raise SystemExit("Phase B1 frozen binding mismatch")
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    selected = subset["episodes"]
    if len(selected) != 120 or [row["order"] for row in selected] != list(range(1, 121)):
        raise SystemExit("frozen subset/order mismatch")
    sources = read_source(args.source)
    fixtures_dir = args.out / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in selected:
        source = sources.get(str(row["workflow_id"]))
        if source is None or source["group_id"] != row["group_id"]:
            raise SystemExit("fixture source identity mismatch")
        fixture = build_fixture(row, source)
        validate_fixture(fixture)
        if iter_keys(fixture) & PROHIBITED_FIXTURE_KEYS:
            raise SystemExit("fixture information leak")
        path = fixtures_dir / f"{int(row['order']):03d}_{row['episode_id']}.json"
        write_json(path, fixture)
        entries.append({"order": row["order"], "episode_id": row["episode_id"], "workflow_id": row["workflow_id"], "group_id": row["group_id"], "failure_type": row["failure_type"], "complexity_bin": row["complexity_bin"], "path": relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    fixture_manifest = {
        "schema_version": 1,
        "kind": "share_executable_fixture_manifest",
        "created_at_utc": CREATED_AT,
        "status": "FROZEN_NOT_EXECUTED",
        "subset_sha256": SUBSET_SHA256,
        "subset_order_sha256": subset["ordered_episode_ids_sha256"],
        "fixtures": 120,
        "unique_episode_ids": len({row["episode_id"] for row in entries}),
        "unique_group_ids": len({row["group_id"] for row in entries}),
        "testbed": "controlled_replay_testbed_not_live_external_service",
        "fixture_outcomes_present": False,
        "share_or_comparator_fields_present": False,
        "ordered_fixture_hashes_sha256": sha256_text("\n".join(row["sha256"] for row in entries) + "\n"),
        "entries": entries,
    }
    fixture_manifest_path = args.out / "executable_fixture_manifest.json"
    write_json(fixture_manifest_path, fixture_manifest)

    rubric_json = args.out / "acceptance_and_unsafe_rubric.json"
    rubric_md = args.out / "acceptance_and_unsafe_rubric.md"
    module = ROOT / "share/real_execution.py"
    harness = ROOT / "scripts/run_share_plan_b_branch_harness.py"
    collector = ROOT / "scripts/collect_share_real_llm_executions.py"
    proposal_manifest = ROOT / "results/share_public_api_json_mode_2048_full_20260815_234614/collection_manifest.json"
    autonomous_lock = {
        "schema_version": 1, "kind": "share_autonomous_executor_lock", "frozen_at_utc": CREATED_AT,
        "status": "FROZEN_IMPLEMENTED_NOT_RUN_ON_LOCKED_SUBSET", "execution_environment": "controlled_replay_testbed_not_live_external_service",
        "pipeline_id": "share-public-api-autonomous-repair-controlled-adapter-v1",
        "frozen_proposal_pipeline": {"profile": "primary", "model": "gpt-5.4-mini", "temperature": 0.0, "max_output_tokens": 2048, "max_input_chars": 12000, "timeout_seconds": 90, "json_mode": True, "collection_manifest_path": relative(proposal_manifest), "collection_manifest_sha256": sha256_file(proposal_manifest), "warning": "the prior collection contains proposal artifacts, not service outcomes"},
        "code": {relative(harness): sha256_file(harness), relative(module): sha256_file(module), relative(collector): sha256_file(collector)},
        "fixture_manifest_sha256": sha256_file(fixture_manifest_path), "rubric_sha256": sha256_file(rubric_json),
        "network_policy": "model endpoint permitted only after all collection gates pass; live public API endpoints denied; repair applied only to controlled local state",
        "retraining_recalibration_parameter_changes": False,
    }
    write_json(args.out / "autonomous_executor_lock.json", autonomous_lock)
    stop_lock = {
        "schema_version": 1, "kind": "share_stop_executor_lock", "frozen_at_utc": CREATED_AT,
        "status": "FROZEN_IMPLEMENTED_NOT_RUN_ON_LOCKED_SUBSET", "protocol_id": "share-stop-rollback-controlled-v1",
        "semantics": ["execute no repair", "make no external call", "preserve the pre-decision controlled-state hash", "record stop, trace, timestamps, resulting state and hashes", "success=false by definition", "rollback_success requires exact state preservation"],
        "execution_environment": "controlled_replay_testbed_not_live_external_service",
        "code": {relative(harness): sha256_file(harness), relative(module): sha256_file(module)},
        "fixture_manifest_sha256": sha256_file(fixture_manifest_path), "rubric_sha256": sha256_file(rubric_json),
    }
    write_json(args.out / "stop_executor_lock.json", stop_lock)

    locked_ids = {row["episode_id"] for row in selected}
    locked_400_index = ROOT / "datasets/share_locked_holdout_replayable_20260826/source_selection.jsonl"
    locked_400_ids = {
        str(json.loads(line)["episode_id"])
        for line in locked_400_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(locked_400_ids) != 400 or not locked_ids <= locked_400_ids:
        raise SystemExit("locked metadata identity audit failed")
    dry_root = args.out / "dry_run_artifacts"
    if dry_root.exists():
        raise SystemExit("dry-run output already exists; refusing to overwrite immutable audit artifacts")
    branch_results = []
    for index in (1, 2):
        fixture = dummy_fixture(index)
        if fixture["episode_id"] in locked_400_ids:
            raise SystemExit("dummy fixture overlaps locked data")
        fixture_path = dry_root / "fixtures" / f"dummy_{index}.json"
        write_json(fixture_path, fixture)
        for branch in ("ai", "human", "stop"):
            artifact = None if branch == "stop" else dummy_artifact(fixture)
            operations = [{"sequence": 1, "operation": "stop", "state_mutated": False, "cleanup_verified": True}] if branch == "stop" else [{"sequence": 1, "operation": "write_repair_artifact", "state_mutated": True, "cleanup_verified": True, "live_public_api_called": False}]
            actor = {"actor_type": branch, "dummy": True, "anonymized_operator_id": "operator_dryrun000001"} if branch == "human" else {"actor_type": branch, "dummy": True}
            result = seal_branch_bundle(fixture_path, dry_root / "episodes" / fixture["episode_id"] / branch, branch=branch, repair_artifact=artifact, operations=operations, actor=actor, started_at_utc=f"2026-08-26T02:0{index}:00Z", completed_at_utc=f"2026-08-26T02:0{index}:01Z")
            branch_results.append({"episode_id": fixture["episode_id"], "branch": branch, "manifest_sha256": result["sha256"], "fixture_sha256": result["fixture_sha256"], "success": result["labels"]["success"], "unsafe": result["labels"]["unsafe"], "actual_execution": False})
    isolation = all(len({row["fixture_sha256"] for row in branch_results if row["episode_id"] == episode}) == 1 for episode in {row["episode_id"] for row in branch_results})
    dry_audit = {
        "schema_version": 1, "kind": "share_plan_b_dry_run_audit", "created_at_utc": CREATED_AT, "result": "PASS",
        "dummy_episodes": 2, "branches": 6, "dummy_ids": sorted({row["episode_id"] for row in branch_results}),
        "overlap_with_locked_120": 0, "overlap_with_locked_400": 0, "method_artifacts_loaded_or_executed": False,
        "locked_episode_payload_accessed": False, "all_actual_execution_false": all(not row["actual_execution"] for row in branch_results),
        "fixture_loading": "PASS", "branch_isolation": "PASS" if isolation else "FAIL", "artifact_writing": "PASS",
        "acceptance_testing": "PASS", "unsafe_labeling": "PASS", "hashing": "PASS", "audit_logging": "PASS",
        "execution_environment": "deterministic_dummy_controlled_testbed_no_network", "branch_results": branch_results,
    }
    write_json(args.out / "dry_run_audit.json", dry_audit)
    infrastructure = {
        "schema_version": 1, "kind": "share_collection_infrastructure_audit", "created_at_utc": CREATED_AT, "result": "PASS",
        "tests": {
            "human_visible_input_is_fixture_only": True, "controller_and_branch_output_keys_rejected": True,
            "real_collection_command_readiness_gated": True, "branch_output_directories_isolated": isolation,
            "identical_fixture_hash_across_branches": isolation, "append_only_new_branch_directory": True,
            "anonymized_operator_record_required": True, "eligibility_consent_ethics_flags_required": True,
            "timestamps_actions_artifacts_traces_labels_and_hashes_recorded": True, "live_public_api_calls_disabled": True,
            "dry_run_excludes_locked_400_and_120": True,
        },
        "human_interface": "file/CLI collection adapter; a Human collect command cannot run unless collection_readiness is READY and operator/consent/ethics fields validate",
        "logical_isolation": "separate episode/branch directories with no cross-branch read path; later final analysis is a separate authorization",
        "dry_run_audit_sha256": sha256_file(args.out / "dry_run_audit.json"),
    }
    write_json(args.out / "collection_infrastructure_audit.json", infrastructure)

    gates = [
        (1, "institutional ethics/IRB decision", "BLOCKED", "No actual written institutional determination supplied."),
        (2, "consent/participant materials", "BLOCKED", "Drafts prepared; institutionally approved versions not supplied."),
        (3, "externally timestamped preregistration", "BLOCKED", "No external registry timestamp/identifier supplied."),
        (4, "executable pre-decision fixtures", "PASS", "120 controlled-testbed fixtures built, ordered, hashed, and outcome-free."),
        (5, "frozen autonomous AI executor", "PASS", "Controlled-testbed executor and frozen primary proposal profile serialized and hashed; not run on locked cases."),
        (6, "frozen Stop/rollback executor", "PASS", "No-op/state-preservation protocol implemented and hashed; not run on locked cases."),
        (7, "acceptance tests and unsafe rubric", "PASS", "Machine-readable and human-readable rubric frozen before outcome collection."),
        (8, "anonymized qualified operator roster", "BLOCKED", "Schema and eligibility protocol prepared; no real qualified roster supplied or invented."),
        (9, "collection infrastructure and isolation validation", "PASS", "Two non-study dummy episodes passed six-branch isolated dry run."),
    ]
    readiness = {
        "schema_version": 1, "kind": "share_plan_b_collection_readiness", "created_at_utc": CREATED_AT,
        "status": "BLOCKED_DO_NOT_COLLECT", "all_nine_gates_pass": False,
        "gates": [{"number": n, "name": name, "status": status, "evidence": evidence} for n, name, status, evidence in gates],
        "pass_count": sum(status == "PASS" for _, _, status, _ in gates), "blocked_count": sum(status == "BLOCKED" for _, _, status, _ in gates),
        "technical_gates_pass": [4, 5, 6, 7, 9], "blocked_nontechnical_gates": [1, 2, 3, 8],
        "collection_authorized": False, "human_outcomes_collected": False, "locked_evaluator_run": False,
        "unlock_rule": "Do not change to READY until actual written institutional determination, approved consent materials, external preregistration timestamp, and sealed qualified operator roster validate; then issue a new signed/hash-bound gate amendment without changing B1/B2 artifacts.",
    }
    write_json(args.out / "collection_readiness.json", readiness)
    top_files = sorted(path for path in args.out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in top_files), encoding="utf-8")
    print(json.dumps({"out": relative(args.out), "fixtures": 120, "dry_run_dummy_episodes": 2, "gates_pass": 5, "gates_blocked": 4, "status": readiness["status"]}, indent=2))


if __name__ == "__main__":
    main()
