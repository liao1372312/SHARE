from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import load_llm_profiles
from share.real_execution import EXPECTED_ACTION, seal_branch_bundle, validate_fixture
from scripts.collect_share_real_llm_executions import call_profile, request_payload, response_content


FROZEN_MODEL = "gpt-5.4-mini"
FROZEN_MAX_INPUT_CHARS = 12000
FROZEN_MAX_OUTPUT_TOKENS = 2048
FROZEN_TIMEOUT_SECONDS = 90.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def locked_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError("locked metadata index is missing")
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["episode_id"]))
    return ids


def require_collection_ready(path: Path) -> None:
    value = load_json(path)
    if value.get("status") != "READY_FOR_AUTHORIZED_COLLECTION" or value.get("all_nine_gates_pass") is not True:
        raise ValueError("real collection is blocked: readiness file does not authorize collection")


def dummy_artifact(fixture: Mapping[str, Any]) -> dict[str, Any]:
    operation = fixture["decision_time_information"]["operation"]
    params = operation.get("required_parameters", [])
    return {
        "action": EXPECTED_ACTION[str(fixture["failure_type"])],
        "target_operation": operation["operation_name"],
        "parameter_bindings": {str(parameter): f"dry-run-{index}" for index, parameter in enumerate(params, 1)},
        "safety_checks": ["controlled_testbed_only", "no_external_calls"],
        "rationale": "Deterministic dummy artifact for infrastructure validation only.",
    }


def ai_artifact_from_model(fixture: Mapping[str, Any], env_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    profiles = load_llm_profiles(env_path)
    profile = next(profile for profile in profiles if profile.name == "primary")
    if profile.model != FROZEN_MODEL:
        raise ValueError(f"primary model mismatch: expected {FROZEN_MODEL}")
    workflow = {
        "workflow_id": fixture["workflow_id"],
        "task": fixture["decision_time_information"]["task"],
        "context": {"operation": fixture["decision_time_information"]["operation"], "observed_failure": fixture["decision_time_information"]["observed_failure"]},
    }
    payload, truncated = request_payload(
        profile, workflow, FROZEN_MAX_INPUT_CHARS, FROZEN_MAX_OUTPUT_TOKENS, json_mode=True
    )
    response = call_profile(profile, payload, FROZEN_TIMEOUT_SECONDS)
    content = response_content(response.get("response", {})) if isinstance(response.get("response"), Mapping) else None
    artifact = None
    if content:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            operation = fixture["decision_time_information"]["operation"]
            artifact = {
                "action": parsed.get("action"),
                "target_operation": parsed.get("target_operation", operation["operation_name"]),
                "parameter_bindings": parsed.get("parameter_bindings", {}),
                "safety_checks": parsed.get("safety_checks", []),
                "rationale": parsed.get("rationale", ""),
            }
    operations = [{
        "sequence": 1,
        "operation": "frozen_model_request",
        "event_type": "model_call",
        "status": response.get("status"),
        "request_truncated": truncated,
        "external_service": "frozen_llm_endpoint",
        "live_public_api_called": False,
        "cleanup_verified": True,
    }]
    actor = {
        "actor_type": "ai",
        "pipeline_id": "share-public-api-autonomous-repair-controlled-adapter-v1",
        "profile": "primary",
        "model": profile.model,
        "temperature": 0.0,
        "max_output_tokens": FROZEN_MAX_OUTPUT_TOKENS,
        "max_input_chars": FROZEN_MAX_INPUT_CHARS,
        "json_mode": True,
        "response_status": response.get("status"),
    }
    return artifact, operations, actor


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated SHARE Plan B branch harness; real collection remains readiness-gated.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--branch", choices=("ai", "human", "stop"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "collect"), required=True)
    parser.add_argument("--readiness", type=Path, default=ROOT / "results/share_plan_b_phase_b2_readiness_20260826/collection_readiness.json")
    parser.add_argument("--locked-index", type=Path, default=ROOT / "datasets/share_locked_holdout_replayable_20260826/source_selection.jsonl")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--operator-record", type=Path)
    parser.add_argument("--repair-artifact", type=Path)
    parser.add_argument("--operations", type=Path)
    args = parser.parse_args()

    fixture = load_json(args.fixture)
    is_dry_run = args.mode == "dry-run"
    validate_fixture(fixture, allow_dry_run=is_dry_run)
    if is_dry_run:
        if fixture.get("dry_run") is not True:
            raise SystemExit("dry-run mode requires dry_run=true")
        if fixture["episode_id"] in locked_ids(args.locked_index):
            raise SystemExit("dry-run fixture overlaps the locked 400")
        started = "2026-08-26T02:00:00Z"
        completed = "2026-08-26T02:00:01Z"
        if args.branch == "stop":
            artifact = None
            operations = [{"sequence": 1, "operation": "stop", "state_mutated": False, "cleanup_verified": True}]
            actor = {"actor_type": "stop", "protocol_id": "share-stop-rollback-controlled-v1"}
        else:
            artifact = dummy_artifact(fixture)
            operations = [{"sequence": 1, "operation": "write_repair_artifact", "state_mutated": True, "cleanup_verified": True, "live_public_api_called": False}]
            actor = {"actor_type": args.branch, "dummy": True, "anonymized_operator_id": "operator_dryrun000001"} if args.branch == "human" else {"actor_type": "ai", "pipeline_id": "dummy-autonomous-executor-v1", "model": "deterministic-dummy"}
    else:
        require_collection_ready(args.readiness)
        if fixture.get("dry_run") is not False:
            raise SystemExit("collect mode rejects dry-run fixtures")
        started = utc_now()
        if args.branch == "ai":
            artifact, operations, actor = ai_artifact_from_model(fixture, args.env)
        elif args.branch == "stop":
            artifact = None
            operations = [{"sequence": 1, "operation": "stop", "state_mutated": False, "cleanup_verified": True}]
            actor = {"actor_type": "stop", "protocol_id": "share-stop-rollback-controlled-v1"}
        else:
            if not args.operator_record or not args.repair_artifact or not args.operations:
                raise SystemExit("human collection requires --operator-record, --repair-artifact and --operations")
            operator = load_json(args.operator_record)
            if operator.get("eligible") is not True or operator.get("consent_status") != "valid" or operator.get("ethics_status") != "covered":
                raise SystemExit("operator record does not satisfy eligibility/consent/ethics gate")
            artifact = load_json(args.repair_artifact)
            operations = [json.loads(line) for line in args.operations.read_text(encoding="utf-8").splitlines() if line.strip()]
            actor = {"actor_type": "human", "anonymized_operator_id": operator["anonymized_operator_id"], "operator_record_sha256": operator["record_sha256"]}
        completed = utc_now()
    result = seal_branch_bundle(
        args.fixture, args.out, branch=args.branch, repair_artifact=artifact, operations=operations,
        actor=actor, started_at_utc=started, completed_at_utc=completed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
