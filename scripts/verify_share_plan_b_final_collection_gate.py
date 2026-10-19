from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
B1 = ROOT / "results/share_plan_b_real_execution_prereg_20260826"
B2 = ROOT / "results/share_plan_b_phase_b2_readiness_20260826"
OUT_DEFAULT = ROOT / "results/share_plan_b_final_collection_gate_20260826"
INPUT_DEFAULT = ROOT / "inputs/share_plan_b_final_collection_gate"
VERIFIED_AT = "2026-08-26T03:00:00Z"

FROZEN = {
    "b1_subset": (B1 / "real_execution_subset.json", "187bac275f85d894c8093e0ad170656a035109817180941724fdd6c129515f88"),
    "b1_combined_seal": (B1 / "combined_precollection_seal.json", "0e54adf96acb1229639b92a785f860d9b41cfd132ea1c3767970bff2c219eb88"),
    "b1_sha256sums": (B1 / "SHA256SUMS", "28dcde276a509e4806ee5cb6464aa14b8ac964d6ab24b3f132e2ef21b6e4bb28"),
    "method_artifact_lock": (ROOT / "results/share_locked_evaluation_prereg_20260826/artifact_lock.json", "28e2f7a04a8163066d44d9770e75df40d35eadd382b6d52ac73f744d3c589727"),
    "method_freeze": (ROOT / "results/share_final_development_ablation_20pct_20260826/method_freeze.json", "ee5bd3b552b820a53ca2b6cfb1d1ecaac46be26b1b6c14050308bd7ffe2f8242"),
    "locked_data_manifest": (ROOT / "datasets/share_locked_holdout_replayable_20260826/manifest.json", "3ac8829e3bcb9b89bd1827d5ed8c83aeb4c26295191447d2336d426e9c121df1"),
    "locked_data_seal": (ROOT / "datasets/share_locked_holdout_replayable_20260826/protocol_seal_v3.json", "3f28dc001f5bc80aaf18b09ea438994a440645637155623127141f203ecb11fd"),
    "locked_episodes": (ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl", "a7e66906538a419d1be18e78120f998e3a70a8776d84897b34ffa7cdd301b74f"),
    "fixture_manifest": (B2 / "executable_fixture_manifest.json", "98692255560072c9627039b564974df1cf09e8e403381b7780bc86e3c54f2020"),
    "autonomous_executor_lock": (B2 / "autonomous_executor_lock.json", "c2e409dbc62e4369e280c8fd5e915aca9dae4ab8a8882e761a1fd172e7cd0f14"),
    "stop_executor_lock": (B2 / "stop_executor_lock.json", "fda22b430637d1aca535536a15823a1d0388b15f3d284b74754343984f9def94"),
    "rubric_json": (B2 / "acceptance_and_unsafe_rubric.json", "a136b9531295e79a08413c8ec62a1a59a4fe50edd4cebb29fb2e1d9c64634151"),
    "rubric_md": (B2 / "acceptance_and_unsafe_rubric.md", "8d17f85a56695fef84706692d064df7e2aabafb46b8978e6e9c02063bcaf1692"),
    "infrastructure_audit": (B2 / "collection_infrastructure_audit.json", "bce3186a5104e47294cb031a0ac0164eedbffdf3c53228d47012928fb4b66e53"),
    "dry_run_audit": (B2 / "dry_run_audit.json", "1d33a51495bf671983b9464c58a0f52002405ed75775d939ed5f8b8b1f94e76c"),
    "b2_readiness": (B2 / "collection_readiness.json", "26cf8adcf19b0eb9325e4dba2418a127df0114f2e8dee9d4c3fcc8ffa0a00111"),
    "b2_sha256sums": (B2 / "SHA256SUMS", "fecb2d033342c718145fb8f9bec7be6ace951766953021fbf39f9ffdcc28cb9f"),
}


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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def file_binding(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": rel(path), "exists": path.is_file()}
    if not path.is_file():
        result.update({"sha256": None, "bytes": None, "hash_matches": False if expected_sha else None})
        return result
    digest = sha256_file(path)
    result.update({"sha256": digest, "bytes": path.stat().st_size, "hash_matches": digest == expected_sha if expected_sha else None})
    return result


def verify_ethics(path: Path) -> dict[str, Any]:
    required = ["schema_version", "kind", "institution", "reviewing_body", "determination_type", "decision_id", "protocol_id", "version", "decision_date", "human_participant_collection_permitted", "document_path", "document_sha256"]
    result = {"gate": 1, "name": "institutional ethics/IRB written determination", "status": "BLOCKED", "record": file_binding(path), "required_fields": required, "missing_fields": required, "consistency": "NOT_CHECKED"}
    if not path.is_file():
        result["reason"] = "No researcher-supplied ethics/IRB determination record exists."
        return result
    try:
        value = load_json(path)
    except Exception as error:
        result.update({"reason": f"Record is not valid JSON: {error}", "missing_fields": []})
        return result
    missing = [field for field in required if value.get(field) in (None, "")]
    document = (path.parent / str(value.get("document_path", ""))).resolve()
    document_ok = document.is_file() and valid_sha(value.get("document_sha256")) and sha256_file(document) == value["document_sha256"]
    consistent = value.get("protocol_id") == "share-plan-b-real-execution-v1" and value.get("human_participant_collection_permitted") is True and value.get("determination_type") in {"approved", "exempt", "approved_with_conditions"}
    result.update({"metadata": {key: value.get(key) for key in ("institution", "reviewing_body", "determination_type", "decision_id", "protocol_id", "version", "decision_date")}, "document": file_binding(document, value.get("document_sha256") if valid_sha(value.get("document_sha256")) else None), "missing_fields": missing, "consistency": "PASS" if consistent else "FAIL"})
    if not missing and document_ok and consistent:
        result.update({"status": "PASS", "reason": "Written institutional determination and bound document verified."})
    else:
        result["reason"] = "Ethics record, bound document, permission, or protocol consistency is incomplete."
    return result


def verify_materials(path: Path, ethics: Mapping[str, Any]) -> dict[str, Any]:
    required = ["schema_version", "kind", "protocol_id", "ethics_decision_id", "approved", "participant_information", "consent_form"]
    result = {"gate": 2, "name": "final participant-information and informed-consent materials", "status": "BLOCKED", "record": file_binding(path), "required_fields": required, "missing_fields": required, "consistency": "NOT_CHECKED"}
    if not path.is_file():
        result["reason"] = "No researcher-supplied finalized/approved participant-materials record exists; Phase B2 drafts are not approvals."
        return result
    try:
        value = load_json(path)
    except Exception as error:
        result.update({"reason": f"Record is not valid JSON: {error}", "missing_fields": []})
        return result
    missing = [field for field in required if value.get(field) in (None, "")]
    bound = {}
    files_ok = True
    for key in ("participant_information", "consent_form"):
        item = value.get(key)
        item_required = ["path", "sha256", "version", "date", "approved"]
        if not isinstance(item, Mapping):
            missing.extend(f"{key}.{field}" for field in item_required)
            files_ok = False
            continue
        missing.extend(f"{key}.{field}" for field in item_required if item.get(field) in (None, ""))
        document = (path.parent / str(item.get("path", ""))).resolve()
        expected = item.get("sha256") if valid_sha(item.get("sha256")) else None
        bound[key] = {**file_binding(document, expected), "version": item.get("version"), "date": item.get("date"), "approved": item.get("approved")}
        files_ok &= document.is_file() and expected is not None and sha256_file(document) == expected and item.get("approved") is True
    ethics_id = ethics.get("metadata", {}).get("decision_id") if isinstance(ethics.get("metadata"), Mapping) else None
    consistent = value.get("protocol_id") == "share-plan-b-real-execution-v1" and value.get("approved") is True and ethics.get("status") == "PASS" and value.get("ethics_decision_id") == ethics_id
    result.update({"metadata": {key: value.get(key) for key in ("protocol_id", "ethics_decision_id", "approved")}, "documents": bound, "missing_fields": sorted(set(missing)), "consistency": "PASS" if consistent else "FAIL"})
    if not missing and files_ok and consistent:
        result.update({"status": "PASS", "reason": "Approved participant information and consent documents verified against the ethics determination."})
    else:
        result["reason"] = "Final approved materials, hashes, versions/dates, or ethics linkage are incomplete."
    return result


def verify_preregistration(path: Path) -> dict[str, Any]:
    required = ["schema_version", "kind", "registry", "registration_id", "registered_at_utc", "protocol_id", "b1_seal_sha256", "b2_readiness_sha256", "evidence_path", "evidence_sha256", "participant_outcomes_collected_before_registration"]
    result = {"gate": 3, "name": "externally timestamped preregistration", "status": "BLOCKED", "record": file_binding(path), "required_fields": required, "missing_fields": required, "consistency": "NOT_CHECKED"}
    if not path.is_file():
        result["reason"] = "No researcher-supplied external registry record or timestamp evidence exists; a local preregistration file is insufficient."
        return result
    try:
        value = load_json(path)
    except Exception as error:
        result.update({"reason": f"Record is not valid JSON: {error}", "missing_fields": []})
        return result
    missing = [field for field in required if value.get(field) in (None, "") and field != "participant_outcomes_collected_before_registration"]
    evidence = (path.parent / str(value.get("evidence_path", ""))).resolve()
    expected = value.get("evidence_sha256") if valid_sha(value.get("evidence_sha256")) else None
    evidence_ok = evidence.is_file() and expected is not None and sha256_file(evidence) == expected
    consistent = value.get("protocol_id") == "share-plan-b-real-execution-v1" and value.get("b1_seal_sha256") == FROZEN["b1_combined_seal"][1] and value.get("b2_readiness_sha256") == FROZEN["b2_readiness"][1] and value.get("participant_outcomes_collected_before_registration") is False
    result.update({"metadata": {key: value.get(key) for key in ("registry", "registration_id", "registered_at_utc", "protocol_id")}, "evidence": file_binding(evidence, expected), "missing_fields": missing, "consistency": "PASS" if consistent else "FAIL"})
    if not missing and evidence_ok and consistent:
        result.update({"status": "PASS", "reason": "External timestamp evidence and frozen protocol bindings verified."})
    else:
        result["reason"] = "External timestamp evidence, pre-outcome declaration, or frozen protocol bindings are incomplete."
    return result


def verify_roster(path: Path, ethics: Mapping[str, Any], materials: Mapping[str, Any]) -> dict[str, Any]:
    result = {"gate": 8, "name": "anonymized qualified operator roster", "status": "BLOCKED", "record": file_binding(path), "required_fields": ["schema_version", "kind", "ethics_determination_sha256", "consent_form_sha256", "roster_sealed_at_utc", "operators"], "missing_fields": ["actual roster"], "consistency": "NOT_CHECKED"}
    if not path.is_file():
        result["reason"] = "No researcher-supplied operator roster exists; the schema cannot substitute for actual eligible participants."
        return result
    try:
        value = load_json(path)
        from jsonschema import Draft202012Validator
        schema = load_json(B2 / "operator_roster_schema.json")
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    except Exception as error:
        result.update({"reason": f"Roster or schema validation failed: {error}", "missing_fields": []})
        return result
    operators = value.get("operators", []) if isinstance(value.get("operators"), list) else []
    ids = [operator.get("anonymized_operator_id") for operator in operators if isinstance(operator, Mapping)]
    direct_identifier_keys = {"name", "employer", "email", "phone", "address", "payment_account", "government_identifier"}
    direct_identifiers = sorted({key for operator in operators if isinstance(operator, Mapping) for key in operator if key.lower() in direct_identifier_keys})
    ethics_document_sha = ethics.get("document", {}).get("sha256") if isinstance(ethics.get("document"), Mapping) else None
    consent_sha = materials.get("documents", {}).get("consent_form", {}).get("sha256") if isinstance(materials.get("documents"), Mapping) else None
    consistent = ethics.get("status") == "PASS" and materials.get("status") == "PASS" and value.get("ethics_determination_sha256") == ethics_document_sha and value.get("consent_form_sha256") == consent_sha
    result.update({"metadata": {"roster_sealed_at_utc": value.get("roster_sealed_at_utc"), "operator_count": len(operators), "unique_operator_ids": len(set(ids))}, "schema_errors": [error.message for error in errors], "direct_identifier_fields": direct_identifiers, "missing_fields": [], "consistency": "PASS" if consistent else "FAIL"})
    if not errors and operators and len(set(ids)) == len(ids) and not direct_identifiers and consistent:
        result.update({"status": "PASS", "reason": "All anonymized operators satisfy the frozen eligibility schema and ethics/consent bindings."})
    else:
        result["reason"] = "Roster is empty, invalid, non-anonymous, duplicate, or inconsistent with verified ethics/consent materials."
    return result


def verify_technical() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bindings = {}
    for name, (path, expected) in FROZEN.items():
        binding = file_binding(path, expected)
        bindings[name] = binding
        if not binding["exists"] or binding["hash_matches"] is not True:
            raise ValueError(f"frozen artifact mismatch: {name}")
    manifest = load_json(FROZEN["fixture_manifest"][0])
    fixture_failures = []
    for entry in manifest.get("entries", []):
        path = ROOT / entry["path"]
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            fixture_failures.append(entry.get("episode_id"))
    ai = load_json(FROZEN["autonomous_executor_lock"][0])
    stop = load_json(FROZEN["stop_executor_lock"][0])
    rubric = load_json(FROZEN["rubric_json"][0])
    dry = load_json(FROZEN["dry_run_audit"][0])
    infrastructure = load_json(FROZEN["infrastructure_audit"][0])
    gates = [
        {"gate": 4, "name": "executable pre-decision fixtures", "status": "PASS" if not fixture_failures and manifest.get("fixtures") == 120 and manifest.get("fixture_outcomes_present") is False else "BLOCKED", "evidence": {"manifest_sha256": bindings["fixture_manifest"]["sha256"], "fixtures_verified": 120 - len(fixture_failures), "fixture_hash_failures": fixture_failures}},
        {"gate": 5, "name": "frozen autonomous AI executor", "status": "PASS" if ai.get("status") == "FROZEN_IMPLEMENTED_NOT_RUN_ON_LOCKED_SUBSET" else "BLOCKED", "evidence": {"lock_sha256": bindings["autonomous_executor_lock"]["sha256"], "execution_environment": ai.get("execution_environment")}},
        {"gate": 6, "name": "frozen Stop/rollback executor", "status": "PASS" if stop.get("status") == "FROZEN_IMPLEMENTED_NOT_RUN_ON_LOCKED_SUBSET" else "BLOCKED", "evidence": {"lock_sha256": bindings["stop_executor_lock"]["sha256"], "execution_environment": stop.get("execution_environment")}},
        {"gate": 7, "name": "acceptance tests and unsafe rubric", "status": "PASS" if rubric.get("rubric_id") == "share-acceptance-unsafe-controlled-testbed-v1" else "BLOCKED", "evidence": {"json_sha256": bindings["rubric_json"]["sha256"], "markdown_sha256": bindings["rubric_md"]["sha256"]}},
        {"gate": 9, "name": "collection infrastructure and isolation validation", "status": "PASS" if dry.get("result") == "PASS" and infrastructure.get("result") == "PASS" and dry.get("overlap_with_locked_400") == 0 and dry.get("overlap_with_locked_120") == 0 else "BLOCKED", "evidence": {"infrastructure_audit_sha256": bindings["infrastructure_audit"]["sha256"], "dry_run_audit_sha256": bindings["dry_run_audit"]["sha256"], "dummy_overlap_locked_400": dry.get("overlap_with_locked_400"), "dummy_overlap_locked_120": dry.get("overlap_with_locked_120")}},
    ]
    return gates, bindings


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def report_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Plan B Final Pre-Collection Gate Report",
        "",
        f"Status: **{audit['status']}**",
        "",
        "No participant outcomes were collected, and no SHARE/comparator/locked evaluation was run during this verification.",
        "",
        "## Gate results",
        "",
        "| Gate | Requirement | Status | Finding |",
        "|---:|---|---|---|",
    ]
    for gate in sorted(audit["gates"], key=lambda item: item["gate"]):
        finding = gate.get("reason") or "Frozen technical evidence and hashes verified."
        lines.append(f"| {gate['gate']} | {gate['name']} | {gate['status']} | {finding} |")
    lines.extend(["", "## Unresolved requirements", ""])
    if audit["unresolved"]:
        lines.extend(f"- {item}" for item in audit["unresolved"])
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Evidence boundary",
        "",
        "Drafts, schemas, local preregistration files, or researcher statements were not treated as institutional decisions, approved consent, external registry timestamps, or actual participants. Collection remains prohibited unless all nine gates pass in one immutable audit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the final SHARE Plan B pre-collection gate without running any method or collecting outcomes.")
    parser.add_argument("--inputs", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ethics = verify_ethics(args.inputs / "ethics_determination.json")
    materials = verify_materials(args.inputs / "participant_materials.json", ethics)
    prereg = verify_preregistration(args.inputs / "external_preregistration.json")
    roster = verify_roster(args.inputs / "operator_roster.json", ethics, materials)
    technical, bindings = verify_technical()
    gates = [ethics, materials, prereg, *technical, roster]
    gates.sort(key=lambda item: item["gate"])
    all_pass = all(gate["status"] == "PASS" for gate in gates)
    unresolved = [f"Gate {gate['gate']} ({gate['name']}): {gate.get('reason', 'verification did not pass')}" for gate in gates if gate["status"] != "PASS"]
    audit = {
        "schema_version": 1,
        "kind": "share_plan_b_final_collection_gate_audit",
        "verified_at_utc": VERIFIED_AT,
        "status": "READY_TO_COLLECT" if all_pass else "BLOCKED_DO_NOT_COLLECT",
        "all_nine_gates_pass": all_pass,
        "gates": gates,
        "pass_count": sum(gate["status"] == "PASS" for gate in gates),
        "blocked_count": sum(gate["status"] != "PASS" for gate in gates),
        "unresolved": unresolved,
        "researcher_supplied_input_directory": rel(args.inputs),
        "input_directory_exists": args.inputs.is_dir(),
        "technical_frozen_bindings": bindings,
        "participant_outcomes_collected": False,
        "share_or_comparators_run": False,
        "locked_evaluator_run": False,
        "method_level_locked_results_accessed": False,
        "frozen_artifacts_modified": False,
    }
    audit_path = args.out / "final_collection_gate_audit.json"
    write_json(audit_path, audit)
    report_path = args.out / "final_collection_gate_report.md"
    report_path.write_text(report_markdown(audit), encoding="utf-8")
    readiness = {
        "schema_version": 1,
        "kind": "share_plan_b_updated_collection_readiness",
        "updated_at_utc": VERIFIED_AT,
        "status": audit["status"],
        "all_nine_gates_pass": all_pass,
        "collection_authorized": all_pass,
        "gate_status": {str(gate["gate"]): gate["status"] for gate in gates},
        "pass_count": audit["pass_count"],
        "blocked_count": audit["blocked_count"],
        "unresolved": unresolved,
        "final_audit_sha256": sha256_file(audit_path),
        "previous_b2_readiness_sha256": FROZEN["b2_readiness"][1],
        "participant_outcomes_collected": False,
        "locked_evaluator_run": False,
        "next_action": "Phase B3 collection may begin only under separate authorization" if all_pass else "Supply and verify all unresolved researcher documents/records; do not collect",
    }
    readiness_path = args.out / "updated_collection_readiness.json"
    write_json(readiness_path, readiness)
    seal_binding = {
        "final_audit_sha256": sha256_file(audit_path),
        "final_report_sha256": sha256_file(report_path),
        "updated_readiness_sha256": sha256_file(readiness_path),
        "status": audit["status"],
        "all_nine_gates_pass": all_pass,
        "frozen_binding_sha256": sha256_text(canonical_json({name: binding["sha256"] for name, binding in bindings.items()})),
        "researcher_supplied_records": {
            "ethics": ethics["record"],
            "participant_materials": materials["record"],
            "external_preregistration": prereg["record"],
            "operator_roster": roster["record"],
        },
    }
    seal = {
        "schema_version": 1,
        "kind": "share_plan_b_precollection_final_seal",
        "sealed_at_utc": "2026-08-26T03:00:01Z",
        "binding": seal_binding,
        "binding_sha256": sha256_text(canonical_json(seal_binding)),
        "status": audit["status"],
        "collection_authorized": all_pass,
        "phase_b3_started": False,
    }
    seal_path = args.out / "precollection_final_seal.json"
    write_json(seal_path, seal)
    files = sorted(path for path in args.out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"out": rel(args.out), "status": audit["status"], "pass": audit["pass_count"], "blocked": audit["blocked_count"], "unresolved": unresolved}, indent=2))


if __name__ == "__main__":
    main()
