from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from share import proposal_annotation as annotation


ROOT = Path(__file__).resolve().parents[1]
PREANNOTATION_SPEC = importlib.util.spec_from_file_location(
    "share_proposal_preannotation", ROOT / "scripts" / "preannotate_share_proposal_study.py"
)
assert PREANNOTATION_SPEC is not None and PREANNOTATION_SPEC.loader is not None
preannotation_script = importlib.util.module_from_spec(PREANNOTATION_SPEC)
PREANNOTATION_SPEC.loader.exec_module(preannotation_script)
SIMULATION_SPEC = importlib.util.spec_from_file_location(
    "share_simulated_expert_assessments", ROOT / "scripts" / "simulate_share_expert_assessments.py"
)
assert SIMULATION_SPEC is not None and SIMULATION_SPEC.loader is not None
simulation_script = importlib.util.module_from_spec(SIMULATION_SPEC)
SIMULATION_SPEC.loader.exec_module(simulation_script)
SIMULATION_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "share_simulated_expert_analysis", ROOT / "scripts" / "analyze_share_simulated_expert_assessments.py"
)
assert SIMULATION_ANALYZER_SPEC is not None and SIMULATION_ANALYZER_SPEC.loader is not None
simulation_analyzer = importlib.util.module_from_spec(SIMULATION_ANALYZER_SPEC)
SIMULATION_ANALYZER_SPEC.loader.exec_module(simulation_analyzer)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "share_proposal_analysis", ROOT / "scripts" / "analyze_share_proposal_annotations.py"
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
analysis_script = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(analysis_script)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_collection(root: Path, workflows: int = 1) -> Path:
    collection = root / "collection"
    artifacts = collection / "artifacts"
    artifacts.mkdir(parents=True)
    collection_id = "test-collection"
    profiles = [
        ("primary", "model-primary"),
        ("backbone_2", "model-second"),
        ("backbone_3", "model-third"),
    ]
    (collection / "collection_manifest.json").write_text(
        json.dumps({"kind": "share_real_llm_execution_collection", "collection_id": collection_id, "profiles": [{"name": name} for name, _model in profiles]}),
        encoding="utf-8",
    )
    index = []
    for workflow_number in range(1, workflows + 1):
        workflow = {
            "workflow_id": f"workflow-{workflow_number}",
            "group_id": f"group-{workflow_number}",
            "task": "Review this documented API operation.",
            "context": {
                "operation": {
                    "service_name": "Sample Service",
                    "operation_name": f"Find records {workflow_number}",
                    "method": "GET",
                    "endpoint_path": f"/records/{workflow_number}",
                    "description": "Public documentation at https://private.example.test/docs",
                    "required_parameters": [
                        {"name": "query", "type": "string", "description": "Search input"},
                        {"name": "api_key", "type": "string", "description": "must be hidden"},
                    ],
                }
            },
        }
        for name, model in profiles:
            key = f"key-{workflow_number}-{name}"
            artifact = {
                "kind": "share_real_llm_execution_artifact",
                "collection_id": collection_id,
                "workflow": workflow,
                "profile": {"name": name, "model": model},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "action": "Review inputs",
                                        "rationale": "Use https://private.example.test and api_key=secret-value only after validation.",
                                        "assumptions": ["The query input is available."],
                                        "confidence": 0.7,
                                    }
                                )
                            }
                        }
                    ]
                },
            }
            path = artifacts / f"{key}.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            index.append(
                {
                    "execution_key": key,
                    "workflow_id": workflow["workflow_id"],
                    "profile": name,
                    "status": "completed",
                    "execution_artifact_file": f"artifacts/{key}.json",
                    "execution_artifact_sha256": sha256_file(path),
                }
            )
    (collection / "execution_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index), encoding="utf-8")
    return collection


def complete_response(case: dict) -> dict:
    return {
        "proposal_reviews": {
            proposal["label"]: {"feasibility": 4, "safety": 4, "input_coverage": 3, "verdict": "minor_revision"}
            for proposal in case["proposals"]
        },
        "preferred_proposal": "a",
        "engineer_review": "yes",
        "case_confidence": 4,
        "comment": "The recommendation needs an engineer review before use.",
        "attested_documentation_only": True,
    }


def analysis_row(case_id: str, annotator_id: str, *, assisted: bool = False) -> dict:
    response = {
        "proposal_reviews": {
            "A": {"feasibility": 4, "safety": 4, "input_coverage": 4, "verdict": "minor_revision"},
            "B": {"feasibility": 3, "safety": 3, "input_coverage": 3, "verdict": "major_revision"},
            "C": {"feasibility": 2, "safety": 2, "input_coverage": 2, "verdict": "reject"},
        },
        "preferred_proposal": "A",
        "engineer_review": "yes",
        "case_confidence": 4,
        "llm_assisted_expert_audit": {"mode": "ephemeral_on_demand_reference"} if assisted else None,
    }
    return {
        "kind": "share_proposal_assessment_export_row",
        "assessment_scope": annotation.ASSESSMENT_SCOPE,
        "study_id": "analysis-study",
        "collection_id": "analysis-collection",
        "annotator_id": annotator_id,
        "case": {"case_id": case_id},
        "model_by_blind_label": {
            "A": {"model": "model-a"},
            "B": {"model": "model-b"},
            "C": {"model": "model-c"},
        },
        "response": response,
    }


def test_analysis_separates_manifested_calibration_from_all_human_only_assessments():
    rows = [
        analysis_row("calibration-case", "expert_a"),
        analysis_row("calibration-case", "expert_b"),
        analysis_row("ordinary-case", "expert_a"),
        analysis_row("ordinary-case", "expert_b", assisted=True),
    ]

    report = analysis_script.summary(rows, {"case_ids": {"calibration-case"}})

    assert report["annotation_mode_counts"] == {"human_only": 3, "llm_assisted_expert_audit": 1}
    agreement = report["interrater_agreement"]
    assert "human_only_calibration" not in agreement
    assert agreement["all_human_only_completed_assessments"]["completed_assessments"] == 3
    calibration = agreement["manifested_human_only_calibration"]
    assert calibration["manifest_cases"] == 1
    assert calibration["cases_with_completed_assessments"] == 1
    assert calibration["human_only_completed_assessments"] == 2
    assert calibration["preferred_choice"]["pairs"] == 1


def preannotation_response(case: dict) -> dict:
    response = complete_response(case)
    response.pop("attested_documentation_only")
    return response


def write_annotation_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "LLM_ANNOTATION_BASE_URL=https://annotation.example.test/v1",
                "LLM_ANNOTATION_API_KEY=annotation-secret-key",
                "LLM_ANNOTATION_MODEL=strong-judge-model",
                "LLM_ANNOTATION_REASONING_EFFORT=high",
            ]
        ),
        encoding="utf-8",
    )


def preannotation_args(study: Path, env: Path, out: Path, **overrides):
    values = {
        "study": study,
        "calibration_manifest": None,
        "env": env,
        "out": out,
        "max_cases": None,
        "max_output_tokens": 256,
        "timeout_seconds": 1,
        "resume": False,
        "retry_failed": False,
        "retry_uncertain": False,
        "max_rate_limit_retries": 0,
        "rate_limit_backoff_seconds": 1.0,
        "progress": False,
        "progress_every": 1,
        "dry_run": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def simulation_args(study: Path, env: Path, out: Path, **overrides):
    values = {
        "study": study,
        "env": env,
        "out": out,
        "simulation_id": "test-simulated-experts-v1",
        "max_cases": None,
        "simulated_reviewers_per_case": 2,
        "max_output_tokens": 256,
        "timeout_seconds": 1,
        "resume": False,
        "retry_failed": False,
        "retry_uncertain": False,
        "max_rate_limit_retries": 0,
        "rate_limit_backoff_seconds": 1.0,
        "progress": False,
        "progress_every": 1,
        "dry_run": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def write_calibration_manifest(study: Path, case_ids: list[str], path: Path, *, required_ratings: int = 2) -> Path:
    store = annotation.StudyStore(study)
    metadata = store.metadata()
    available = {item["case"]["case_id"]: item for item in store.cases_for_preannotation()}
    value = {
        "schema_version": annotation.CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "kind": annotation.CALIBRATION_MANIFEST_KIND,
        "study_id": metadata["study_id"],
        "collection_id": metadata["collection_id"],
        "required_human_only_ratings_per_case": required_ratings,
        "cases": [
            {
                "case_id": case_id,
                "group_id": available[case_id]["case"]["group_id"],
                "case_sha256": available[case_id]["case_sha256"],
            }
            for case_id in case_ids
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_study_blinds_models_sanitizes_cases_and_locks_submissions(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"

    created = annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=2,
        seed=17,
    )

    assert created["cases"] == 1
    assert created["assignments"] == 2
    assert (database.stat().st_mode & 0o777) == 0o600
    backup = tmp_path / "backups" / "study-backup.sqlite"
    annotation.backup_study_database(database, backup)
    assert (backup.stat().st_mode & 0o777) == 0o600
    assert annotation.StudyStore(backup).metadata()["study_id"] == "public-api-pilot-v1"
    store = annotation.StudyStore(database)
    assert store.authenticate("expert_a", created["credentials"]["expert_a"]) == {"user_id": "expert_a", "role": "annotator"}
    assignment = store.next_assignment("expert_a")
    assert assignment is not None
    case = store.assignment_case(assignment, "expert_a")
    encoded_case = annotation.canonical_json(case)
    assert "model-primary" not in encoded_case
    assert "model-second" not in encoded_case
    assert "model-third" not in encoded_case
    assert "private.example.test" not in encoded_case
    assert "secret-value" not in encoded_case
    assert "api_key" not in encoded_case

    store.save_draft(
        assignment,
        "expert_a",
        {
            "proposal_A_feasibility": "5",
            "comment": "A provisional note with api_key=secret-value.",
            "csrf": "must-not-be-persisted",
        },
    )
    restarted_store = annotation.StudyStore(database)
    restored = restarted_store.assignment_draft(assignment, "expert_a")
    assert restored["proposal_A_feasibility"] == "5"
    assert "secret-value" not in restored["comment"]
    assert "csrf" not in restored

    store.submit(assignment, "expert_a", complete_response(case))
    assert store.annotate_progress("expert_a") == {"pending": 0, "in_progress": 0, "completed": 1, "total": 1}
    assert store.assignment_draft(assignment, "expert_a") == {}
    with pytest.raises(ValueError, match="already been submitted"):
        store.submit(assignment, "expert_a", complete_response(case))
    exported = list(store.export_completed_rows())
    assert len(exported) == 1
    assert {binding["model"] for binding in exported[0]["model_by_blind_label"].values()} == {
        "model-primary",
        "model-second",
        "model-third",
    }


def test_response_requires_documentation_only_attestation(tmp_path):
    collection = write_collection(tmp_path)
    _collection_id, cases = annotation.load_collection_cases(collection)
    case, _mapping = annotation.blinded_case(cases[0], "pilot", 17)
    response = complete_response(case)
    response["attested_documentation_only"] = False

    with pytest.raises(ValueError, match="must confirm"):
        annotation.validate_response(response, case)


def test_simulated_assessment_casebook_has_blinded_cases_but_no_human_records(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "simulation-casebook.sqlite"

    created = annotation.create_simulated_assessment_casebook(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-llm-simulation-v1",
        max_workflows=1,
        seed=17,
    )

    assert created["cases"] == 1
    assert created["users"] == 0
    assert created["assignments"] == 0
    assert (database.stat().st_mode & 0o777) == 0o600
    store = annotation.StudyStore(database, migrate=False)
    assert store.metadata()["kind"] == "share_llm_simulated_assessment_casebook"
    assert len(store.cases_for_preannotation()) == 1
    assert len(store.case_model_mappings()) == 1
    with store.connection() as connection:
        assert int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]) == 0
        assert int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]) == 0


def test_add_annotator_preserves_existing_study_and_can_assign_all_cases(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"
    created = annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=1,
        seed=17,
    )
    initial_case = annotation.StudyStore(database).cases_for_preannotation()[0]

    added = annotation.add_annotator(database, annotator_id="expert_b", assign_all_cases=True)

    assert added["annotator_id"] == "expert_b"
    assert added["assignments"] == 1
    store = annotation.StudyStore(database)
    assert store.authenticate("expert_a", created["credentials"]["expert_a"]) == {"user_id": "expert_a", "role": "annotator"}
    assert store.authenticate("expert_b", added["password"]) == {"user_id": "expert_b", "role": "annotator"}
    assert store.annotate_progress("expert_a") == {"pending": 1, "in_progress": 0, "completed": 0, "total": 1}
    assert store.annotate_progress("expert_b") == {"pending": 1, "in_progress": 0, "completed": 0, "total": 1}
    assert store.cases_for_preannotation()[0] == initial_case


def test_human_only_calibration_assignments_are_prioritized_without_prefilling(tmp_path):
    collection = write_collection(tmp_path, workflows=2)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=2,
        ratings_per_workflow=2,
        seed=17,
    )
    store = annotation.StudyStore(database)
    calibration_case = store.cases_for_preannotation()[-1]["case"]
    manifest = write_calibration_manifest(
        database,
        [calibration_case["case_id"]],
        tmp_path / "calibration_manifest.json",
    )
    calibration = annotation.load_human_only_calibration_manifest(manifest, store)
    assert calibration["complete"] is False
    prioritized = store.prioritize_human_only_calibration(
        calibration["case_ids"],
        required_human_only_ratings=calibration["required_human_only_ratings_per_case"],
    )
    assert set(prioritized) == {"expert_a", "expert_b"}
    for annotator_id in prioritized:
        assignment = store.next_assignment(annotator_id)
        assert assignment is not None
        assert store.assignment_case(assignment, annotator_id)["case_id"] == calibration_case["case_id"]
    assert store.preannotation_summary()["not_started"] == 2


def test_llm_defaults_are_editable_not_human_drafts_and_require_expert_confirmation(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"
    created = annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=2,
        seed=17,
    )
    store = annotation.StudyStore(database)
    assignment = store.next_assignment("expert_a")
    assert assignment is not None
    case = store.assignment_case(assignment, "expert_a")
    profile = annotation.LLMProfile("annotation_judge", "https://annotation.example.test/v1", "secret", "strong-judge-model", "high")
    request = annotation.preannotation_request(profile, case, 256)
    store.begin_preannotation(
        case["case_id"],
        profile_binding=annotation.annotation_profile_binding(profile),
        prompt_sha256=annotation.sha256_text(annotation.canonical_json(request["messages"])),
        request_sha256=annotation.sha256_text(annotation.canonical_json(request)),
    )
    defaults = annotation.validate_preannotation(preannotation_response(case), case)
    store.finish_preannotation(case["case_id"], status="completed", preannotation=defaults, response_sha256="response-hash", latency_ms=1.0)

    values, preannotation_present = store.assignment_initial_values(assignment, "expert_a")
    assert preannotation_present is True
    assert values["proposal_A_feasibility"] == "4"
    assert "attested_documentation_only" not in values
    assert store.assignment_draft(assignment, "expert_a") == {}
    html = annotation.render_task({"user_id": "expert_a", "role": "annotator"}, "csrf", assignment, case, values=values, llm_preannotation_present=True)
    assert "已提供 LLM 预标注默认值" in html
    assert "我已逐项审核 LLM 预标注" in html

    with pytest.raises(ValueError, match="reviewed the LLM default"):
        store.submit(assignment, "expert_a", complete_response(case))
    accepted = complete_response(case)
    accepted["llm_preannotation_reviewed"] = True
    accepted["proposal_reviews"]["A"]["safety"] = 5
    store.submit(assignment, "expert_a", accepted)
    exported = list(store.export_completed_rows())
    response = exported[0]["response"]
    assert response["llm_assisted_expert_audit"]["reviewed_by_expert"] is True
    assert "proposal_A_safety" in response["llm_assisted_expert_audit"]["changed_fields"]
    assert exported[0]["llm_preannotation"]["profile_binding"]["model"] == "strong-judge-model"
    assert created["credentials"]["expert_a"] not in annotation.canonical_json(exported[0])


def test_ephemeral_llm_reference_is_visible_to_the_expert_but_not_stored(tmp_path, monkeypatch):
    collection = write_collection(tmp_path, workflows=2)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=2,
        ratings_per_workflow=2,
        seed=17,
    )
    store = annotation.StudyStore(database)
    calibration_assignment = store.next_assignment("expert_a")
    assert calibration_assignment is not None
    calibration_case = store.assignment_case(calibration_assignment, "expert_a")
    store.submit(calibration_assignment, "expert_a", complete_response(calibration_case))
    with store.connection() as connection:
        second_calibration_assignment = connection.execute(
            "SELECT assignment_id FROM assignments WHERE case_id = ? AND annotator_id = ?",
            (calibration_case["case_id"], "expert_b"),
        ).fetchone()["assignment_id"]
    second_calibration_case = store.assignment_case(second_calibration_assignment, "expert_b")
    store.submit(second_calibration_assignment, "expert_b", complete_response(second_calibration_case))
    calibration_manifest = write_calibration_manifest(
        database,
        [calibration_case["case_id"]],
        tmp_path / "calibration_manifest.json",
    )
    profile = annotation.LLMProfile("annotation_judge", "https://annotation.example.test/v1", "secret", "strong-judge-model", "high")
    application = annotation.AnnotationWebApplication(
        store,
        llm_reference_profile=profile,
        calibration_manifest_path=calibration_manifest,
        llm_reference_max_calls=1,
        llm_reference_max_output_tokens=256,
        llm_reference_timeout_seconds=1,
    )
    reference_assignment = store.next_assignment("expert_a")
    assert reference_assignment is not None
    reference_case = store.assignment_case(reference_assignment, "expert_a")
    assert reference_case["case_id"] != calibration_case["case_id"]
    html = annotation.render_task(
        {"user_id": "expert_a", "role": "annotator"},
        "csrf-token",
        reference_assignment,
        reference_case,
        llm_reference_enabled=True,
    )
    assert 'id="request-llm-reference"' in html
    assert f"/task/{reference_assignment}/llm-reference" in html

    monkeypatch.setattr(
        annotation,
        "call_annotation_profile",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "preannotation_response": preannotation_response(reference_case),
        },
    )
    result = application.request_llm_reference(reference_assignment, "expert_a")

    assert result["fields"]["proposal_A_feasibility"] == "4"
    assert result["calls_remaining"] == 0
    assert store.case_preannotation(reference_case["case_id"]) is None
    assert store.preannotation_summary()["not_started"] == 2
    enabled, reason = application.llm_reference_status(reference_assignment, "expert_a", reference_case["case_id"])
    assert enabled is False
    assert reason == "本案例的 LLM 参考次数已用完。"
    calibration_enabled, calibration_reason = application.llm_reference_status(
        calibration_assignment,
        "expert_a",
        calibration_case["case_id"],
    )
    assert calibration_enabled is False
    assert calibration_reason == "该案例属于纯人工校准集，不显示 LLM 参考。"

    store.submit(reference_assignment, "expert_a", complete_response(reference_case))
    response = next(row["response"] for row in store.export_completed_rows() if row["assignment_id"] == reference_assignment)
    audit = response["llm_assisted_expert_audit"]
    assert audit["mode"] == "ephemeral_on_demand_reference"
    assert audit["reference_calls_presented"] == 1
    assert "fields" not in audit
    assert "defaults" not in audit
    with store.connection() as connection:
        details = "\n".join(row[0] for row in connection.execute("SELECT detail_json FROM audit_events"))
    assert "Review inputs" not in details


def test_preannotation_runner_calls_one_judge_per_case_and_keeps_secrets_out_of_reports(tmp_path, monkeypatch):
    collection = write_collection(tmp_path, workflows=2)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=2,
        ratings_per_workflow=2,
        seed=17,
    )
    env = tmp_path / ".env"
    out = tmp_path / "preannotations"
    write_annotation_env(env)
    store = annotation.StudyStore(database)
    calibration_assignment = store.next_assignment("expert_a")
    assert calibration_assignment is not None
    calibration_case = store.assignment_case(calibration_assignment, "expert_a")
    store.submit(calibration_assignment, "expert_a", complete_response(calibration_case))
    with store.connection() as connection:
        second_assignment = connection.execute(
            "SELECT assignment_id FROM assignments WHERE case_id = ? AND annotator_id = ?",
            (calibration_case["case_id"], "expert_b"),
        ).fetchone()["assignment_id"]
    calibration_manifest = write_calibration_manifest(
        database,
        [calibration_case["case_id"]],
        tmp_path / "calibration_manifest.json",
    )
    calls = []

    def fake_call(_profile, _request, _timeout, **_kwargs):
        calls.append(1)
        case = next(
            item["case"]
            for item in annotation.StudyStore(database).cases_for_preannotation()
            if item["case"]["case_id"] != calibration_case["case_id"]
        )
        return (
            {
                "status": "completed",
                "latency_ms": 1.0,
                "response_sha256": "response-hash",
                "preannotation_response": preannotation_response(case),
            },
            0,
            0.0,
        )

    monkeypatch.setattr(preannotation_script, "call_annotation_with_rate_limit_retries", fake_call)
    with pytest.raises(ValueError, match="human-only calibration is incomplete"):
        preannotation_script.run_preannotation(
            preannotation_args(database, env, out, calibration_manifest=calibration_manifest)
        )
    second_case = store.assignment_case(second_assignment, "expert_b")
    assert second_case["case_id"] == calibration_case["case_id"]
    store.submit(second_assignment, "expert_b", complete_response(second_case))
    report = preannotation_script.run_preannotation(
        preannotation_args(database, env, out, calibration_manifest=calibration_manifest)
    )

    assert calls == [1]
    assert report["summary"]["completed"] == 1
    assert report["prefillable_unsubmitted_assignments"] == 2
    manifest = (out / "preannotation_manifest.json").read_text(encoding="utf-8")
    assert "annotation-secret-key" not in manifest
    assert "annotation.example.test" not in manifest
    assert "strong-judge-model" in manifest
    store = annotation.StudyStore(database)
    assert store.preannotation_summary()["completed"] == 1

    dry_out = tmp_path / "dry"
    monkeypatch.setattr(
        preannotation_script,
        "call_annotation_with_rate_limit_retries",
        lambda *_args, **_kwargs: pytest.fail("dry run contacted network"),
    )
    dry_report = preannotation_script.run_preannotation(
        preannotation_args(database, env, dry_out, calibration_manifest=calibration_manifest, dry_run=True)
    )
    assert dry_report["summary"] == {"planned": 1}


def test_simulated_expert_runner_generates_two_llm_assessments_without_touching_human_assignments(tmp_path, monkeypatch):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a", "expert_b"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=2,
        seed=17,
    )
    env = tmp_path / ".env"
    out = tmp_path / "simulation"
    write_annotation_env(env)
    calls = []

    def fake_call(_profile, request, _timeout, **_kwargs):
        calls.append(request["messages"][1]["content"])
        case = annotation.StudyStore(database).cases_for_preannotation()[0]["case"]
        return (
            {
                "status": "completed",
                "latency_ms": 1.0,
                "response_sha256": "response-hash-" + str(len(calls)),
                "preannotation_response": preannotation_response(case),
            },
            0,
            0.0,
        )

    monkeypatch.setattr(simulation_script, "call_annotation_with_rate_limit_retries", fake_call)
    report = simulation_script.run_simulation(simulation_args(database, env, out))

    assert len(calls) == 2
    assert "Simulated reviewer cohort slot: 1" in calls[0]
    assert "Simulated reviewer cohort slot: 2" in calls[1]
    assert report["summary"] == {"started": 0, "completed": 2, "failed": 0, "uncertain": 0, "skipped": 0}
    assert report["caps"]["max_llm_requests"] == 2
    records = [json.loads(line) for line in (out / "simulated_assessments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {record["reviewer_slot"] for record in records} == {1, 2}
    assert all(record["kind"] == "share_llm_simulated_expert_assessment" for record in records)
    assert "annotation-secret-key" not in (out / "simulation_manifest.json").read_text(encoding="utf-8")
    assert "annotation.example.test" not in (out / "simulation_manifest.json").read_text(encoding="utf-8")
    store = annotation.StudyStore(database)
    assert store.annotate_progress("expert_a") == {"pending": 1, "in_progress": 0, "completed": 0, "total": 1}
    assert store.annotate_progress("expert_b") == {"pending": 1, "in_progress": 0, "completed": 0, "total": 1}
    analysis = simulation_analyzer.summary(records, store.case_model_mappings())
    assert analysis["simulated_assessments"] == 2
    assert analysis["unique_cases"] == 1
    assert sum(analysis["preferred_counts"].values()) == 2
    assert "not a human study" in analysis["warning"]
    repeatability = analysis["descriptive_persona_repeatability"]
    assert repeatability["complete_two_slot_cases"] == 1
    assert repeatability["preference"]["exact_match_rate"] == 1.0
    assert "not human inter-rater reliability" in repeatability["warning"]

    dry_out = tmp_path / "simulation-dry"
    monkeypatch.setattr(
        simulation_script,
        "call_annotation_with_rate_limit_retries",
        lambda *_args, **_kwargs: pytest.fail("simulation dry run contacted network"),
    )
    dry_report = simulation_script.run_simulation(simulation_args(database, env, dry_out, dry_run=True))
    assert dry_report["summary"] == {"planned": 2}


def test_expert_interface_explains_the_blinded_assessment_in_chinese():
    guide = annotation.render_guide()

    assert "每个案例展示一项公开 API 文档" in guide
    assert "不是要求你重新设计工作流" in guide
    assert "不要调用 API" in guide
    assert "提交并锁定标注" in guide


def test_annotation_server_healthcheck_is_local(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=1,
        seed=17,
    )
    try:
        server = annotation.serve_study(database, "127.0.0.1", 0)
    except PermissionError:
        pytest.skip("the current sandbox prohibits binding a local test socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
            assert response.read() == b'{"status":"ok"}\n'
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            page = response.read().decode("utf-8")
            assert "SHARE 专家标注" in page
            assert "专家登录" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_non_loopback_server_requires_explicit_network_acknowledgement(tmp_path):
    collection = write_collection(tmp_path)
    database = tmp_path / "study.sqlite"
    annotation.create_study(
        collection_dir=collection,
        database_path=database,
        study_id="public-api-pilot-v1",
        annotator_ids=["expert_a"],
        admin_id="coordinator",
        max_workflows=1,
        ratings_per_workflow=1,
        seed=17,
    )

    with pytest.raises(ValueError, match="allow-network"):
        annotation.serve_study(database, "0.0.0.0", 8765)
    with pytest.raises(ValueError, match="requires TLS"):
        annotation.serve_study(database, "0.0.0.0", 8765, allow_network=True)
    with pytest.raises(ValueError, match="supplied together"):
        annotation.serve_study(database, "127.0.0.1", 8765, tls_cert=tmp_path / "cert.pem")
