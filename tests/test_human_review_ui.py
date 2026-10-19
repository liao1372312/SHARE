from __future__ import annotations

import json
import sys

import pytest

from share.human_review_ui import ReviewApp
from share.real_execution import PROHIBITED_FIXTURE_KEYS, iter_keys


def test_demo_repair_submission_is_append_only_and_unadjudicated(tmp_path):
    app = ReviewApp("demo", tmp_path, "demo_operator")
    assert len(app.meta()["tasks"]) == 2
    task_id = app.meta()["tasks"][0]["id"]
    session = app.start(task_id)
    task = session["task"]
    assert not iter_keys(task["decision_time_information"]) & PROHIBITED_FIXTURE_KEYS
    operation = task["decision_time_information"]["operation"]["operation_name"]
    result = app.submit({
        "session_id": session["session_id"],
        "decision": "submitted_repair",
        "notes": "Reviewed visible fixture only.",
        "repair_artifact": {
            "action": "bind_required_parameters",
            "target_operation": operation,
            "parameter_bindings": {"dummy_id": "demo-1"},
            "safety_checks": ["controlled_testbed_only"],
            "rationale": "Required input is missing in the visible failure.",
        },
        "success": True,
    })
    folder = tmp_path / "demo" / result["record_id"]
    saved = json.loads((folder / "submission.json").read_text())
    assert saved["outcome_status"] == "unadjudicated"
    assert saved["success"] is None and saved["unsafe"] is None
    assert saved["actual_repair_execution_verified"] is False
    assert saved["session_start_sha256"]
    assert len(list((tmp_path / "demo" / "session_starts").glob("*.json"))) == 1
    assert saved["repair_artifact_sha256"]
    assert (folder / "repair_artifact.json").is_file()
    assert (folder / "SHA256SUMS").is_file()
    with pytest.raises(ValueError, match="already submitted"):
        app.submit({"session_id": session["session_id"], "decision": "participant_stop", "notes": "stop"})


def test_demo_stop_requires_reason_and_has_no_repair_artifact(tmp_path):
    app = ReviewApp("demo", tmp_path, "demo_operator")
    session = app.start(app.meta()["tasks"][1]["id"])
    with pytest.raises(ValueError, match="Explain"):
        app.submit({"session_id": session["session_id"], "decision": "participant_stop", "notes": ""})
    result = app.submit({"session_id": session["session_id"], "decision": "participant_stop", "notes": "Evidence is insufficient."})
    folder = tmp_path / "demo" / result["record_id"]
    assert not (folder / "repair_artifact.json").exists()
    assert json.loads((folder / "submission.json").read_text())["completion_reason"] == "participant_stop"


def test_real_collection_refuses_current_blocked_gate(tmp_path):
    audit = tmp_path / "audit.json"
    readiness = tmp_path / "readiness.json"
    audit.write_text(json.dumps({"status": "BLOCKED_DO_NOT_COLLECT", "all_nine_gates_pass": False}))
    readiness.write_text(json.dumps({"status": "BLOCKED_DO_NOT_COLLECT", "all_nine_gates_pass": False}))
    with pytest.raises(ValueError, match="blocked"):
        ReviewApp("collect", tmp_path, "operator_123456789abc", audit, readiness)


def work_app(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workflow = workspace / "workflow.json"
    workflow.write_text('{"request":{"city":null}}\n')
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "tasks": [{
            "id": "weather-binding", "title": "Repair weather request",
            "request": "Bind the required city parameter.",
            "workflow_file": "workflow.json",
            "failure_type": "parameter_binding_failure",
            "observed_failure": {"message": "city is null"},
            "logs": [{"level": "error", "message": "missing city"}],
            "validation_command": [sys.executable, "-m", "json.tool", "{file}"],
        }],
    }))
    return ReviewApp("work", tmp_path / "records", "operator_local", tasks_path=manifest, workspace_root=workspace), workflow


def test_work_mode_edits_real_file_and_runs_check(tmp_path):
    app, workflow = work_app(tmp_path)
    session = app.start("weather-binding")
    assert session["task"]["editable_file"]["text"] == workflow.read_text()
    with pytest.raises(ValueError, match="invalid syntax"):
        app.work_preview({"session_id": session["session_id"], "updated_text": "{"})
    edited = '{"request":{"city":"Beijing"}}\n'
    preview = app.work_preview({"session_id": session["session_id"], "updated_text": edited})
    assert preview["changed"] and "-" in preview["diff"] and "+" in preview["diff"]
    result = app.submit({"session_id": session["session_id"], "decision": "apply_patch", "updated_text": edited, "notes": "Bound city from request."})
    assert workflow.read_text() == edited
    assert result["validation"]["passed"] is True
    record_folder = tmp_path / "records" / "work" / result["record_id"]
    assert (record_folder / "before.json").read_text() == '{"request":{"city":null}}\n'
    assert (record_folder / "after.json").read_text() == edited
    assert (record_folder / "change.diff").is_file()
    record = json.loads((record_folder / "submission.json").read_text())
    assert record["actual_file_modified"] is True and record["success"] is None
    with pytest.raises(ValueError, match="already submitted"):
        app.submit({"session_id": session["session_id"], "decision": "participant_stop", "notes": "stop"})
    reverted = app.revert_work({"record_id": result["record_id"]})
    assert workflow.read_text() == '{"request":{"city":null}}\n'
    assert (tmp_path / "records" / "work" / "reverts" / f"{reverted['revert_id']}.json").is_file()
    with pytest.raises(ValueError, match="automatic revert refused"):
        app.revert_work({"record_id": result["record_id"]})


def test_work_mode_rejects_stale_file(tmp_path):
    app, workflow = work_app(tmp_path)
    session = app.start("weather-binding")
    workflow.write_text('{"request":{"city":"external change"}}\n')
    with pytest.raises(ValueError, match="changed since this session"):
        app.submit({"session_id": session["session_id"], "decision": "apply_patch", "updated_text": '{"request":{"city":"Beijing"}}\n', "notes": ""})
    assert "external change" in workflow.read_text()


def test_work_mode_supports_python_and_rejects_bad_syntax(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = workspace / "task.py"
    code.write_text("def answer():\n    return None\n")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": [{
        "id": "code-task", "title": "Fix answer", "request": "Return a value.",
        "workflow_file": "task.py", "logs": []
    }]}))
    app = ReviewApp("work", tmp_path / "records", "operator_local", tasks_path=manifest, workspace_root=workspace)
    session = app.start("code-task")
    with pytest.raises(ValueError, match="invalid syntax"):
        app.work_preview({"session_id": session["session_id"], "updated_text": "def answer(: pass"})
    result = app.submit({"session_id": session["session_id"], "decision": "apply_patch", "updated_text": "def answer():\n    return 42\n", "notes": "Return expected value."})
    assert code.read_text().endswith("return 42\n")
    assert result["validation"]["syntax_kind"] == "py"


def test_work_mode_cannot_edit_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.py").write_text("value = 1\n")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": [{
        "id": "escape", "title": "Unsafe path", "request": "Edit file",
        "workflow_file": "../outside.py"
    }]}))
    with pytest.raises(ValueError, match="unsafe workflow path"):
        ReviewApp("work", tmp_path / "records", "operator_local", tasks_path=manifest, workspace_root=workspace)
