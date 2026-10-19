"""Local human repair workbench and outcome-blinded study authoring interface.

This records a person's proposed repair or decision to stop. It does not run
the repair, adjudicate success, or convert a simulated branch into an observed
execution. Formal collection is gated by the existing Plan-B readiness files.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from share.real_execution import PROTOCOL_ID, sha256_file, validate_fixture


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web/human_review"
DEMO_FIXTURES = ROOT / "results/share_plan_b_phase_b2_readiness_20260826/dry_run_artifacts/fixtures"
COLLECTION_MANIFEST = ROOT / "results/share_plan_b_phase_b2_readiness_20260826/executable_fixture_manifest.json"
GATE_AUDIT = ROOT / "results/share_plan_b_final_collection_gate_20260826/final_collection_gate_audit.json"
READINESS = ROOT / "results/share_plan_b_final_collection_gate_20260826/updated_collection_readiness.json"
MAX_BODY = 512 * 1024
MAX_EDIT_FILE = 256 * 1024
EDITABLE_SUFFIXES = {".json", ".py", ".toml"}
OPERATOR_ID = re.compile(r"^operator_[0-9a-f]{12,64}$")
WORK_PROTOCOL = "share-human-workbench-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def require_collection_ready(audit_path: Path, readiness_path: Path, operator_id: str) -> None:
    audit, readiness = read_json(audit_path), read_json(readiness_path)
    if audit.get("status") != "READY_TO_COLLECT" or audit.get("all_nine_gates_pass") is not True:
        raise ValueError("Formal Human collection is blocked by the final nine-gate audit")
    if readiness.get("status") != "READY_TO_COLLECT" or readiness.get("all_nine_gates_pass") is not True:
        raise ValueError("Formal Human collection is blocked by the readiness record")
    if readiness.get("final_audit_sha256") != sha256_file(audit_path):
        raise ValueError("Readiness record does not bind the current gate audit")
    gates = audit.get("gates", [])
    if len(gates) != 9 or {item.get("gate") for item in gates} != set(range(1, 10)) or any(item.get("status") != "PASS" for item in gates):
        raise ValueError("At least one collection gate has not passed")
    for item in gates:
        ref = item.get("record")
        if isinstance(ref, dict) and ref.get("sha256"):
            record_path = ROOT / str(ref.get("path", ""))
            if not record_path.is_file() or sha256_file(record_path) != ref["sha256"]:
                raise ValueError(f"Verified gate {item['gate']} record is missing or changed")
    roster_gate = next((item for item in gates if item.get("gate") == 8), None)
    if roster_gate is None:
        raise ValueError("Verified operator roster gate is missing")
    roster_ref = roster_gate.get("record", {})
    roster_path = ROOT / str(roster_ref.get("path", ""))
    if not roster_path.is_file() or sha256_file(roster_path) != roster_ref.get("sha256"):
        raise ValueError("Qualified operator roster is missing or changed")
    roster = read_json(roster_path)
    members = [item for item in roster.get("operators", []) if item.get("anonymized_operator_id") == operator_id]
    if len(members) != 1 or members[0].get("eligible") is not True or members[0].get("consent_status") != "valid":
        raise ValueError("Operator is not in the verified, consented roster")


def load_work_tasks(tasks_path: Path, workspace_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    if not workspace_root.is_dir():
        raise ValueError("Work mode requires an existing workspace directory")
    manifest = read_json(tasks_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("tasks"), list) or not manifest["tasks"]:
        raise ValueError("Work task manifest needs schema_version=1 and a nonempty tasks list")
    tasks = {}
    root = workspace_root.resolve()
    for index, item in enumerate(manifest["tasks"], 1):
        if not isinstance(item, dict):
            raise ValueError("Each work task must be a JSON object")
        task_id = str(item.get("id", ""))
        relative = Path(str(item.get("workflow_file", "")))
        if not task_id or len(task_id) > 100 or not re.fullmatch(r"[A-Za-z0-9._-]+", task_id):
            raise ValueError("Work task IDs may contain only letters, numbers, dot, underscore, and dash")
        if task_id in tasks or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Duplicate task ID or unsafe workflow path")
        file_path = (root / relative).resolve(strict=True)
        if not file_path.is_relative_to(root) or file_path.suffix.lower() not in EDITABLE_SUFFIXES or not file_path.is_file() or file_path.stat().st_size > MAX_EDIT_FILE:
            raise ValueError("Each work task must reference a JSON, Python, or TOML file inside the workspace, at most 256 KiB")
        check = item.get("validation_command")
        if check is not None and (not isinstance(check, list) or not check or len(check) > 30 or any(not isinstance(part, str) or not part or len(part) > 500 for part in check)):
            raise ValueError("Validation command must be a nonempty argument list")
        title, request = item.get("title"), item.get("request")
        if not isinstance(title, str) or not title.strip() or not isinstance(request, str) or not request.strip():
            raise ValueError("Every work task needs a title and request")
        logs = item.get("logs", [])
        if not isinstance(logs, list) or len(logs) > 100:
            raise ValueError("Task logs must be a short list")
        failure = str(item.get("failure_type", "unspecified_failure"))
        fixture = {
            "fixture_id": f"work-{task_id}", "episode_id": task_id,
            "subset_order": index, "failure_type": failure,
            "complexity_bin": str(item.get("difficulty", "unrated")),
            "decision_time_information": {
                "task": request,
                "operation": {"operation_name": title, "workflow_file": str(relative)},
                "observed_failure": item.get("observed_failure", {}),
                "available_logs": logs,
                "allowed_actions": ["edit_source_file", "check_syntax", "run_configured_check", "stop"],
            },
            "candidate_repair": item.get("candidate_repair"),
            "_work": {"file": file_path, "relative": str(relative), "root": root, "validation_command": check},
        }
        tasks[task_id] = tasks_path, fixture
    return tasks


def load_tasks(mode: str, tasks_path: Path | None = None, workspace_root: Path | None = None) -> dict[str, tuple[Path, dict[str, Any]]]:
    if mode == "work":
        if tasks_path is None or workspace_root is None:
            raise ValueError("Work mode requires --tasks and --workspace-root")
        return load_work_tasks(tasks_path, workspace_root)
    if mode == "demo":
        paths = sorted(DEMO_FIXTURES.glob("dummy_*.json"))
        if len(paths) != 2:
            raise ValueError("Expected two non-study demo fixtures")
        expected_hashes: dict[Path, str] = {}
    else:
        manifest = read_json(COLLECTION_MANIFEST)
        if manifest.get("fixtures") != 120 or manifest.get("fixture_outcomes_present") is not False:
            raise ValueError("Invalid frozen collection fixture manifest")
        paths = [ROOT / item["path"] for item in manifest["entries"]]
        expected_hashes = {ROOT / item["path"]: item["sha256"] for item in manifest["entries"]}
    tasks = {}
    for path in paths:
        if path in expected_hashes and sha256_file(path) != expected_hashes[path]:
            raise ValueError(f"Fixture changed: {path}")
        fixture = read_json(path)
        validate_fixture(fixture, allow_dry_run=mode == "demo")
        if fixture["dry_run"] is not (mode == "demo"):
            raise ValueError("Demo and collection fixtures must stay separate")
        key = str(fixture["episode_id"])
        if key in tasks:
            raise ValueError("Duplicate episode ID")
        tasks[key] = path, fixture
    if len(tasks) != (2 if mode == "demo" else 120):
        raise ValueError("Unexpected task count")
    return tasks


def repair_from_payload(payload: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("repair_artifact")
    required = {"action", "target_operation", "parameter_bindings", "safety_checks", "rationale"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("Repair artifact must contain action, target, bindings, safety checks, and rationale")
    if not all(isinstance(raw[key], str) and 0 < len(raw[key].strip()) <= 4000 for key in ("action", "target_operation", "rationale")):
        raise ValueError("Action, target, and rationale must be nonempty text")
    if raw["target_operation"].strip() != fixture["decision_time_information"]["operation"]["operation_name"]:
        raise ValueError("Target operation must match the visible fixture")
    bindings = raw["parameter_bindings"]
    if not isinstance(bindings, dict) or len(bindings) > 50 or any(not isinstance(k, str) or not k or len(k) > 120 or not isinstance(v, (str, int, float, bool, type(None))) for k, v in bindings.items()):
        raise ValueError("Bindings must be a small JSON object of scalar values")
    checks = raw["safety_checks"]
    if not isinstance(checks, list) or not checks or len(checks) > 30 or any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in checks):
        raise ValueError("List at least one concise safety check")
    return {"action": raw["action"].strip(), "target_operation": raw["target_operation"].strip(), "parameter_bindings": bindings, "safety_checks": [v.strip() for v in checks], "rationale": raw["rationale"].strip()}


@dataclass
class Session:
    task_id: str
    started_at_utc: str
    started_monotonic: float
    fixture_sha256: str
    work_file_sha256: str | None = None
    completed: bool = False


class ReviewApp:
    def __init__(self, mode: str, out: Path, operator_id: str, audit_path: Path = GATE_AUDIT, readiness_path: Path = READINESS, tasks_path: Path | None = None, workspace_root: Path | None = None):
        if mode not in {"demo", "collect", "work"}:
            raise ValueError("mode must be demo, collect, or work")
        if mode == "work" and (not operator_id.strip() or operator_id == "demo_operator"):
            raise ValueError("Work mode requires an explicit --operator-id pseudonym")
        if mode == "collect":
            if not OPERATOR_ID.fullmatch(operator_id):
                raise ValueError("Formal collection requires an anonymized operator ID")
            require_collection_ready(audit_path, readiness_path, operator_id)
        self.mode, self.out, self.operator_id = mode, out, operator_id
        self.access_token = secrets.token_urlsafe(24) if mode in {"collect", "work"} else ""
        self.audit_path, self.readiness_path = audit_path, readiness_path
        self.tasks = load_tasks(mode, tasks_path, workspace_root)
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()

    def ready(self) -> None:
        if self.mode == "collect":
            require_collection_ready(self.audit_path, self.readiness_path, self.operator_id)

    def meta(self) -> dict[str, Any]:
        self.ready()
        tasks = []
        for key, (_, fixture) in self.tasks.items():
            tasks.append({"id": key, "order": fixture["subset_order"], "failure_type": fixture["failure_type"], "complexity": fixture["complexity_bin"]})
        return {"mode": self.mode, "protocol": WORK_PROTOCOL if self.mode == "work" else PROTOCOL_ID, "operator_id": self.operator_id, "tasks": tasks, "outcome_labels_visible": False}

    def start(self, task_id: str) -> dict[str, Any]:
        self.ready()
        if task_id not in self.tasks:
            raise ValueError("Unknown task")
        path, fixture = self.tasks[task_id]
        token = uuid.uuid4().hex
        session = Session(task_id, utc_now(), time.monotonic(), sha256_file(path))
        if self.mode == "work":
            session.work_file_sha256 = sha256_file(fixture["_work"]["file"])
        with self.lock:
            starts = self.out / self.mode / "session_starts"
            starts.mkdir(parents=True, exist_ok=True)
            write_new(starts / f"{token}.json", {
                "schema_version": 1, "kind": "share_human_review_session_start",
                "protocol_id": WORK_PROTOCOL if self.mode == "work" else PROTOCOL_ID,
                "mode": self.mode, "episode_id": task_id,
                "fixture_sha256": session.fixture_sha256,
                "work_file_sha256": session.work_file_sha256,
                "anonymized_operator_id": self.operator_id,
                "started_at_utc": session.started_at_utc,
                "outcome_status": "unadjudicated",
            })
            self.sessions[token] = session
        response = {
            "session_id": token, "started_at_utc": session.started_at_utc,
            "task": {
                "episode_id": task_id, "fixture_id": fixture["fixture_id"],
                "failure_type": fixture["failure_type"], "complexity": fixture["complexity_bin"],
                "decision_time_information": fixture["decision_time_information"],
                "fixture_sha256": session.fixture_sha256,
            },
        }
        if self.mode == "work":
            work = fixture["_work"]
            response["task"]["editable_file"] = {
                "path": work["relative"], "sha256": session.work_file_sha256,
                "text": work["file"].read_text(encoding="utf-8"),
                "format": work["file"].suffix.lower().lstrip("."),
            }
            response["task"]["candidate_repair"] = fixture["candidate_repair"]
        return response

    def work_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mode != "work":
            raise ValueError("File preview is available only in work mode")
        with self.lock:
            session, path, before, after = self._work_edit(payload)
            diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"before/{session.task_id}", tofile=f"after/{session.task_id}",
            ))
            return {
                "changed": before != after,
                "before_sha256": session.work_file_sha256,
                "after_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
                "syntax_valid": True,
                "syntax_kind": path.suffix.lower().lstrip("."),
                "diff": diff[:100_000] + ("\n… diff truncated in preview" if len(diff) > 100_000 else ""),
                "file": str(path),
            }

    def _work_edit(self, payload: dict[str, Any]) -> tuple[Session, Path, str, str]:
        token = payload.get("session_id")
        session = self.sessions.get(token) if isinstance(token, str) else None
        if session is None or session.completed:
            raise ValueError("Session does not exist or was already submitted")
        source, fixture = self.tasks[session.task_id]
        work = fixture["_work"]
        path = (work["root"] / work["relative"]).resolve(strict=True)
        if path != work["file"] or not path.is_relative_to(work["root"]):
            raise ValueError("Workflow path changed or left the configured workspace")
        if sha256_file(source) != session.fixture_sha256 or sha256_file(path) != session.work_file_sha256:
            raise ValueError("Task definition or workflow file changed since this session began; reopen the task")
        before = path.read_text(encoding="utf-8")
        after = payload.get("updated_text")
        if not isinstance(after, str) or len(after.encode("utf-8")) > MAX_EDIT_FILE:
            raise ValueError("Edited workflow must be UTF-8 text of at most 256 KiB")
        try:
            if path.suffix.lower() == ".json":
                parsed = json.loads(after)
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("Edited JSON workflow must be an object or array")
            elif path.suffix.lower() == ".py":
                ast.parse(after, filename=str(path))
            elif path.suffix.lower() == ".toml":
                tomllib.loads(after)
            else:
                raise ValueError("Unsupported source file format")
        except (json.JSONDecodeError, SyntaxError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"Edited source file has invalid syntax: {error}") from error
        return session, path, before, after

    def _submit_work(self, payload: dict[str, Any], session: Session, token: str) -> dict[str, Any]:
        decision = payload.get("decision")
        if decision not in {"apply_patch", "participant_stop"}:
            raise ValueError("Work decision must be apply_patch or participant_stop")
        notes = payload.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 8000:
            raise ValueError("Notes must be text of at most 8,000 characters")
        if decision == "participant_stop" and not notes.strip():
            raise ValueError("Explain why the task was stopped")
        _, fixture = self.tasks[session.task_id]
        work = fixture["_work"]
        if decision == "apply_patch":
            _, path, before, after = self._work_edit(payload)
            if before == after:
                raise ValueError("No file changes to apply")
        else:
            path = work["file"]
            if sha256_file(path) != session.work_file_sha256:
                raise ValueError("Workflow file changed since this session began; reopen the task")
            before, after = path.read_text(encoding="utf-8"), None
        record_id = uuid.uuid4().hex
        folder = self.out / "work" / record_id
        folder.mkdir(parents=True, exist_ok=False)
        validation = {"syntax_valid": None, "syntax_kind": path.suffix.lower().lstrip("."), "configured_check_ran": False, "passed": None}
        after_sha = None
        if after is not None:
            before_path = folder / f"before{path.suffix.lower()}"
            after_path = folder / f"after{path.suffix.lower()}"
            before_path.write_text(before, encoding="utf-8")
            after_path.write_text(after, encoding="utf-8")
            diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"before/{work['relative']}", tofile=f"after/{work['relative']}",
            ))
            (folder / "change.diff").write_text(diff, encoding="utf-8")
            temporary = path.with_name(f".{path.name}.{record_id}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(after.encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, path.stat().st_mode)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            after_sha = sha256_file(path)
            validation["syntax_valid"] = True
            command = work["validation_command"]
            if command:
                argv = [part.replace("{file}", str(path)) for part in command]
                validation["configured_check_ran"] = True
                validation["command"] = argv
                try:
                    check = subprocess.run(argv, cwd=work["root"], capture_output=True, text=True, timeout=60, check=False)
                    validation.update({"passed": check.returncode == 0, "returncode": check.returncode, "stdout": check.stdout[-20_000:], "stderr": check.stderr[-20_000:]})
                except (OSError, subprocess.TimeoutExpired) as error:
                    validation.update({"passed": False, "error": str(error)})
            final_sha = sha256_file(path)
            validation["file_changed_during_check"] = final_sha != after_sha
            if final_sha != after_sha:
                validation["passed"] = False
                validation["error"] = "Configured check changed the edited file; inspect and reopen the task"
                after_sha = final_sha
        record = {
            "schema_version": 1, "kind": "share_human_workflow_edit_record",
            "protocol_id": WORK_PROTOCOL,
            "record_id": record_id, "mode": "work", "submitted_by_operator_ui": True,
            "human_authorship_independently_verified": False,
            "actual_file_modified": after is not None,
            "outcome_status": "unadjudicated", "success": None, "unsafe": None,
            "task_id": session.task_id, "task_manifest_sha256": session.fixture_sha256,
            "session_start_sha256": sha256_file(self.out / "work" / "session_starts" / f"{token}.json"),
            "anonymized_operator_id": self.operator_id,
            "workspace_file": work["relative"],
            "file_before_sha256": session.work_file_sha256,
            "file_after_sha256": after_sha,
            "started_at_utc": session.started_at_utc, "completed_at_utc": utc_now(),
            "intervention_seconds": round(max(0.0, time.monotonic() - session.started_monotonic), 3),
            "completion_reason": decision, "notes": notes.strip(), "validation": validation,
        }
        write_new(folder / "submission.json", record)
        (folder / "SHA256SUMS").write_text("".join(f"{sha256_file(file)}  {file.name}\n" for file in sorted(folder.iterdir())), encoding="utf-8")
        session.completed = True
        return {"record_id": record_id, "saved_to": str(folder), "completion_reason": decision,
                "outcome_status": "unadjudicated", "file_after_sha256": after_sha, "validation": validation}

    def revert_work(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mode != "work":
            raise ValueError("Revert is available only in work mode")
        record_id = payload.get("record_id")
        if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{32}", record_id):
            raise ValueError("Invalid edit record ID")
        with self.lock:
            folder = self.out / "work" / record_id
            if not (folder / "submission.json").is_file():
                raise ValueError("Unknown edit record")
            record = read_json(folder / "submission.json")
            if record.get("completion_reason") != "apply_patch" or record.get("anonymized_operator_id") != self.operator_id:
                raise ValueError("Edit record does not belong to this operator")
            task_id = record.get("task_id")
            if task_id not in self.tasks:
                raise ValueError("Edit task is not loaded")
            work = self.tasks[task_id][1]["_work"]
            path = (work["root"] / work["relative"]).resolve(strict=True)
            if path != work["file"] or sha256_file(path) != record.get("file_after_sha256"):
                raise ValueError("File changed after this edit; automatic revert refused")
            backup = folder / f"before{path.suffix.lower()}"
            if sha256_file(backup) != record.get("file_before_sha256"):
                raise ValueError("Original-file backup failed hash verification")
            revert_dir = self.out / "work" / "reverts"
            revert_dir.mkdir(parents=True, exist_ok=True)
            revert_id = uuid.uuid4().hex
            temporary = path.with_name(f".{path.name}.{revert_id}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(backup.read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, path.stat().st_mode)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            reverted_sha = sha256_file(path)
            write_new(revert_dir / f"{revert_id}.json", {
                "schema_version": 1, "kind": "share_human_workflow_edit_revert",
                "protocol_id": WORK_PROTOCOL, "revert_id": revert_id,
                "edit_record_id": record_id, "operator_id": self.operator_id,
                "task_id": task_id, "workspace_file": work["relative"],
                "reverted_at_utc": utc_now(), "file_sha256": reverted_sha,
                "success": None, "unsafe": None,
            })
            return {"revert_id": revert_id, "edit_record_id": record_id, "file_sha256": reverted_sha}

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ready()
        token = payload.get("session_id")
        if not isinstance(token, str):
            raise ValueError("Missing session")
        with self.lock:
            session = self.sessions.get(token)
            if session is None or session.completed:
                raise ValueError("Session does not exist or was already submitted")
            if self.mode == "work":
                return self._submit_work(payload, session, token)
            path, fixture = self.tasks[session.task_id]
            if sha256_file(path) != session.fixture_sha256:
                raise ValueError("Fixture changed after session start")
            decision = payload.get("decision")
            if decision not in {"submitted_repair", "participant_stop"}:
                raise ValueError("Decision must be submitted_repair or participant_stop")
            notes = payload.get("notes", "")
            if not isinstance(notes, str) or len(notes) > 8000:
                raise ValueError("Notes must be text of at most 8,000 characters")
            repair = repair_from_payload(payload, fixture) if decision == "submitted_repair" else None
            if decision == "participant_stop" and not notes.strip():
                raise ValueError("Explain why the task was stopped")
            record_id = uuid.uuid4().hex
            folder = self.out / self.mode / record_id
            folder.mkdir(parents=True, exist_ok=False)
            completed = utc_now()
            record = {
                "schema_version": 1, "kind": "share_human_repair_authoring_record",
                "protocol_id": PROTOCOL_ID, "record_id": record_id,
                "mode": self.mode, "dry_run": self.mode == "demo",
                "real_human_authoring_observed": self.mode == "collect",
                "actual_repair_execution_verified": False,
                "outcome_status": "unadjudicated", "success": None, "unsafe": None,
                "episode_id": session.task_id, "fixture_sha256": session.fixture_sha256,
                "session_start_sha256": sha256_file(self.out / self.mode / "session_starts" / f"{token}.json"),
                "anonymized_operator_id": self.operator_id,
                "started_at_utc": session.started_at_utc, "completed_at_utc": completed,
                "intervention_seconds": round(max(0.0, time.monotonic() - session.started_monotonic), 3),
                "completion_reason": decision, "notes": notes.strip(),
                "repair_artifact_sha256": None,
            }
            if repair is not None:
                repair_path = folder / "repair_artifact.json"
                write_new(repair_path, repair)
                record["repair_artifact_sha256"] = sha256_file(repair_path)
            write_new(folder / "submission.json", record)
            sums = "".join(f"{sha256_file(file)}  {file.name}\n" for file in sorted(folder.iterdir()))
            (folder / "SHA256SUMS").write_text(sums, encoding="utf-8")
            session.completed = True
            return {"record_id": record_id, "saved_to": str(folder), "completion_reason": decision, "outcome_status": "unadjudicated"}


class Handler(BaseHTTPRequestHandler):
    app: ReviewApp

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def respond(self, status: int, data: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status: int, value: Any) -> None:
        self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def authenticated(self) -> bool:
        if self.app.mode == "demo" or secrets.compare_digest(self.headers.get("X-SHARE-TOKEN", ""), self.app.access_token):
            return True
        self.json_response(403, {"error": "Missing or invalid operator session token"})
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/meta":
            if not self.authenticated():
                return
            try:
                self.json_response(200, self.app.meta())
            except ValueError as error:
                self.json_response(403, {"error": str(error)})
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
        if parsed.path not in assets:
            self.json_response(404, {"error": "Not found"})
            return
        name, mime = assets[parsed.path]
        self.respond(200, (ASSETS / name).read_bytes(), mime)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/session", "/api/preview", "/api/submit", "/api/revert"}:
            self.json_response(404, {"error": "Not found"})
            return
        if not self.authenticated():
            return
        if self.headers.get("X-SHARE-UI") != "1" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json_response(400, {"error": "Invalid request headers"})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.json_response(403, {"error": "Cross-origin request rejected"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise ValueError("Request body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            if parsed.path == "/api/session":
                result = self.app.start(str(payload.get("task_id", "")))
            elif parsed.path == "/api/preview":
                result = self.app.work_preview(payload)
            elif parsed.path == "/api/revert":
                result = self.app.revert_work(payload)
            else:
                result = self.app.submit(payload)
        except (ValueError, json.JSONDecodeError) as error:
            self.json_response(400, {"error": str(error)})
            return
        self.json_response(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "collect", "work"), default="work")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, default=ROOT / "results/share_human_review_ui")
    parser.add_argument("--operator-id", default="demo_operator")
    parser.add_argument("--gate-audit", type=Path, default=GATE_AUDIT)
    parser.add_argument("--readiness", type=Path, default=READINESS)
    parser.add_argument("--tasks", type=Path, help="JSON task manifest for real work mode")
    parser.add_argument("--workspace-root", type=Path, help="Directory containing editable workflow JSON files")
    args = parser.parse_args()
    app = ReviewApp(args.mode, args.out, args.operator_id, args.gate_audit, args.readiness, args.tasks, args.workspace_root)
    handler = type("BoundReviewHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    if app.access_token:
        url += f"?key={app.access_token}"
    print(f"SHARE Human Review ({args.mode}): {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
