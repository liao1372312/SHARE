from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("share_real_llm_collection", ROOT / "scripts" / "collect_share_real_llm_executions.py")
assert SPEC is not None and SPEC.loader is not None
collection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collection)


def write_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "LLM_BASE_URL=https://example.test/v1",
                "LLM_API_KEY=primary-secret-key",
                "LLM_MODEL=primary-model",
                "LLM_BACKBONE_2_BASE_URL=https://example.test/v1",
                "LLM_BACKBONE_2_API_KEY=second-secret-key",
                "LLM_BACKBONE_2_MODEL=second-model",
                "LLM_BACKBONE_2_REASONING_EFFORT=low",
                "LLM_BACKBONE_3_BASE_URL=https://example.test/v1",
                "LLM_BACKBONE_3_API_KEY=third-secret-key",
                "LLM_BACKBONE_3_MODEL=third-model",
            ]
        ),
        encoding="utf-8",
    )


def write_workflows(path: Path, *, synthetic: bool = False, template_only: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template_only": template_only,
                "workflow_id": "workflow-1",
                "group_id": "group-1",
                "source": "externally-authorized-workflow-collection",
                "synthetic": synthetic,
                "task": "Repair a failed authorized service-composition workflow.",
                "context": {"failed_trace": "minimal failure trace"},
                "metadata": {"collection_reference": "record-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def args_for(env: Path, workflows: Path, out: Path, **overrides):
    values = {
        "env": env,
        "input": workflows,
        "out": out,
        "collection_id": "collection-1",
        "max_workflows": 1,
        "max_output_tokens": 64,
        "max_input_chars": 500,
        "timeout_seconds": 1,
        "resume": False,
        "retry_failed": False,
        "retry_uncertain": False,
        "max_rate_limit_retries": 0,
        "rate_limit_backoff_seconds": 60.0,
        "idempotency_header": None,
        "json_mode": False,
        "dry_run": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_collection_rejects_synthetic_and_template_inputs(tmp_path):
    synthetic = tmp_path / "synthetic.jsonl"
    template = tmp_path / "template.jsonl"
    write_workflows(synthetic, synthetic=True)
    write_workflows(template, template_only=True)

    with pytest.raises(ValueError, match="synthetic=false"):
        collection.load_real_workflows(synthetic)
    with pytest.raises(ValueError, match="template"):
        collection.load_real_workflows(template)


def test_collection_cli_accepts_the_2048_output_token_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_share_real_llm_executions.py",
            "--input",
            str(tmp_path / "workflows.jsonl"),
            "--collection-id",
            "collection-1",
            "--out",
            str(tmp_path / "out"),
            "--max-workflows",
            "1",
            "--max-output-tokens",
            "2048",
        ],
    )

    assert collection.parse_args().max_output_tokens == 2048


def test_collection_cli_supports_progress_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_share_real_llm_executions.py",
            "--input",
            str(tmp_path / "workflows.jsonl"),
            "--collection-id",
            "collection-1",
            "--out",
            str(tmp_path / "out"),
            "--max-workflows",
            "1",
            "--no-progress",
            "--progress-every",
            "7",
        ],
    )

    args = collection.parse_args()

    assert args.progress is False
    assert args.progress_every == 7


def test_request_payload_can_explicitly_request_json_object_mode():
    profile = collection.LLMProfile("primary", "https://example.test/v1", "test-key", "test-model")
    workflow = {"workflow_id": "workflow-1", "task": "Repair this workflow.", "context": {}}

    payload, truncated = collection.request_payload(profile, workflow, 500, 128, json_mode=True)

    assert truncated is False
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "test-model"


def test_collection_dry_run_binds_input_and_never_calls_network(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    monkeypatch.setattr(collection, "call_profile", lambda *_args, **_kwargs: pytest.fail("dry run contacted network"))

    report = collection.run_collection(args_for(env, workflows, out, dry_run=True))

    assert report["dry_run"] is True
    assert report["summary"] == {"planned": 3}
    manifest = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_kind"] == "real_ai_execution_artifacts_not_paired_outcomes"
    assert manifest["caps"]["max_requests"] == 3


def test_collection_writes_private_artifacts_and_sanitized_index(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    monkeypatch.setattr(
        collection,
        "call_profile",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "latency_ms": 1.0,
            "response": {"id": "response-1", "choices": [{"message": {"content": '{"action":"RetryTool"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            "response_content": '{"action":"RetryTool"}',
            "content_is_json": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    report = collection.run_collection(args_for(env, workflows, out))

    assert report["summary"] == {"started": 0, "completed": 3, "failed": 0, "uncertain": 0, "skipped": 0}
    index = (out / "execution_index.jsonl").read_text(encoding="utf-8")
    assert "primary-secret-key" not in index
    assert "second-secret-key" not in index
    assert "third-secret-key" not in index
    assert "example.test" not in index
    records = [json.loads(line) for line in index.splitlines() if line.strip()]
    completed = [record for record in records if record["status"] == "completed"]
    assert len(completed) == 3
    artifact = out / completed[0]["execution_artifact_file"]
    assert artifact.is_file()
    assert (artifact.stat().st_mode & 0o777) == 0o600
    assert (out.stat().st_mode & 0o777) == 0o700
    assert (artifact.parent.stat().st_mode & 0o777) == 0o700
    assert completed[0]["execution_artifact_sha256"] == collection.sha256_file(artifact)


def test_collection_reports_sanitized_progress_on_stderr(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    monkeypatch.setattr(
        collection,
        "call_profile",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "latency_ms": 1.0,
            "response": {"choices": [{"message": {"content": "{}"}}]},
            "response_content": "{}",
            "content_is_json": True,
            "usage": {},
        },
    )

    report = collection.run_collection(args_for(env, workflows, out))

    captured = capsys.readouterr()
    assert report["summary"]["completed"] == 3
    assert captured.out == ""
    assert "[SHARE] starting" in captured.err
    assert "[SHARE] requesting" in captured.err
    assert "completed=3/3 (100.0%)" in captured.err
    assert "primary-secret-key" not in captured.err
    assert "second-secret-key" not in captured.err
    assert "third-secret-key" not in captured.err
    assert "example.test" not in captured.err


def test_collection_resume_rejects_changed_endpoint_and_missing_completed_artifact(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    changed_env = tmp_path / "changed.env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_env(changed_env)
    changed_env.write_text(changed_env.read_text(encoding="utf-8").replace("https://example.test/v1", "https://other.test/v1"), encoding="utf-8")
    write_workflows(workflows)
    monkeypatch.setattr(
        collection,
        "call_profile",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "latency_ms": 1,
            "response": {"choices": [{"message": {"content": "{}"}}]},
            "response_content": "{}",
            "content_is_json": True,
            "usage": {},
        },
    )
    collection.run_collection(args_for(env, workflows, out))

    with pytest.raises(ValueError, match="manifest does not match"):
        collection.run_collection(args_for(changed_env, workflows, out, resume=True))

    record = next(
        json.loads(line)
        for line in (out / "execution_index.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["status"] == "completed"
    )
    (out / record["execution_artifact_file"]).unlink()
    with pytest.raises(ValueError, match="completed artifact"):
        collection.run_collection(args_for(env, workflows, out, resume=True))


def test_collection_marks_transport_failures_uncertain_and_requires_explicit_retry(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    monkeypatch.setattr(
        collection,
        "call_profile",
        lambda *_args, **_kwargs: {"status": "uncertain", "error_kind": "transport_error", "latency_ms": 1},
    )

    report = collection.run_collection(args_for(env, workflows, out))

    assert report["summary"] == {"started": 0, "completed": 0, "failed": 0, "uncertain": 3, "skipped": 0}
    with pytest.raises(ValueError, match="uncertain"):
        collection.run_collection(args_for(env, workflows, out, resume=True))


def test_collection_rejects_non_object_json_response_without_crashing(monkeypatch):
    class FakeResponse:
        def read(self):
            return b"[]"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *_args: FakeOpener())
    profile = collection.LLMProfile("primary", "https://example.test/v1", "test-key", "test-model")

    result = collection.call_profile(profile, {"model": "test-model"}, timeout_seconds=1)

    assert result["status"] == "failed"
    assert result["error_kind"] == "invalid_chat_completion_response"
    assert result["response"] == []


def test_collection_marks_empty_completion_content_as_invalid_proposal_json(monkeypatch):
    class FakeResponse:
        def read(self):
            return b'{"choices":[{"message":{"content":""}}],"usage":{"total_tokens":3}}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *_args: FakeOpener())
    profile = collection.LLMProfile("primary", "https://example.test/v1", "test-key", "test-model")

    result = collection.call_profile(profile, {"model": "test-model"}, timeout_seconds=1)

    assert result["status"] == "failed"
    assert result["error_kind"] == "invalid_proposal_json"
    assert result["content_is_json"] is False
    assert result["usage"] == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 3}


def test_collection_stops_a_profile_after_its_first_error(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    calls = []

    def failed_call(profile, *_args, **_kwargs):
        calls.append(profile.name)
        return {"status": "failed", "error_kind": "http_error", "http_status": 400, "latency_ms": 1}

    monkeypatch.setattr(collection, "call_profile", failed_call)

    report = collection.run_collection(args_for(env, workflows, out))

    assert calls == ["primary", "backbone_2", "backbone_3"]
    assert report["summary"] == {"started": 0, "completed": 0, "failed": 3, "uncertain": 0, "skipped": 0}


def test_collection_retries_explicit_rate_limits_and_records_the_wait(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    workflows = tmp_path / "workflows.jsonl"
    out = tmp_path / "out"
    write_env(env)
    write_workflows(workflows)
    calls = []
    waits = []

    def rate_limited_then_completed(profile, *_args, **_kwargs):
        calls.append(profile.name)
        if profile.name == "backbone_2" and calls.count(profile.name) == 1:
            return {
                "status": "failed",
                "error_kind": "rate_limited",
                "http_status": 429,
                "retry_after_seconds": 12.0,
                "latency_ms": 1,
            }
        return {
            "status": "completed",
            "latency_ms": 1,
            "response": {"choices": [{"message": {"content": "{}"}}]},
            "response_content": "{}",
            "content_is_json": True,
            "usage": {},
        }

    monkeypatch.setattr(collection, "call_profile", rate_limited_then_completed)
    monkeypatch.setattr(collection.time, "sleep", waits.append)

    report = collection.run_collection(
        args_for(env, workflows, out, max_rate_limit_retries=1, rate_limit_backoff_seconds=7.0)
    )

    assert report["summary"] == {"started": 0, "completed": 3, "failed": 0, "uncertain": 0, "skipped": 0}
    assert calls == ["primary", "backbone_2", "backbone_2", "backbone_3"]
    assert waits == [12.0]
    records = [json.loads(line) for line in (out / "execution_index.jsonl").read_text(encoding="utf-8").splitlines()]
    backbone_2 = [record for record in records if record["profile"] == "backbone_2" and record["status"] == "completed"]
    assert backbone_2[0]["rate_limit_retries"] == 1
    assert backbone_2[0]["rate_limit_wait_seconds"] == 12.0
