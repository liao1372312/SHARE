from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "share_public_api_workflow_input", ROOT / "scripts" / "build_share_public_api_workflow_input.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
COLLECTION_SPEC = importlib.util.spec_from_file_location(
    "share_real_llm_collection", ROOT / "scripts" / "collect_share_real_llm_executions.py"
)
assert COLLECTION_SPEC is not None and COLLECTION_SPEC.loader is not None
collection = importlib.util.module_from_spec(COLLECTION_SPEC)
COLLECTION_SPEC.loader.exec_module(collection)


def test_builder_creates_non_synthetic_document_inputs_and_removes_sensitive_fields(tmp_path):
    source = tmp_path / "api"
    source.mkdir()
    (source / "service.json").write_text(
        json.dumps(
            {
                "name": "Example Service",
                "api_list": [
                    {
                        "name": "Get a record",
                        "method": "GET",
                        "url": "https://example.test/v1/records/{record_id}?debug=true",
                        "description": "Read a record from https://example.test/docs.",
                        "required_parameters": [
                            {"name": "record_id", "type": "string", "description": "Stable record ID", "default": "do-not-copy"},
                            {"name": "x-api-key", "type": "string", "description": "Credential", "default": "secret"},
                        ],
                    },
                    {"name": "Incomplete", "method": "GET"},
                ],
            }
        ),
        encoding="utf-8",
    )

    workflows, skipped = builder.build_workflows(source, 1)

    assert len(workflows) == 1
    assert skipped == {}
    workflow = workflows[0]
    assert workflow["synthetic"] is False
    assert workflow["source"] == builder.SOURCE_NAME
    assert workflow["context"]["evidence_scope"] == builder.EVIDENCE_SCOPE
    operation = workflow["context"]["operation"]
    assert operation["endpoint_path"] == "/v1/records/{record_id}"
    assert "example.test" not in operation["description"]
    assert operation["required_parameters"] == [
        {"name": "record_id", "type": "string", "description": "Stable record ID"}
    ]


def test_builder_skips_incomplete_operations_and_detects_duplicate_content(tmp_path):
    source = tmp_path / "api"
    source.mkdir()
    document = {
        "name": "Example Service",
        "api_list": [
            {"name": "Get a record", "method": "GET", "url": "https://example.test/records"},
            {"name": "Get a record", "method": "GET", "url": "https://example.test/records"},
            {"name": "Missing URL", "method": "GET"},
        ],
    }
    (source / "service.json").write_text(json.dumps(document), encoding="utf-8")

    workflows, skipped = builder.build_workflows(source, 10)

    assert len(workflows) == 1
    assert skipped == {"duplicate_operation": 1, "incomplete_operation": 1}


def test_builder_excludes_operations_with_credential_language_or_malformed_url(tmp_path):
    source = tmp_path / "api"
    source.mkdir()
    (source / "service.json").write_text(
        json.dumps(
            {
                "name": "Example Service",
                "api_list": [
                    {"name": "Validate API Key", "method": "GET", "url": "https://example.test/validate"},
                    {"name": "RapidAPI-Key status", "method": "GET", "url": "https://example.test/status"},
                    {"name": "Reset password", "method": "GET", "url": "https://example.test/reset"},
                    {"name": "Malformed URL", "method": "GET", "url": "/https://example.test/path"},
                    {"name": "List records", "method": "GET", "url": "https://example.test/records"},
                ],
            }
        ),
        encoding="utf-8",
    )

    workflows, skipped = builder.build_workflows(source, 10)

    assert len(workflows) == 1
    assert skipped == {"incomplete_operation": 4}


def test_builder_writes_jsonl_that_the_collector_accepts(tmp_path):
    source = tmp_path / "api"
    source.mkdir()
    (source / "service.json").write_text(
        json.dumps({"name": "Example Service", "api_list": [{"name": "List records", "method": "GET", "url": "https://example.test/records"}]}),
        encoding="utf-8",
    )
    workflows, _ = builder.build_workflows(source, 1)
    out = tmp_path / "workflows.jsonl"
    builder.write_jsonl(out, workflows)

    assert json.loads(out.read_text(encoding="utf-8"))["synthetic"] is False
    accepted = collection.load_real_workflows(out)
    assert len(accepted) == 1
    assert accepted[0]["source"] == builder.SOURCE_NAME
