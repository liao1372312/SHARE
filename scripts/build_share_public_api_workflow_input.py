from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


SOURCE_NAME = "repository_public_api_document_corpus"
EVIDENCE_SCOPE = "public_api_document_review_not_observed_service_execution_or_paired_outcome"
URL_PATTERN = re.compile(r"https?://[^\s'\"<>]*", re.IGNORECASE)
SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?:api[_ -]?key|apikey|authorization|bearer|password|secret|credential|access[_ -]?token|accesstoken|api[_ -]?token|apitoken)",
    re.IGNORECASE,
)
SENSITIVE_NAME_PATTERN = re.compile(r"\b(?:key|token)\b|" + SENSITIVE_TEXT_PATTERN.pattern, re.IGNORECASE)
MAX_TEXT_CHARS = 1_600
MAX_PARAMETERS = 16


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


def clean_text(value: object, *, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    text = URL_PATTERN.sub("[URL omitted]", value)
    text = " ".join(text.split())
    return text[:limit]


def operation_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    parsed = urlparse(value)
    path = parsed.path if parsed.netloc else value
    path = path.split("?", 1)[0].split("#", 1)[0].strip()
    if "://" in path or path.startswith("//"):
        return ""
    return path[:512]


def is_sensitive_name(value: object) -> bool:
    return isinstance(value, str) and bool(SENSITIVE_NAME_PATTERN.search(value))


def contains_sensitive_text(value: object) -> bool:
    return isinstance(value, str) and bool(SENSITIVE_TEXT_PATTERN.search(value))


def contains_url(value: object) -> bool:
    return isinstance(value, str) and bool(URL_PATTERN.search(value))


def public_parameters(raw_parameters: object) -> list[dict[str, str]]:
    if not isinstance(raw_parameters, list):
        return []
    parameters: list[dict[str, str]] = []
    for raw in raw_parameters:
        if not isinstance(raw, Mapping):
            continue
        name = clean_text(raw.get("name"), limit=160)
        if not name or is_sensitive_name(name):
            continue
        parameter = {"name": name}
        parameter_type = clean_text(raw.get("type"), limit=120)
        description = clean_text(raw.get("description"), limit=400)
        if parameter_type:
            parameter["type"] = parameter_type
        if description and not contains_sensitive_text(description):
            parameter["description"] = description
        parameters.append(parameter)
        if len(parameters) >= MAX_PARAMETERS:
            break
    return parameters


def iter_operations(source_root: Path) -> Iterable[tuple[Path, Mapping[str, Any], Mapping[str, Any]]]:
    for path in sorted(source_root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping) or not isinstance(document.get("api_list"), list):
            continue
        for operation in document["api_list"]:
            if isinstance(operation, Mapping):
                yield path, document, operation


def workflow_record(path: Path, document: Mapping[str, Any], operation: Mapping[str, Any]) -> dict[str, Any] | None:
    document_sha256 = sha256_file(path)
    raw_operation_name = operation.get("name") or operation.get("title")
    raw_description = operation.get("description")
    raw_service_name = document.get("name") or document.get("title")
    operation_name = clean_text(raw_operation_name, limit=300)
    description = clean_text(raw_description)
    method = clean_text(operation.get("method"), limit=24).upper()
    endpoint_path = operation_path(operation.get("url") or operation.get("endpoint"))
    product_name = clean_text(raw_service_name, limit=300)
    if (
        not operation_name
        or not method
        or not endpoint_path
        or contains_url(raw_operation_name)
        or contains_url(raw_service_name)
        or any(contains_sensitive_text(value) for value in (raw_operation_name, raw_description, raw_service_name, endpoint_path))
    ):
        return None

    operation_context: dict[str, Any] = {
        "operation_name": operation_name,
        "method": method,
        "endpoint_path": endpoint_path,
        "required_parameters": public_parameters(operation.get("required_parameters")),
    }
    if description:
        operation_context["description"] = description
    if product_name:
        operation_context["service_name"] = product_name

    operation_digest = sha256_text(canonical_json(operation_context))
    group_digest = sha256_text(canonical_json({"document_sha256": document_sha256, "service_name": product_name}))
    return {
        "schema_version": 1,
        "workflow_id": f"public-api-operation-{operation_digest[:24]}",
        "group_id": f"public-api-document-{group_digest[:24]}",
        "source": SOURCE_NAME,
        "synthetic": False,
        "task": (
            "Review this documented public API operation for safe use in a service-composition workflow. "
            "Recommend the appropriate next action and identify any missing required inputs."
        ),
        "context": {
            "workflow_type": "public_api_document_review",
            "operation": operation_context,
            "evidence_scope": EVIDENCE_SCOPE,
        },
        "metadata": {
            "source_document_sha256": document_sha256,
            "operation_sha256": operation_digest,
            "input_provenance": "local_repository_public_api_document",
            "contains_observed_failure_trace": False,
            "contains_service_execution_outcome": False,
            "evidence_scope": EVIDENCE_SCOPE,
        },
    }


def build_workflows(source_root: Path, limit: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    workflows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    workflow_ids: set[str] = set()
    for path, document, operation in iter_operations(source_root):
        record = workflow_record(path, document, operation)
        if record is None:
            skipped["incomplete_operation"] += 1
            continue
        if record["workflow_id"] in workflow_ids:
            skipped["duplicate_operation"] += 1
            continue
        workflow_ids.add(record["workflow_id"])
        workflows.append(record)
        if len(workflows) >= limit:
            break
    return workflows, skipped


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build public API-document inputs for SHARE LLM proposal collection without claiming observed repair outcomes."
    )
    parser.add_argument("--source", type=Path, default=Path("dataset/api"), help="Directory containing local public API JSON documents.")
    parser.add_argument("--out", type=Path, default=Path("datasets/share_public_api_workflow_input.jsonl"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3000, help="Exact number of distinct API operations to retain.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.source.is_dir():
        parser.error("--source must be an existing directory")
    if args.summary is None:
        args.summary = args.out.with_name(args.out.stem + "_summary.json")
    if args.out.resolve() == args.summary.resolve():
        parser.error("--out and --summary must differ")
    return args


def main() -> None:
    args = parse_args()
    workflows, skipped = build_workflows(args.source, args.limit)
    if len(workflows) != args.limit:
        raise SystemExit(
            f"requested {args.limit} public API operations but found only {len(workflows)} usable distinct operations under {args.source}"
        )
    write_jsonl(args.out, workflows)
    summary = {
        "schema_version": 1,
        "kind": "share_public_api_workflow_input_summary",
        "input": str(args.out),
        "input_sha256": sha256_file(args.out),
        "workflows": len(workflows),
        "groups": len({workflow["group_id"] for workflow in workflows}),
        "source": SOURCE_NAME,
        "synthetic": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "warning": (
            "These are public API documentation review inputs. Completed LLM calls are real AI proposal artifacts, "
            "but are not observed service executions, paired Human/Stop outcomes, or paper-result evidence."
        ),
        "skipped": dict(sorted(skipped.items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
