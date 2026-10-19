from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import LLMProfile, load_llm_profiles


INPUT_SCHEMA_VERSION = 1
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_CHARS = 24_000
MAX_RATE_LIMIT_RETRIES = 5
MAX_RATE_LIMIT_BACKOFF_SECONDS = 300.0


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward bearer credentials to an unexpected redirect target."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


class ProgressReporter:
    """Render collection status without exposing workflow or credential data."""

    def __init__(self, total: int, counts: Mapping[str, int], *, enabled: bool, every: int) -> None:
        self.total = total
        self.counts = {status: int(counts.get(status, 0)) for status in ("completed", "failed", "uncertain", "skipped")}
        self.enabled = enabled
        self.every = every
        self.last_profile: str | None = None
        self.updated = 0
        self.started_at = time.monotonic()
        self.interactive = sys.stderr.isatty()

    def render(self, *, prefix: str, force: bool = False) -> None:
        if not self.enabled or (not force and self.updated % self.every != 0):
            return
        completed = self.counts["completed"]
        failed = self.counts["failed"]
        pending = self.total - completed - failed
        elapsed_seconds = int(time.monotonic() - self.started_at)
        fields = [
            f"[SHARE] {prefix}",
            f"completed={completed}/{self.total} ({completed / self.total:.1%})",
            f"failed={failed}",
            f"uncertain={self.counts['uncertain']}",
            f"skipped={self.counts['skipped']}",
            f"pending={pending}",
            f"elapsed={elapsed_seconds}s",
        ]
        if self.last_profile is not None:
            fields.append(f"profile={self.last_profile}")
        line = " | ".join(fields)
        if self.interactive:
            print(f"\r{line:<180}", end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    def start_request(self, profile: LLMProfile, previous_status: object) -> None:
        if isinstance(previous_status, str) and previous_status in self.counts:
            self.counts[previous_status] -= 1
        self.last_profile = profile.name
        self.render(prefix="requesting", force=True)

    def record(self, profile: LLMProfile, status: str) -> None:
        if status in self.counts:
            self.counts[status] += 1
        self.last_profile = profile.name
        self.updated += 1
        self.render(prefix="progress")

    def finish(self) -> None:
        self.render(prefix="finished", force=True)
        if self.enabled and self.interactive:
            print(file=sys.stderr, flush=True)


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def profile_binding(profile: LLMProfile) -> dict[str, Any]:
    """Bind a run to the exact API root without retaining the endpoint itself."""
    parsed = urlparse(profile.base_url)
    default_port = 443 if parsed.scheme == "https" else 80
    normalized_root = {
        "scheme": parsed.scheme.lower(),
        "host": (parsed.hostname or "").lower(),
        "port": parsed.port or default_port,
        "path": parsed.path.rstrip("/") or "/",
        "params": parsed.params,
        "query": parsed.query,
    }
    return {
        "name": profile.name,
        "model": profile.model,
        "reasoning_effort": profile.reasoning_effort,
        "api_root_sha256": sha256_text(canonical_json(normalized_root)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect auditable, real AI execution artifacts across all configured SHARE LLM profiles."
    )
    parser.add_argument("--input", type=Path, required=True, help="Preselected real-workflow JSONL using schema version 1.")
    parser.add_argument("--collection-id", required=True, help="External collection or preregistration identifier.")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory or a resumable collection directory.")
    parser.add_argument("--max-workflows", type=int, required=True, help="Hard cap on input workflows; each selected workflow is sent to all profiles.")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-input-chars", type=int, default=12_000)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true", help="Resume only an output directory whose manifest matches the current input and profiles.")
    parser.add_argument("--retry-failed", action="store_true", help="Required with --resume before retrying a profile that previously returned an error.")
    parser.add_argument("--retry-uncertain", action="store_true", help="Required with --resume before retrying a request recorded as started without a final result.")
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=0,
        help="Maximum in-process retries after HTTP 429. Defaults to 0 so retries are always explicit.",
    )
    parser.add_argument(
        "--rate-limit-backoff-seconds",
        type=float,
        default=60.0,
        help="Minimum wait before each explicit HTTP 429 retry; a valid provider Retry-After value can increase it.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request OpenAI-compatible response_format=json_object. Use only after a representative provider check.",
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Suppress terminal progress on stderr.")
    parser.add_argument("--progress-every", type=int, default=1, help="Render progress after this many status updates. Defaults to 1.")
    parser.add_argument("--idempotency-header", default=None, help="Optional provider-supported header name for the stable execution key.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write a manifest without contacting any LLM endpoint.")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if not args.collection_id.strip():
        parser.error("--collection-id must be non-empty")
    if args.max_workflows < 1:
        parser.error("--max-workflows must be positive")
    if not 1 <= args.max_output_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"--max-output-tokens must be between 1 and {MAX_OUTPUT_TOKENS}")
    if not 1 <= args.max_input_chars <= MAX_INPUT_CHARS:
        parser.error(f"--max-input-chars must be between 1 and {MAX_INPUT_CHARS}")
    if not 1 <= args.timeout_seconds <= 90:
        parser.error("--timeout-seconds must be between 1 and 90")
    if not 0 <= args.max_rate_limit_retries <= MAX_RATE_LIMIT_RETRIES:
        parser.error(f"--max-rate-limit-retries must be between 0 and {MAX_RATE_LIMIT_RETRIES}")
    if not 1 <= args.rate_limit_backoff_seconds <= MAX_RATE_LIMIT_BACKOFF_SECONDS:
        parser.error(f"--rate-limit-backoff-seconds must be between 1 and {int(MAX_RATE_LIMIT_BACKOFF_SECONDS)}")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if (args.retry_failed or args.retry_uncertain) and not args.resume:
        parser.error("--retry-failed and --retry-uncertain require --resume")
    if args.idempotency_header is not None and not re.fullmatch(r"[A-Za-z0-9-]+", args.idempotency_header):
        parser.error("--idempotency-header must contain only letters, digits, and hyphens")
    return args


def load_real_workflows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("--input must be one existing JSONL file")
    workflows = []
    workflow_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"input line {line_number} is not JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"input line {line_number} must be a JSON object")
            required = ("schema_version", "workflow_id", "group_id", "source", "synthetic", "task")
            missing = [field for field in required if field not in value]
            if missing:
                raise ValueError(f"input line {line_number} is missing {', '.join(missing)}")
            if value["schema_version"] != INPUT_SCHEMA_VERSION:
                raise ValueError(f"input line {line_number} has unsupported schema_version")
            if value["synthetic"] is not False:
                raise ValueError(f"input line {line_number} must set synthetic=false")
            if value.get("template_only") is True:
                raise ValueError(f"input line {line_number} is a template, not a real workflow")
            fields = {name: value[name] for name in ("workflow_id", "group_id", "source", "task")}
            if not all(isinstance(item, str) and item.strip() for item in fields.values()):
                raise ValueError(f"input line {line_number} has an empty workflow_id, group_id, source, or task")
            if fields["workflow_id"] in workflow_ids:
                raise ValueError(f"duplicate workflow_id: {fields['workflow_id']}")
            context = value.get("context", {})
            metadata = value.get("metadata", {})
            if not isinstance(context, Mapping) or not isinstance(metadata, Mapping):
                raise ValueError(f"input line {line_number} context and metadata must be JSON objects")
            workflow_ids.add(fields["workflow_id"])
            workflows.append(
                {
                    "schema_version": INPUT_SCHEMA_VERSION,
                    "workflow_id": fields["workflow_id"],
                    "group_id": fields["group_id"],
                    "source": fields["source"],
                    "synthetic": False,
                    "task": fields["task"],
                    "context": dict(context),
                    "metadata": dict(metadata),
                    "input_record_sha256": sha256_text(canonical_json(value)),
                }
            )
    if not workflows:
        raise ValueError("--input contains no workflows")
    return workflows


def select_workflows(workflows: Sequence[dict[str, Any]], max_workflows: int) -> list[dict[str, Any]]:
    return list(workflows[:max_workflows])


def prompt_payload(workflow: Mapping[str, Any], max_input_chars: int) -> tuple[dict[str, Any], bool]:
    workflow_input = {
        "workflow_id": workflow["workflow_id"],
        "task": workflow["task"],
        "context": workflow["context"],
    }
    serialized_input = canonical_json(workflow_input)
    truncated = len(serialized_input) > max_input_chars
    if truncated:
        serialized_input = serialized_input[:max_input_chars] + "\n[TRUNCATED]"
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are evaluating a real service-composition workflow for research collection. "
                    "Do not execute tools or claim that an action was performed. Return only a JSON object with "
                    "action, rationale, confidence, and assumptions."
                ),
            },
            {"role": "user", "content": "Workflow input:\n" + serialized_input},
        ],
        "temperature": 0.0,
    }
    return payload, truncated


def request_payload(
    profile: LLMProfile,
    workflow: Mapping[str, Any],
    max_input_chars: int,
    max_output_tokens: int,
    *,
    json_mode: bool,
) -> tuple[dict[str, Any], bool]:
    payload, truncated = prompt_payload(workflow, max_input_chars)
    payload = {"model": profile.model, **payload, "max_tokens": max_output_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if profile.reasoning_effort is not None:
        payload["reasoning_effort"] = profile.reasoning_effort
    return payload, truncated


def response_content(response: Mapping[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def content_is_json(content: str | None) -> bool:
    if content is None:
        return False
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return False
    return True


def retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    headers = error.headers
    raw_value = headers.get("Retry-After") if headers is not None else None
    if not isinstance(raw_value, str):
        return None
    try:
        value = float(raw_value.strip())
    except ValueError:
        return None
    return value if 0 <= value <= MAX_RATE_LIMIT_BACKOFF_SECONDS else None


def call_profile(
    profile: LLMProfile,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    idempotency_header: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {profile.api_key}", "Content-Type": "application/json"}
    if idempotency_header is not None and idempotency_key is not None:
        headers[idempotency_header] = idempotency_key
    request = urllib.request.Request(
        profile.base_url.rstrip("/") + "/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        opener = urllib.request.build_opener(NoRedirectHandler())
        with opener.open(request, timeout=timeout_seconds) as raw_response:
            response = json.loads(raw_response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 429:
            return {
                "status": "failed",
                "error_kind": "rate_limited",
                "http_status": error.code,
                "retry_after_seconds": retry_after_seconds(error),
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            }
        status = "uncertain" if error.code == 408 or 500 <= error.code <= 599 else "failed"
        return {"status": status, "error_kind": "http_error", "http_status": error.code, "latency_ms": round((time.monotonic() - started) * 1000, 1)}
    except urllib.error.URLError:
        return {"status": "uncertain", "error_kind": "transport_error", "latency_ms": round((time.monotonic() - started) * 1000, 1)}
    except (TimeoutError, OSError):
        return {"status": "uncertain", "error_kind": "transport_error", "latency_ms": round((time.monotonic() - started) * 1000, 1)}
    except json.JSONDecodeError:
        return {"status": "failed", "error_kind": "invalid_json_body", "latency_ms": round((time.monotonic() - started) * 1000, 1)}
    if not isinstance(response, Mapping):
        return {"status": "failed", "error_kind": "invalid_chat_completion_response", "latency_ms": round((time.monotonic() - started) * 1000, 1), "response": response}
    content = response_content(response)
    if content is None:
        return {"status": "failed", "error_kind": "invalid_chat_completion_response", "latency_ms": round((time.monotonic() - started) * 1000, 1), "response": response}
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    if not content_is_json(content):
        return {
            "status": "failed",
            "error_kind": "invalid_proposal_json",
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "response": response,
            "content_is_json": False,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }
    return {
        "status": "completed",
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "response": response,
        "response_content": content,
        "content_is_json": content_is_json(content),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def call_with_rate_limit_retries(
    profile: LLMProfile,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    max_retries: int,
    backoff_seconds: float,
    idempotency_header: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], int, float]:
    result = call_profile(
        profile,
        payload,
        timeout_seconds,
        idempotency_header=idempotency_header,
        idempotency_key=idempotency_key,
    )
    retries = 0
    total_wait_seconds = 0.0
    while result.get("error_kind") == "rate_limited" and retries < max_retries:
        provider_wait = result.get("retry_after_seconds")
        wait_seconds = max(backoff_seconds, provider_wait if isinstance(provider_wait, (int, float)) else 0.0)
        time.sleep(wait_seconds)
        total_wait_seconds += wait_seconds
        retries += 1
        result = call_profile(
            profile,
            payload,
            timeout_seconds,
            idempotency_header=idempotency_header,
            idempotency_key=idempotency_key,
        )
    return result, retries, total_wait_seconds


def write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def append_private_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def execution_key(workflow: Mapping[str, Any], profile: LLMProfile) -> str:
    return sha256_text(canonical_json({"workflow_input_sha256": workflow["input_record_sha256"], "profile": profile_binding(profile)}))


def collection_manifest(args: argparse.Namespace, profiles: Sequence[LLMProfile], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identifiers = [workflow["workflow_id"] for workflow in selected]
    return {
        "schema_version": 1,
        "kind": "share_real_llm_execution_collection",
        "evidence_kind": "real_ai_execution_artifacts_not_paired_outcomes",
        "collection_id": args.collection_id.strip(),
        "created_at_utc": utc_now(),
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "selected_workflows": len(selected),
        "selected_workflow_ids_sha256": sha256_text(canonical_json(identifiers)),
        "profiles": [profile_binding(profile) for profile in profiles],
        "caps": {
            "max_workflows": args.max_workflows,
            "max_requests": len(selected) * len(profiles),
            "max_output_tokens": args.max_output_tokens,
            "max_input_chars": args.max_input_chars,
            "timeout_seconds": args.timeout_seconds,
            "idempotency_header": args.idempotency_header,
            "json_mode": args.json_mode,
        },
        "execution_index": "execution_index.jsonl",
        "artifact_directory": "artifacts",
        "warning": (
            "Each completed record is an AI proposal artifact only. It is not a service execution outcome, "
            "a Human/Stop counterfactual, a RepairEpisode, or paper evidence until externally paired and validated."
        ),
    }


def load_latest_index(path: Path) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                key = value.get("execution_key")
                if isinstance(key, str):
                    latest[key] = value
    return latest


def validate_resume_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError("--resume requires collection_manifest.json")
    actual = json.loads(path.read_text(encoding="utf-8"))
    fields = ("schema_version", "kind", "evidence_kind", "collection_id", "input_sha256", "selected_workflow_ids_sha256", "profiles", "caps")
    if any(actual.get(field) != expected.get(field) for field in fields):
        raise ValueError("--resume manifest does not match the current collection input, profiles, or caps")


def validate_completed_artifact(out: Path, workflow: Mapping[str, Any], profile: LLMProfile, key: str, record: Mapping[str, Any]) -> None:
    relative_path = record.get("execution_artifact_file")
    expected_digest = record.get("execution_artifact_sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
        raise ValueError("completed index record lacks an artifact path or digest")
    artifact_root = (out / "artifacts").resolve()
    artifact_path = (out / relative_path).resolve()
    try:
        artifact_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("completed artifact path escapes the collection artifact directory") from error
    if not artifact_path.is_file() or sha256_file(artifact_path) != expected_digest:
        raise ValueError("completed artifact is missing or does not match its recorded digest")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("completed artifact is not valid JSON") from error
    if not isinstance(artifact, Mapping):
        raise ValueError("completed artifact must be a JSON object")
    artifact_workflow = artifact.get("workflow")
    if (
        artifact.get("kind") != "share_real_llm_execution_artifact"
        or artifact.get("execution_key") != key
        or artifact.get("profile") != profile_binding(profile)
        or not isinstance(artifact_workflow, Mapping)
        or artifact_workflow.get("input_record_sha256") != workflow["input_record_sha256"]
    ):
        raise ValueError("completed artifact does not match the current workflow/profile binding")


def index_record(
    workflow: Mapping[str, Any],
    profile: LLMProfile,
    key: str,
    request: Mapping[str, Any],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "share_real_llm_execution_index",
        "execution_key": key,
        "workflow_id": workflow["workflow_id"],
        "group_id": workflow["group_id"],
        "workflow_input_sha256": workflow["input_record_sha256"],
        "profile": profile.name,
        "model": profile.model,
        "profile_binding": profile_binding(profile),
        "request_sha256": sha256_text(canonical_json(request)),
        "status": status,
        **extra,
    }


def collection_summary(index_path: Path) -> dict[str, int]:
    counts = {"started": 0, "completed": 0, "failed": 0, "uncertain": 0, "skipped": 0}
    if not index_path.is_file():
        return counts
    for record in load_latest_index(index_path).values():
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def summary_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"started": 0, "completed": 0, "failed": 0, "uncertain": 0, "skipped": 0}
    for record in records:
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def execution_artifact(
    args: argparse.Namespace,
    workflow: Mapping[str, Any],
    profile: LLMProfile,
    key: str,
    request: Mapping[str, Any],
    truncated: bool,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "share_real_llm_execution_artifact",
        "collection_id": args.collection_id.strip(),
        "execution_key": key,
        "workflow": workflow,
        "profile": profile_binding(profile),
        "request": request,
        "request_sha256": sha256_text(canonical_json(request)),
        "input_truncated": truncated,
        "completed_at_utc": utc_now(),
        "status": result["status"],
        "latency_ms": result["latency_ms"],
        "response": result.get("response"),
        "content_is_json": result.get("content_is_json"),
    }


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    workflows = load_real_workflows(args.input)
    selected = select_workflows(workflows, args.max_workflows)
    profiles = load_llm_profiles(args.env)
    manifest = collection_manifest(args, profiles, selected)
    manifest_path = args.out / "collection_manifest.json"
    index_path = args.out / "execution_index.jsonl"
    if args.out.exists() and not args.resume:
        raise ValueError(f"output already exists: {args.out}; use a fresh directory or --resume")
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.out, 0o700)
    if args.resume:
        validate_resume_manifest(manifest_path, manifest)
    else:
        write_private_json(manifest_path, manifest)

    if args.dry_run:
        report = {**manifest, "dry_run": True, "summary": {"planned": len(selected) * len(profiles)}}
        write_private_json(args.out / "collection_report.json", report)
        return report

    latest = load_latest_index(index_path)
    progress = ProgressReporter(
        len(selected) * len(profiles),
        summary_from_records(list(latest.values())),
        enabled=getattr(args, "progress", True),
        every=getattr(args, "progress_every", 1),
    )
    progress.render(prefix="resuming" if args.resume else "starting", force=True)
    unavailable_profiles: set[str] = set()
    for workflow in selected:
        for profile in profiles:
            request, truncated = request_payload(
                profile,
                workflow,
                args.max_input_chars,
                args.max_output_tokens,
                json_mode=args.json_mode,
            )
            key = execution_key(workflow, profile)
            previous = latest.get(key)
            if previous is not None:
                previous_status = previous.get("status")
                if previous_status == "completed":
                    validate_completed_artifact(args.out, workflow, profile, key, previous)
                    continue
                if previous_status in {"started", "uncertain"} and not args.retry_uncertain:
                    raise ValueError("found an uncertain started request; inspect the index or pass --retry-uncertain explicitly")
                if previous_status in {"failed", "skipped"} and not args.retry_failed:
                    raise ValueError("found a prior failed/skipped request; pass --retry-failed explicitly to retry it")
            if profile.name in unavailable_profiles:
                record = index_record(
                    workflow,
                    profile,
                    key,
                    request,
                    "skipped",
                    recorded_at_utc=utc_now(),
                    reason="profile_failed_earlier_in_this_run",
                )
                append_private_jsonl(index_path, record)
                latest[key] = record
                if previous is None:
                    progress.record(profile, "skipped")
                continue
            progress.start_request(profile, previous.get("status") if previous is not None else None)
            started_record = index_record(workflow, profile, key, request, "started", dispatched_at_utc=utc_now(), input_truncated=truncated)
            append_private_jsonl(index_path, started_record)
            result, rate_limit_retries, rate_limit_wait_seconds = call_with_rate_limit_retries(
                profile,
                request,
                args.timeout_seconds,
                max_retries=args.max_rate_limit_retries,
                backoff_seconds=args.rate_limit_backoff_seconds,
                idempotency_header=args.idempotency_header,
                idempotency_key=key,
            )
            if result["status"] != "completed":
                artifact_fields: dict[str, Any] = {}
                if "response" in result:
                    artifact = execution_artifact(args, workflow, profile, key, request, truncated, result)
                    artifact_path = args.out / "artifacts" / f"{key}.json"
                    write_private_json(artifact_path, artifact)
                    artifact_fields = {
                        "execution_artifact_file": str(artifact_path.relative_to(args.out)),
                        "execution_artifact_sha256": sha256_file(artifact_path),
                    }
                record = index_record(
                    workflow,
                    profile,
                    key,
                    request,
                    result["status"],
                    finished_at_utc=utc_now(),
                    latency_ms=result["latency_ms"],
                    error_kind=result["error_kind"],
                    http_status=result.get("http_status"),
                    input_truncated=truncated,
                    content_is_json=result.get("content_is_json"),
                    response_sha256=sha256_text(canonical_json(result["response"])) if isinstance(result.get("response"), Mapping) else None,
                    usage=result.get("usage"),
                    rate_limit_retries=rate_limit_retries,
                    rate_limit_wait_seconds=rate_limit_wait_seconds,
                    **artifact_fields,
                )
                append_private_jsonl(index_path, record)
                latest[key] = record
                unavailable_profiles.add(profile.name)
                progress.record(profile, result["status"])
                continue
            artifact = execution_artifact(args, workflow, profile, key, request, truncated, result)
            artifact_path = args.out / "artifacts" / f"{key}.json"
            write_private_json(artifact_path, artifact)
            response = result["response"]
            response_id = response.get("id") if isinstance(response, Mapping) else None
            record = index_record(
                workflow,
                profile,
                key,
                request,
                "completed",
                completed_at_utc=artifact["completed_at_utc"],
                latency_ms=result["latency_ms"],
                input_truncated=truncated,
                content_is_json=result["content_is_json"],
                response_sha256=sha256_text(canonical_json(response)),
                response_id_sha256=sha256_text(str(response_id)) if response_id else None,
                usage=result["usage"],
                rate_limit_retries=rate_limit_retries,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                execution_artifact_file=str(artifact_path.relative_to(args.out)),
                execution_artifact_sha256=sha256_file(artifact_path),
            )
            append_private_jsonl(index_path, record)
            latest[key] = record
            progress.record(profile, "completed")
    report = {
        **manifest,
        "completed_at_utc": utc_now(),
        "dry_run": False,
        "summary": collection_summary(index_path),
    }
    write_private_json(args.out / "collection_report.json", report)
    progress.finish()
    return report


def main() -> None:
    args = parse_args()
    try:
        report = run_collection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "out": str(args.out),
                "dry_run": report["dry_run"],
                "selected_workflows": report["selected_workflows"],
                "max_requests": report["caps"]["max_requests"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
