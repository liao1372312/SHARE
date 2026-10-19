from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import load_annotation_profile
from share.proposal_annotation import (
    PREANNOTATION_SCHEMA_VERSION,
    SIMULATED_EXPERT_PERSONAS,
    StudyStore,
    annotation_profile_binding,
    call_annotation_with_rate_limit_retries,
    canonical_json,
    ensure_private_directory,
    sha256_text,
    simulated_expert_request,
    utc_now,
    validate_preannotation,
)


SIMULATION_PROMPT_VERSION = "share-llm-simulated-expert-v1"
MAX_OUTPUT_TOKENS = 2048
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_BACKOFF_SECONDS = 900.0


class ProgressReporter:
    def __init__(self, total: int, counts: Mapping[str, int], *, enabled: bool, every: int):
        self.total = total
        self.enabled = enabled
        self.every = every
        self.updated = 0
        self.counts = {status: int(counts.get(status, 0)) for status in ("completed", "failed", "uncertain", "skipped")}

    def render(self, stage: str, *, force: bool = False) -> None:
        if not self.enabled or (not force and self.updated % self.every):
            return
        done = sum(self.counts.values())
        percentage = (100.0 * done / self.total) if self.total else 100.0
        print(
            "[SHARE simulated-experts] "
            f"{stage} | completed={self.counts['completed']} failed={self.counts['failed']} "
            f"uncertain={self.counts['uncertain']} skipped={self.counts['skipped']} "
            f"done={done}/{self.total} ({percentage:.1f}%)",
            file=sys.stderr,
            flush=True,
        )

    def start(self) -> None:
        self.render("requesting", force=True)

    def record(self, status: str, *, previous_status: object = None) -> None:
        if isinstance(previous_status, str) and previous_status in self.counts:
            self.counts[previous_status] -= 1
        if status in self.counts:
            self.counts[status] += 1
        self.updated += 1
        self.render("progress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM-simulated expert assessments for SHARE without claiming a human study."
    )
    parser.add_argument("--study", type=Path, required=True, help="Private blinded study database; it is read only.")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="Contains independent LLM_ANNOTATION_* judge settings.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only output directory, or the same directory with --resume.")
    parser.add_argument("--simulation-id", default="share-llm-simulated-experts-v1")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap. Omit to assess every case in the study.")
    parser.add_argument("--simulated-reviewers-per-case", type=int, default=2, help="Number of predefined simulated reviewer slots per case.")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true", help="Resume the same simulation after an interruption or failure.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry a prior failed LLM call.")
    parser.add_argument("--retry-uncertain", action="store_true", help="With --resume, retry a call interrupted after dispatch.")
    parser.add_argument("--max-rate-limit-retries", type=int, default=1)
    parser.add_argument("--rate-limit-backoff-seconds", type=float, default=70.0)
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Suppress sanitized stderr progress.")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write a manifest without network calls.")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if not args.simulation_id.strip():
        parser.error("--simulation-id must be non-empty")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if not 1 <= args.simulated_reviewers_per_case <= len(SIMULATED_EXPERT_PERSONAS):
        parser.error(f"--simulated-reviewers-per-case must be between 1 and {len(SIMULATED_EXPERT_PERSONAS)}")
    if not 1 <= args.max_output_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"--max-output-tokens must be between 1 and {MAX_OUTPUT_TOKENS}")
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
    return args


def write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    ensure_private_directory(path.parent)
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
    ensure_private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def load_latest_index(path: Path) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return latest
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"simulation index has invalid JSON at line {line_number}") from error
        if not isinstance(item, Mapping) or not isinstance(item.get("execution_key"), str):
            raise ValueError(f"simulation index has an invalid record at line {line_number}")
        latest[item["execution_key"]] = item
    return latest


def load_assessments(path: Path) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"simulated assessment file has invalid JSON at line {line_number}") from error
        key = item.get("execution_key") if isinstance(item, Mapping) else None
        if not isinstance(key, str) or item.get("kind") != "share_llm_simulated_expert_assessment":
            raise ValueError(f"simulated assessment file has an invalid record at line {line_number}")
        values[key] = item
    return values


def count_statuses(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"started": 0, "completed": 0, "failed": 0, "uncertain": 0, "skipped": 0}
    for record in records:
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def execution_key(case: Mapping[str, Any], case_sha256: str, reviewer_slot: int, profile_binding: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "case_id": case["case_id"],
                "case_sha256": case_sha256,
                "reviewer_slot": reviewer_slot,
                "profile_binding": profile_binding,
                "request_sha256": sha256_text(canonical_json(request)),
            }
        )
    )


def run_manifest(args: argparse.Namespace, store: StudyStore, binding: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metadata = store.metadata()
    return {
        "schema_version": 1,
        "kind": "share_llm_simulated_expert_assessment_run",
        "simulation_id": args.simulation_id.strip(),
        "created_at_utc": utc_now(),
        "study_id": metadata.get("study_id"),
        "collection_id": metadata.get("collection_id"),
        "selected_cases": len(selected),
        "selected_case_ids_sha256": sha256_text(canonical_json([item["case"]["case_id"] for item in selected])),
        "case_sha256s_sha256": sha256_text(canonical_json([item["case_sha256"] for item in selected])),
        "profile_binding": dict(binding),
        "prompt_version": SIMULATION_PROMPT_VERSION,
        "simulated_reviewer_persona_count": args.simulated_reviewers_per_case,
        "caps": {
            "max_cases": args.max_cases,
            "max_llm_requests": len(selected) * args.simulated_reviewers_per_case,
            "max_output_tokens": args.max_output_tokens,
            "timeout_seconds": args.timeout_seconds,
        },
        "warning": (
            "These are LLM-simulated expert assessments. They are not human annotations, practitioner-study data, independent human ratings, "
            "service-execution outcomes, paired Human/Stop outcomes, or paper outcome evidence."
        ),
    }


def validate_resume_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError("--resume requires simulation_manifest.json")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("simulation manifest is invalid") from error
    fields = (
        "schema_version",
        "kind",
        "simulation_id",
        "study_id",
        "collection_id",
        "selected_cases",
        "selected_case_ids_sha256",
        "case_sha256s_sha256",
        "profile_binding",
        "prompt_version",
        "simulated_reviewer_persona_count",
        "caps",
    )
    if any(actual.get(field) != expected.get(field) for field in fields):
        raise ValueError("--resume manifest does not match the study, selected cases, judge profile, simulation prompt, or caps")


def index_record(
    *,
    args: argparse.Namespace,
    case: Mapping[str, Any],
    case_sha256: str,
    reviewer_slot: int,
    profile_binding: Mapping[str, Any],
    request: Mapping[str, Any],
    key: str,
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "share_llm_simulated_expert_assessment_index",
        "simulation_id": args.simulation_id.strip(),
        "execution_key": key,
        "case_id": case["case_id"],
        "workflow_id": case["workflow_id"],
        "case_sha256": case_sha256,
        "reviewer_slot": reviewer_slot,
        "profile_binding": dict(profile_binding),
        "request_sha256": sha256_text(canonical_json(request)),
        "status": status,
        **extra,
    }


def assessment_record(
    *,
    args: argparse.Namespace,
    case: Mapping[str, Any],
    case_sha256: str,
    reviewer_slot: int,
    profile_binding: Mapping[str, Any],
    request: Mapping[str, Any],
    key: str,
    assessment: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PREANNOTATION_SCHEMA_VERSION,
        "kind": "share_llm_simulated_expert_assessment",
        "simulation_id": args.simulation_id.strip(),
        "execution_key": key,
        "case_id": case["case_id"],
        "workflow_id": case["workflow_id"],
        "case_sha256": case_sha256,
        "reviewer_slot": reviewer_slot,
        "profile_binding": dict(profile_binding),
        "request_sha256": sha256_text(canonical_json(request)),
        "completed_at_utc": utc_now(),
        "latency_ms": result.get("latency_ms"),
        "response_sha256": result.get("response_sha256"),
        "assessment": dict(assessment),
        "assessment_sha256": sha256_text(canonical_json(assessment)),
        "warning": "LLM-simulated expert assessment only; not a human label or service-execution outcome.",
    }


def validate_simulated_assessment(response: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    assessment = validate_preannotation(response, case)
    assessment["kind"] = "share_llm_simulated_expert_assessment_default"
    assessment["prompt_version"] = SIMULATION_PROMPT_VERSION
    assessment["warning"] = "LLM-simulated expert assessment only; not a human label or service-execution outcome."
    return assessment


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    store = StudyStore(args.study, migrate=False)
    profile = load_annotation_profile(args.env)
    binding = annotation_profile_binding(profile)
    available = store.cases_for_preannotation()
    selected = available if args.max_cases is None else available[: args.max_cases]
    if not selected:
        raise ValueError("study contains no cases")
    manifest = run_manifest(args, store, binding, selected)
    manifest_path = args.out / "simulation_manifest.json"
    index_path = args.out / "simulation_index.jsonl"
    assessments_path = args.out / "simulated_assessments.jsonl"
    if args.out.exists() and not args.resume:
        raise ValueError(f"output already exists: {args.out}; use a fresh directory or --resume")
    ensure_private_directory(args.out)
    if args.resume:
        validate_resume_manifest(manifest_path, manifest)
    else:
        write_private_json(manifest_path, manifest)
    if args.dry_run:
        return {**manifest, "dry_run": True, "summary": {"planned": manifest["caps"]["max_llm_requests"]}}

    latest = load_latest_index(index_path)
    assessments = load_assessments(assessments_path)
    progress = ProgressReporter(manifest["caps"]["max_llm_requests"], count_statuses(list(latest.values())), enabled=args.progress, every=args.progress_every)
    progress.render("starting", force=True)
    rate_limit_retries = 0
    rate_limit_wait_seconds = 0.0
    for item in selected:
        case = item["case"]
        case_sha256 = item["case_sha256"]
        for reviewer_slot in range(1, args.simulated_reviewers_per_case + 1):
            request = simulated_expert_request(profile, case, reviewer_slot, args.max_output_tokens)
            key = execution_key(case, case_sha256, reviewer_slot, binding, request)
            previous = latest.get(key)
            prior_status = previous.get("status") if previous is not None else None
            if previous is not None:
                same_request = previous.get("request_sha256") == sha256_text(canonical_json(request))
                if previous.get("profile_binding") != binding or not same_request:
                    raise ValueError("stored simulated assessment does not match the judge profile, prompt, case, or caps")
                previous_status = previous.get("status")
                if previous_status == "completed":
                    assessment = assessments.get(key)
                    if not isinstance(assessment, Mapping) or assessment.get("assessment_sha256") != previous.get("assessment_sha256"):
                        raise ValueError("completed simulated assessment is missing or does not match the index")
                    continue
                if not args.resume:
                    raise ValueError("simulation has a prior unfinished LLM call; use --resume after inspecting the prior run")
                if previous_status in {"started", "uncertain"} and not args.retry_uncertain:
                    raise ValueError("found an uncertain simulated assessment; pass --retry-uncertain explicitly")
                if previous_status in {"failed", "skipped"} and not args.retry_failed:
                    raise ValueError("found a failed simulated assessment; pass --retry-failed explicitly")
            progress.start()
            started = index_record(
                args=args,
                case=case,
                case_sha256=case_sha256,
                reviewer_slot=reviewer_slot,
                profile_binding=binding,
                request=request,
                key=key,
                status="started",
                dispatched_at_utc=utc_now(),
            )
            append_private_jsonl(index_path, started)
            latest[key] = started
            result, retries, wait_seconds = call_annotation_with_rate_limit_retries(
                profile,
                request,
                args.timeout_seconds,
                max_retries=args.max_rate_limit_retries,
                backoff_seconds=args.rate_limit_backoff_seconds,
            )
            rate_limit_retries += retries
            rate_limit_wait_seconds += wait_seconds
            if result["status"] == "completed":
                try:
                    assessment = validate_simulated_assessment(result["preannotation_response"], case)
                except ValueError:
                    record = index_record(
                        args=args,
                        case=case,
                        case_sha256=case_sha256,
                        reviewer_slot=reviewer_slot,
                        profile_binding=binding,
                        request=request,
                        key=key,
                        status="failed",
                        finished_at_utc=utc_now(),
                        error_kind="invalid_simulated_assessment_schema",
                        response_sha256=result.get("response_sha256"),
                        latency_ms=result.get("latency_ms"),
                    )
                    append_private_jsonl(index_path, record)
                    latest[key] = record
                    progress.record("failed", previous_status=prior_status)
                    continue
                completed = assessment_record(
                    args=args,
                    case=case,
                    case_sha256=case_sha256,
                    reviewer_slot=reviewer_slot,
                    profile_binding=binding,
                    request=request,
                    key=key,
                    assessment=assessment,
                    result=result,
                )
                append_private_jsonl(assessments_path, completed)
                record = index_record(
                    args=args,
                    case=case,
                    case_sha256=case_sha256,
                    reviewer_slot=reviewer_slot,
                    profile_binding=binding,
                    request=request,
                    key=key,
                    status="completed",
                    completed_at_utc=completed["completed_at_utc"],
                    assessment_sha256=completed["assessment_sha256"],
                    response_sha256=completed["response_sha256"],
                    latency_ms=completed["latency_ms"],
                    simulated_assessment_file=assessments_path.name,
                )
                append_private_jsonl(index_path, record)
                latest[key] = record
                progress.record("completed", previous_status=prior_status)
                continue
            record = index_record(
                args=args,
                case=case,
                case_sha256=case_sha256,
                reviewer_slot=reviewer_slot,
                profile_binding=binding,
                request=request,
                key=key,
                status=result["status"],
                finished_at_utc=utc_now(),
                error_kind=result.get("error_kind"),
                http_status=result.get("http_status"),
                response_sha256=result.get("response_sha256"),
                latency_ms=result.get("latency_ms"),
            )
            append_private_jsonl(index_path, record)
            latest[key] = record
            progress.record(result["status"], previous_status=prior_status)
    progress.render("finished", force=True)
    return {
        **manifest,
        "dry_run": False,
        "completed_at_utc": utc_now(),
        "summary": count_statuses(list(latest.values())),
        "rate_limit_retries": rate_limit_retries,
        "rate_limit_wait_seconds": rate_limit_wait_seconds,
        "assessment_file": assessments_path.name,
    }


def main() -> None:
    args = parse_args()
    try:
        report = run_simulation(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_private_json(args.out / "simulation_report.json", report)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "dry_run": report["dry_run"],
                "selected_cases": report["selected_cases"],
                "simulated_reviewers_per_case": report["simulated_reviewer_persona_count"],
                "max_llm_requests": report["caps"]["max_llm_requests"],
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
