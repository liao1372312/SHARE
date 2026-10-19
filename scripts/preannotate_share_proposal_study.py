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
    PREANNOTATION_PROMPT_VERSION,
    StudyStore,
    annotation_profile_binding,
    call_annotation_with_rate_limit_retries,
    canonical_json,
    ensure_private_directory,
    load_human_only_calibration_manifest,
    preannotation_request,
    sha256_text,
    utc_now,
    validate_preannotation,
)


MAX_OUTPUT_TOKENS = 2048
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_BACKOFF_SECONDS = 900.0


class ProgressReporter:
    def __init__(self, total: int, *, enabled: bool, every: int):
        self.total = total
        self.enabled = enabled
        self.every = every
        self.updated = 0
        self.counts = {"completed": 0, "failed": 0, "uncertain": 0, "skipped": 0}

    def render(self, stage: str, *, force: bool = False) -> None:
        if not self.enabled or (not force and self.updated % self.every):
            return
        done = sum(self.counts.values())
        percentage = (100.0 * done / self.total) if self.total else 100.0
        print(
            "[SHARE preannotation] "
            f"{stage} | completed={self.counts['completed']} failed={self.counts['failed']} "
            f"uncertain={self.counts['uncertain']} skipped={self.counts['skipped']} "
            f"done={done}/{self.total} ({percentage:.1f}%)",
            file=sys.stderr,
            flush=True,
        )

    def record(self, status: str) -> None:
        if status in self.counts:
            self.counts[status] += 1
        self.updated += 1
        self.render("progress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one independent LLM review default per blinded SHARE case for expert confirmation."
    )
    parser.add_argument("--study", type=Path, required=True, help="Private SQLite study database.")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        required=True,
        help="Human-only calibration manifest created before any LLM defaults are shown.",
    )
    parser.add_argument("--env", type=Path, default=Path(".env"), help="Contains LLM_ANNOTATION_* only for the independent judge.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only run directory, or the same directory with --resume.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional hard cap. Omit to preannotate every study case.")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true", help="Resume the same manifest after a prior interruption or failure.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry failed judge calls.")
    parser.add_argument("--retry-uncertain", action="store_true", help="With --resume, retry calls interrupted after dispatch.")
    parser.add_argument("--max-rate-limit-retries", type=int, default=1)
    parser.add_argument("--rate-limit-backoff-seconds", type=float, default=70.0)
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Suppress sanitized stderr progress.")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate the study and configuration without contacting an LLM or changing the study.")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
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


def run_manifest(
    args: argparse.Namespace,
    store: StudyStore,
    profile_binding: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = store.metadata()
    return {
        "schema_version": 1,
        "kind": "share_llm_assisted_expert_audit_preannotation_run",
        "created_at_utc": utc_now(),
        "study_id": metadata.get("study_id"),
        "collection_id": metadata.get("collection_id"),
        "selected_cases": len(selected),
        "selected_case_ids_sha256": sha256_text(canonical_json([item["case"]["case_id"] for item in selected])),
        "case_sha256s_sha256": sha256_text(canonical_json([item["case_sha256"] for item in selected])),
        "calibration_manifest": {
            "sha256": calibration["sha256"],
            "reserved_case_ids_sha256": sha256_text(canonical_json(calibration["case_ids"])),
            "required_human_only_ratings_per_case": calibration["required_human_only_ratings_per_case"],
        },
        "profile_binding": dict(profile_binding),
        "prompt_version": PREANNOTATION_PROMPT_VERSION,
        "caps": {
            "max_cases": args.max_cases,
            "max_llm_requests": len(selected),
            "max_output_tokens": args.max_output_tokens,
            "timeout_seconds": args.timeout_seconds,
        },
        "warning": (
            "One independent LLM default is generated per blinded case and reused only as a visible, editable default. "
            "It is not a human judgment, service-execution outcome, paired Human/Stop outcome, or paper outcome evidence."
        ),
    }


def validate_resume_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError("--resume requires preannotation_manifest.json")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("preannotation manifest is invalid") from error
    fields = ("schema_version", "kind", "study_id", "collection_id", "selected_cases", "selected_case_ids_sha256", "case_sha256s_sha256", "calibration_manifest", "profile_binding", "prompt_version", "caps")
    if any(actual.get(field) != expected.get(field) for field in fields):
        raise ValueError("--resume manifest does not match the study, selected cases, judge profile, prompt, or caps")


def previous_is_compatible(previous: Mapping[str, Any], binding: Mapping[str, Any], request_sha256: str) -> bool:
    return previous.get("profile_binding") == dict(binding) and previous.get("request_sha256") == request_sha256


def run_preannotation(args: argparse.Namespace) -> dict[str, Any]:
    store = StudyStore(args.study)
    profile = load_annotation_profile(args.env)
    binding = annotation_profile_binding(profile)
    calibration = load_human_only_calibration_manifest(args.calibration_manifest, store)
    if not calibration["complete"]:
        raise ValueError(
            "human-only calibration is incomplete; require "
            f"{calibration['required_human_only_ratings_per_case']} independent ratings for every reserved case "
            f"({len(calibration['incomplete_case_ids'])}/{len(calibration['case_ids'])} cases remain)"
        )
    reserved_case_ids = set(calibration["case_ids"])
    available = [item for item in store.cases_for_preannotation() if item["case"]["case_id"] not in reserved_case_ids]
    selected = available if args.max_cases is None else available[: args.max_cases]
    if not selected:
        raise ValueError("study contains no LLM-reference cases outside the human-only calibration set")
    manifest = run_manifest(args, store, binding, selected, calibration)
    manifest_path = args.out / "preannotation_manifest.json"
    if args.out.exists() and not args.resume:
        raise ValueError(f"output already exists: {args.out}; use a fresh directory or --resume")
    ensure_private_directory(args.out)
    if args.resume:
        validate_resume_manifest(manifest_path, manifest)
    else:
        write_private_json(manifest_path, manifest)

    if args.dry_run:
        return {**manifest, "dry_run": True, "summary": {"planned": len(selected)}}

    progress = ProgressReporter(len(selected), enabled=args.progress, every=args.progress_every)
    progress.render("starting", force=True)
    rate_limit_retries = 0
    rate_limit_wait_seconds = 0.0
    for item in selected:
        case = item["case"]
        case_id = case["case_id"]
        request = preannotation_request(profile, case, args.max_output_tokens)
        request_sha256 = sha256_text(canonical_json(request))
        previous = store.case_preannotation(case_id)
        if previous is not None:
            if not previous_is_compatible(previous, binding, request_sha256):
                raise ValueError("stored LLM preannotation does not match the judge profile, prompt, or caps; create a separate study instead")
            previous_status = previous.get("status")
            if previous_status == "completed":
                progress.record("skipped")
                continue
            if not args.resume:
                raise ValueError("study has a prior unfinished LLM preannotation; use --resume after inspecting the prior run")
            if previous_status in {"started", "uncertain"} and not args.retry_uncertain:
                raise ValueError("found an uncertain LLM preannotation; pass --retry-uncertain explicitly")
            if previous_status == "failed" and not args.retry_failed:
                raise ValueError("found a failed LLM preannotation; pass --retry-failed explicitly")
        store.begin_preannotation(
            case_id,
            profile_binding=binding,
            prompt_sha256=sha256_text(canonical_json(preannotation_request(profile, case, args.max_output_tokens)["messages"])),
            request_sha256=request_sha256,
        )
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
                defaults = validate_preannotation(result["preannotation_response"], case)
            except ValueError:
                store.finish_preannotation(
                    case_id,
                    status="failed",
                    response_sha256=result.get("response_sha256"),
                    error_kind="invalid_preannotation_schema",
                    latency_ms=result.get("latency_ms"),
                )
                progress.record("failed")
                continue
            store.finish_preannotation(
                case_id,
                status="completed",
                preannotation=defaults,
                response_sha256=result.get("response_sha256"),
                latency_ms=result.get("latency_ms"),
            )
            progress.record("completed")
            continue
        store.finish_preannotation(
            case_id,
            status=result["status"],
            response_sha256=result.get("response_sha256"),
            error_kind=result.get("error_kind"),
            http_status=result.get("http_status"),
            latency_ms=result.get("latency_ms"),
        )
        progress.record(result["status"])
    progress.render("finished", force=True)
    return {
        **manifest,
        "dry_run": False,
        "completed_at_utc": utc_now(),
        "summary": store.preannotation_summary(),
        "prefillable_unsubmitted_assignments": store.prefillable_assignment_count(),
        "rate_limit_retries": rate_limit_retries,
        "rate_limit_wait_seconds": rate_limit_wait_seconds,
    }


def main() -> None:
    args = parse_args()
    try:
        report = run_preannotation(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_private_json(args.out / "preannotation_report.json", report)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "dry_run": report["dry_run"],
                "selected_cases": report["selected_cases"],
                "max_llm_requests": report["caps"]["max_llm_requests"],
                "summary": summary,
                "prefillable_unsubmitted_assignments": report.get("prefillable_unsubmitted_assignments", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
