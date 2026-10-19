from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUN_FILES = ("eval.sh", "report.json", "run_instance.log", "test_output.txt")
MODEL_NAME = "share-stop-no-solution-patch"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON file {path}: {error}") from error


def read_spec(path: Path, instance_id: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"execution spec line {line_number} is not JSON") from error
            if isinstance(value, Mapping) and value.get("instance_id") == instance_id:
                return value
    raise ValueError(f"instance is not in pilot execution specs: {instance_id}")


def validate_binding(binding: Mapping[str, Any], instance_id: str, run_id: str) -> None:
    expected = {
        "schema_version": 1,
        "kind": "share_swebench_stop_run_binding",
        "instance_id": instance_id,
        "run_id": run_id,
        "semantics": "base_commit_tests_without_solution_patch",
        "solution_patch_applied": False,
        "harness_skip_patch": True,
        "resolved": False,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"Stop binding has invalid {key}: expected {value!r}")


def validate_result(
    summary: Mapping[str, Any], detail: Mapping[str, Any], instance_id: str
) -> Mapping[str, Any]:
    if summary.get("schema_version") != 2:
        raise ValueError("unsupported SWE-bench summary schema")
    if summary.get("submitted_ids") != [instance_id]:
        raise ValueError("SWE-bench Stop run was not limited to the expected instance")
    if summary.get("completed_ids") != [instance_id] or summary.get("unresolved_ids") != [instance_id]:
        raise ValueError("Stop execution did not complete unresolved for the expected instance")
    if summary.get("resolved_ids"):
        raise ValueError("Stop execution unexpectedly resolved the instance")
    if summary.get("error_ids") or summary.get("infra_failure_ids") or summary.get("ambiguous_failure_ids"):
        raise ValueError("Stop execution contains an error or infrastructure failure")
    result = detail.get(instance_id)
    if not isinstance(result, Mapping):
        raise ValueError("detailed report lacks the expected instance")
    if result.get("resolved") is not False or result.get("infra_failure") is not False:
        raise ValueError("detailed report does not certify a valid unresolved Stop execution")
    tests_status = result.get("tests_status")
    if not isinstance(tests_status, Mapping):
        raise ValueError("detailed report lacks tests_status")
    fail_to_pass = tests_status.get("FAIL_TO_PASS")
    pass_to_pass = tests_status.get("PASS_TO_PASS")
    if not isinstance(fail_to_pass, Mapping) or fail_to_pass.get("success") != []:
        raise ValueError("Stop execution unexpectedly passed a FAIL_TO_PASS test")
    if not isinstance(fail_to_pass.get("failure"), list) or not fail_to_pass["failure"]:
        raise ValueError("Stop execution does not show the target failure")
    if not isinstance(pass_to_pass, Mapping) or pass_to_pass.get("failure") != []:
        raise ValueError("Stop execution regressed a PASS_TO_PASS test")
    if not isinstance(pass_to_pass.get("success"), list):
        raise ValueError("Stop execution has invalid PASS_TO_PASS results")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one valid SWE-bench no-solution-patch Stop run.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/run_evaluation"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite Stop evidence directory: {args.out}")
    summary_path = args.report_dir / f"{MODEL_NAME}.{args.run_id}.json"
    binding_path = args.report_dir / f"{MODEL_NAME}.{args.run_id}.binding.json"
    run_dir = args.logs_root / args.run_id / MODEL_NAME / args.instance
    detail_path = run_dir / "report.json"
    try:
        spec = read_spec(args.pilot / "execution_specs.jsonl", args.instance)
        pilot_manifest_path = args.pilot / "pilot_manifest.json"
        pilot_manifest = read_json(pilot_manifest_path)
        image_provenance_path = args.pilot / "image_provenance" / f"{args.instance}.json"
        image_provenance = read_json(image_provenance_path)
        binding = read_json(binding_path)
        summary = read_json(summary_path)
        detail = read_json(detail_path)
        validate_binding(binding, args.instance, args.run_id)
        result = validate_result(summary, detail, args.instance)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.instance not in (pilot_manifest.get("instance_ids") or []):
        raise SystemExit("pilot manifest does not contain the expected instance")
    if image_provenance.get("instance_id") != args.instance:
        raise SystemExit("image provenance does not match the expected instance")
    if spec.get("branches", {}).get("stop", {}).get("semantics") != "execute_at_base_commit_without_solution_patch":
        raise SystemExit("pilot execution spec does not define the required Stop semantics")
    if binding.get("report_sha256") != sha256_file(summary_path):
        raise SystemExit("Stop summary digest does not match its binding")
    if binding.get("detail_report_sha256") != sha256_file(detail_path):
        raise SystemExit("Stop detailed report digest does not match its binding")
    if not all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES):
        raise SystemExit("SWE-bench Stop run directory lacks a required raw file")

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True)
    sources = {
        **{name: run_dir / name for name in REQUIRED_RUN_FILES},
        "run_summary.json": summary_path,
        "run_binding.json": binding_path,
        "run_metadata.json": args.logs_root / args.run_id / "run.json",
        "image_provenance.json": image_provenance_path,
        "pilot_manifest.json": pilot_manifest_path,
    }
    copied: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise SystemExit(f"missing evidence source file: {source}")
        destination = raw_dir / name
        shutil.copy2(source, destination)
        copied[name] = {
            "file": str(destination.relative_to(args.out)),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }

    tests_status = result["tests_status"]
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_stop_execution_artifact",
        "instance_id": args.instance,
        "branch": "stop",
        "evidence_role": "executed_base_commit_no_solution_patch",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "semantics": binding["semantics"],
        "resolved": False,
        "solution_patch_applied": False,
        "harness_skip_patch": True,
        "generic_report_patch_fields_ignored": True,
        "tests": {
            "fail_to_pass_success": len(tests_status["FAIL_TO_PASS"]["success"]),
            "fail_to_pass_failure": len(tests_status["FAIL_TO_PASS"]["failure"]),
            "pass_to_pass_success": len(tests_status["PASS_TO_PASS"]["success"]),
            "pass_to_pass_failure": len(tests_status["PASS_TO_PASS"]["failure"]),
        },
        "image": image_provenance.get("image"),
        "raw_files": copied,
        "evidence_status": "human_and_stop_executed_ai_branch_pending",
        "warning": binding.get("warning"),
    }
    artifact_path = args.out / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "share_swebench_stop_archive_manifest",
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "raw_file_count": len(copied),
        "evidence_status": artifact["evidence_status"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "instance_id": args.instance, "resolved": False, "raw_files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
