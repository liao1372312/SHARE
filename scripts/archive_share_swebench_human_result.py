from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUN_FILES = ("eval.sh", "patch.diff", "report.json", "run_instance.log", "test_output.txt")


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


def validate_result(summary: Mapping[str, Any], detail: Mapping[str, Any], instance_id: str) -> Mapping[str, Any]:
    if summary.get("schema_version") != 2:
        raise ValueError("unsupported SWE-bench summary schema")
    if summary.get("submitted_ids") != [instance_id]:
        raise ValueError("SWE-bench run was not limited to the expected instance")
    if summary.get("completed_ids") != [instance_id] or summary.get("resolved_ids") != [instance_id]:
        raise ValueError("Human gold execution did not complete and resolve the expected instance")
    if summary.get("error_ids") or summary.get("infra_failure_ids") or summary.get("ambiguous_failure_ids"):
        raise ValueError("Human gold execution contains an error or infrastructure failure")
    result = detail.get(instance_id)
    if not isinstance(result, Mapping):
        raise ValueError("detailed report lacks the expected instance")
    if result.get("patch_successfully_applied") is not True or result.get("resolved") is not True or result.get("infra_failure") is not False:
        raise ValueError("detailed report does not certify a resolved Human gold execution")
    tests_status = result.get("tests_status")
    if not isinstance(tests_status, Mapping):
        raise ValueError("detailed report lacks tests_status")
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        status = tests_status.get(category)
        if not isinstance(status, Mapping) or status.get("failure") != [] or not isinstance(status.get("success"), list):
            raise ValueError(f"detailed report has a failing or invalid {category} result")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one successful SWE-bench gold run as raw SHARE Human evidence.")
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
        raise SystemExit(f"refusing to overwrite Human evidence directory: {args.out}")
    try:
        spec = read_spec(args.pilot / "execution_specs.jsonl", args.instance)
        pilot_manifest_path = args.pilot / "pilot_manifest.json"
        pilot_manifest = read_json(pilot_manifest_path)
        image_provenance_path = args.pilot / "image_provenance" / f"{args.instance}.json"
        image_provenance = read_json(image_provenance_path)
        summary_path = args.report_dir / f"gold.{args.run_id}.json"
        summary = read_json(summary_path)
        run_dir = args.logs_root / args.run_id / "gold" / args.instance
        detail_path = run_dir / "report.json"
        detail = read_json(detail_path)
        result = validate_result(summary, detail, args.instance)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if pilot_manifest.get("instance_ids") is None or args.instance not in pilot_manifest["instance_ids"]:
        raise SystemExit("pilot manifest does not contain the expected instance")
    if image_provenance.get("instance_id") != args.instance:
        raise SystemExit("image provenance does not match the expected instance")
    human_patch = args.pilot / str(spec["branches"]["human"]["patch_file"])
    if sha256_file(human_patch) != spec["branches"]["human"]["patch_sha256"]:
        raise SystemExit("pilot Human patch digest mismatch")
    if not all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES):
        raise SystemExit("SWE-bench run directory lacks a required raw file")
    if sha256_file(run_dir / "patch.diff") != sha256_file(human_patch):
        raise SystemExit("executed gold patch does not match the pilot Human patch")

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    sources = {
        **{name: run_dir / name for name in REQUIRED_RUN_FILES},
        "run_summary.json": summary_path,
        "run_metadata.json": args.logs_root / args.run_id / "run.json",
        "image_provenance.json": image_provenance_path,
        "pilot_manifest.json": pilot_manifest_path,
    }
    for name, source in sources.items():
        if not source.is_file():
            raise SystemExit(f"missing evidence source file: {source}")
        destination = raw_dir / name
        shutil.copy2(source, destination)
        copied[name] = {"file": str(destination.relative_to(args.out)), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}

    tests_status = result["tests_status"]
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_historical_human_execution_artifact",
        "instance_id": args.instance,
        "branch": "human",
        "evidence_role": "executed_historical_human_repair_proxy",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "resolved": True,
        "patch_successfully_applied": True,
        "tests": {
            "fail_to_pass_success": len(tests_status["FAIL_TO_PASS"]["success"]),
            "fail_to_pass_failure": 0,
            "pass_to_pass_success": len(tests_status["PASS_TO_PASS"]["success"]),
            "pass_to_pass_failure": 0,
        },
        "human_time": {
            "observed": False,
            "minutes": None,
            "difficulty_annotation": spec.get("difficulty"),
            "warning": "The merged patch is historical. No contemporaneous developer execution time was observed.",
        },
        "image": image_provenance.get("image"),
        "raw_files": copied,
        "evidence_status": "one_executed_human_branch_not_paired_outcomes",
    }
    artifact_path = args.out / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "share_swebench_human_archive_manifest",
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "raw_file_count": len(copied),
        "evidence_status": artifact["evidence_status"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "instance_id": args.instance, "resolved": True, "raw_files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
