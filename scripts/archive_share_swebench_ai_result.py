from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_EVAL_FILES = ("eval.sh", "patch.diff", "report.json", "run_instance.log", "test_output.txt")
REQUIRED_GENERATION_FILES = ("artifact.json", "model.patch", "prediction.jsonl", "transcript.jsonl")


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


def validate_generation(root: Path, instance_id: str, expected_model: str) -> Mapping[str, Any]:
    if (root / "QUARANTINED.json").exists():
        raise ValueError("AI generation is quarantined")
    if not all((root / name).is_file() for name in REQUIRED_GENERATION_FILES):
        raise ValueError("AI generation directory lacks a required file")
    artifact = read_json(root / "artifact.json")
    expected = {
        "schema_version": 1,
        "kind": "share_swebench_ai_patch_generation_artifact",
        "instance_id": instance_id,
        "branch": "ai",
        "model": expected_model,
        "clean_tracked_baseline_verified": True,
        "gold_patch_withheld": True,
        "test_patch_withheld": True,
        "generation_error": None,
        "patch_empty": False,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"AI generation has invalid {key}: expected {value!r}")
    files = artifact.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("AI generation artifact lacks file bindings")
    expected_files = {"model_patch": "model.patch", "prediction": "prediction.jsonl", "transcript": "transcript.jsonl"}
    for key, name in expected_files.items():
        binding = files.get(key)
        if not isinstance(binding, Mapping) or binding.get("path") != name:
            raise ValueError(f"AI generation lacks the expected {key} binding")
        if binding.get("sha256") != sha256_file(root / name):
            raise ValueError(f"AI generation {key} digest mismatch")
    prediction_lines = [line for line in (root / "prediction.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prediction_lines) != 1:
        raise ValueError("prediction JSONL must contain exactly one record")
    prediction = json.loads(prediction_lines[0])
    if prediction.get("instance_id") != instance_id or prediction.get("model_name_or_path") != expected_model:
        raise ValueError("prediction identity does not match the generation artifact")
    if prediction.get("model_patch") != (root / "model.patch").read_text(encoding="utf-8"):
        raise ValueError("prediction patch does not match model.patch")
    return artifact


def validate_evaluation(summary: Mapping[str, Any], detail: Mapping[str, Any], instance_id: str) -> Mapping[str, Any]:
    if summary.get("schema_version") != 2 or summary.get("submitted_ids") != [instance_id]:
        raise ValueError("AI evaluation summary has an invalid schema or instance binding")
    if summary.get("completed_ids") != [instance_id] or summary.get("incomplete_ids"):
        raise ValueError("AI evaluation did not complete the expected instance")
    if summary.get("error_ids") or summary.get("infra_failure_ids") or summary.get("ambiguous_failure_ids"):
        raise ValueError("AI evaluation contains an error or infrastructure failure")
    resolved = summary.get("resolved_ids") == [instance_id]
    unresolved = summary.get("unresolved_ids") == [instance_id]
    if resolved == unresolved:
        raise ValueError("AI evaluation must classify the instance as exactly resolved or unresolved")
    result = detail.get(instance_id)
    if not isinstance(result, Mapping) or result.get("resolved") is not resolved or result.get("infra_failure") is not False:
        raise ValueError("detailed AI report disagrees with the summary")
    if result.get("patch_successfully_applied") is not True:
        raise ValueError("AI patch was not successfully applied")
    tests_status = result.get("tests_status")
    if not isinstance(tests_status, Mapping):
        raise ValueError("detailed AI report lacks tests_status")
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        status = tests_status.get(category)
        if not isinstance(status, Mapping) or not isinstance(status.get("success"), list) or not isinstance(status.get("failure"), list):
            raise ValueError(f"detailed AI report has invalid {category} results")
    if resolved and (tests_status["FAIL_TO_PASS"]["failure"] or tests_status["PASS_TO_PASS"]["failure"]):
        raise ValueError("resolved AI report still contains a required test failure")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one completed SWE-bench AI branch as auditable SHARE evidence.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/run_evaluation"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite AI evidence directory: {args.out}")
    summary_path = args.report_dir / f"{args.model}.{args.run_id}.json"
    run_dir = args.logs_root / args.run_id / args.model / args.instance
    detail_path = run_dir / "report.json"
    image_provenance_path = args.pilot / "image_provenance" / f"{args.instance}.json"
    pilot_manifest_path = args.pilot / "pilot_manifest.json"
    try:
        generation = validate_generation(args.generation, args.instance, args.model)
        summary = read_json(summary_path)
        detail = read_json(detail_path)
        result = validate_evaluation(summary, detail, args.instance)
        image_provenance = read_json(image_provenance_path)
        pilot_manifest = read_json(pilot_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.instance not in (pilot_manifest.get("instance_ids") or []):
        raise SystemExit("pilot manifest does not contain the expected instance")
    if image_provenance.get("instance_id") != args.instance:
        raise SystemExit("image provenance does not match the expected instance")
    if generation.get("image_id") != image_provenance.get("image", {}).get("id"):
        raise SystemExit("AI generation image ID does not match image provenance")
    if not all((run_dir / name).is_file() for name in REQUIRED_EVAL_FILES):
        raise SystemExit("SWE-bench AI run directory lacks a required raw file")
    if sha256_file(run_dir / "patch.diff") != sha256_file(args.generation / "model.patch"):
        raise SystemExit("evaluated patch does not match the generated model patch")

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True)
    sources = {
        **{f"evaluation_{name}": run_dir / name for name in REQUIRED_EVAL_FILES},
        **{f"generation_{name}": args.generation / name for name in REQUIRED_GENERATION_FILES},
        "run_summary.json": summary_path,
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
        copied[name] = {"file": str(destination.relative_to(args.out)), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}

    tests = result["tests_status"]
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_ai_execution_artifact",
        "instance_id": args.instance,
        "branch": "ai",
        "evidence_role": "blind_model_patch_executed_by_official_swebench_harness",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "model": args.model,
        "profile": generation.get("profile"),
        "resolved": bool(result["resolved"]),
        "patch_successfully_applied": True,
        "model_patch_sha256": sha256_file(args.generation / "model.patch"),
        "clean_tracked_generation_baseline_verified": True,
        "gold_patch_withheld_during_generation": True,
        "test_patch_withheld_during_generation": True,
        "evaluation_image_may_contain_environment_setup_edits": True,
        "evaluated_patch_exactly_matches_model_patch": True,
        "tests": {
            "fail_to_pass_success": len(tests["FAIL_TO_PASS"]["success"]),
            "fail_to_pass_failure": len(tests["FAIL_TO_PASS"]["failure"]),
            "pass_to_pass_success": len(tests["PASS_TO_PASS"]["success"]),
            "pass_to_pass_failure": len(tests["PASS_TO_PASS"]["failure"]),
        },
        "image": image_provenance.get("image"),
        "raw_files": copied,
        "evidence_status": "human_stop_ai_branches_executed_triplet_not_yet_normalized",
    }
    artifact_path = args.out / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "share_swebench_ai_archive_manifest",
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "raw_file_count": len(copied),
        "evidence_status": artifact["evidence_status"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "instance_id": args.instance, "resolved": artifact["resolved"], "raw_files": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
