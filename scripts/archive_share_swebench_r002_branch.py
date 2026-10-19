#!/usr/bin/env python3
"""Archive one fixed-attempt R002 AI branch, including failed generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/run_evaluation"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite R002 archive: {args.out}")
    try:
        protocol = read_json(args.protocol)
        generation = read_json(args.generation / "artifact.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    bindings = protocol.get("bindings", {})
    profiles = {row.get("name"): row for row in bindings.get("profiles", [])}
    tasks = {row.get("instance_id"): row for row in bindings.get("tasks", [])}
    expected_profile = profiles.get(args.profile)
    expected_task = tasks.get(args.instance)
    if not isinstance(expected_profile, dict) or not isinstance(expected_task, dict):
        raise SystemExit("branch is outside the frozen protocol")
    if generation.get("instance_id") != args.instance or generation.get("profile") != args.profile:
        raise SystemExit("generation identity differs from requested branch")
    if generation.get("model") != expected_profile.get("model") or generation.get("image") != expected_task.get("image"):
        raise SystemExit("generation model/image differs from frozen protocol")
    gp = generation.get("protocol") or {}
    parameters = bindings.get("parameters") or {}
    files = bindings.get("files") or {}
    expected_protocol = {
        "generator_source_sha256": files["scripts/generate_share_swebench_ai_patch.py"]["sha256"],
        "workflows_sha256": files["workflows"]["sha256"],
        "execution_specs_sha256": files["execution_specs"]["sha256"],
        "pilot_manifest_sha256": files["pilot_manifest"]["sha256"],
        "max_turns": parameters["max_turns"],
        "max_output_tokens": parameters["max_output_tokens"],
        "api_timeout_seconds": parameters["api_timeout_seconds"],
        "command_timeout_seconds": parameters["command_timeout_seconds"],
        "temperature": parameters["temperature"],
        "profile_reasoning_effort": expected_profile.get("reasoning_effort"),
    }
    for key, value in expected_protocol.items():
        if gp.get(key) != value:
            raise SystemExit(f"generation protocol mismatch: {key}")

    status = "generation_error" if generation.get("generation_error") else "empty_patch" if generation.get("patch_empty") else None
    evaluation: dict[str, Any] | None = None
    run_dir = args.logs_root / args.run_id / generation["model"] / args.instance
    summary_path = args.report_dir / f"{generation['model']}.{args.run_id}.json"
    detail_path = run_dir / "report.json"
    if status is None:
        try:
            summary = read_json(summary_path)
            detail = read_json(detail_path)
            row = detail[args.instance]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise SystemExit(f"missing or invalid fixed-attempt evaluation: {error}") from error
        if row.get("infra_failure") is not False:
            raise SystemExit("infrastructure failure cannot be classified as a model outcome")
        applied = row.get("patch_successfully_applied") is True
        resolved = args.instance in (summary.get("resolved_ids") or [])
        status = "resolved" if applied and resolved else "unresolved" if applied else "patch_not_applied"
        evaluation = {"patch_successfully_applied": applied, "resolved": resolved, "tests_status": row.get("tests_status")}

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True)
    sources = {f"generation_{name}": args.generation / name for name in ("artifact.json", "model.patch", "prediction.jsonl", "transcript.jsonl")}
    if evaluation is not None:
        sources.update({"evaluation_summary.json": summary_path, "evaluation_detail.json": detail_path})
        for name in ("eval.sh", "patch.diff", "run_instance.log", "test_output.txt"):
            if (run_dir / name).is_file():
                sources[f"evaluation_{name}"] = run_dir / name
    copied = {}
    for name, source in sources.items():
        if not source.is_file():
            raise SystemExit(f"missing R002 evidence input: {source}")
        destination = raw_dir / name
        shutil.copy2(source, destination)
        copied[name] = {"file": str(destination.relative_to(args.out)), "sha256": sha256_file(destination), "bytes": destination.stat().st_size}
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_r002_fixed_attempt_branch",
        "protocol_id": protocol.get("protocol_id"),
        "instance_id": args.instance,
        "profile": args.profile,
        "model": generation.get("model"),
        "status": status,
        "resolved": status == "resolved",
        "turns_used": generation.get("turns_used"),
        "usage": [json.loads(line).get("response_audit", {}).get("usage") for line in (args.generation / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()],
        "evaluation": evaluation,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": copied,
        "analysis_rule": "fixed_single_attempt_no_retry_substitution",
    }
    (args.out / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "instance_id": args.instance, "profile": args.profile, "status": status}, indent=2))


if __name__ == "__main__":
    main()
