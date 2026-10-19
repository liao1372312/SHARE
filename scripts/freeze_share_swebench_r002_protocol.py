#!/usr/bin/env python3
"""Freeze or verify the protocol-uniform SHARE R002 SWE-bench experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_FILES = (
    "scripts/generate_share_swebench_ai_patch.py",
    "scripts/archive_share_swebench_r002_branch.py",
    "scripts/run_share_swebench_r002_controlled.sh",
    "scripts/freeze_share_swebench_r002_protocol.py",
)
PARAMETERS = {
    "attempts_per_task_model": 1,
    "max_turns": 20,
    "max_output_tokens": 2048,
    "api_timeout_seconds": 180,
    "command_timeout_seconds": 180,
    "evaluation_timeout_seconds": 1800,
    "temperature": 0.0,
    "workers": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"command failed: {command[0]}")
    return result.stdout.strip()


def build_bindings(args: argparse.Namespace) -> dict[str, Any]:
    pilot_manifest_path = args.pilot / "pilot_manifest.json"
    execution_specs_path = args.pilot / "execution_specs.jsonl"
    pilot_workflows_path = args.pilot / "workflows.jsonl"
    pilot_manifest = read_json(pilot_manifest_path)
    specs = read_jsonl(execution_specs_path)
    profiles_report = read_json(args.profiles)
    profiles = [
        {"name": row["name"], "model": row["model"], "reasoning_effort": row.get("reasoning_effort")}
        for row in profiles_report.get("profiles", [])
    ]
    if [row["name"] for row in profiles] != ["primary", "backbone_2", "backbone_3"]:
        raise ValueError("profiles must be primary, backbone_2, and backbone_3 in that order")
    instance_ids = pilot_manifest.get("instance_ids")
    if not isinstance(instance_ids, list) or len(instance_ids) != 11 or len(set(instance_ids)) != 11:
        raise ValueError("R002 requires exactly 11 unique preselected instances")
    specs_by_id = {row.get("instance_id"): row for row in specs}
    tasks = []
    for instance_id in instance_ids:
        spec = specs_by_id.get(instance_id)
        if not isinstance(spec, dict) or not isinstance(spec.get("image"), str):
            raise ValueError(f"missing execution spec for {instance_id}")
        tasks.append(
            {
                "instance_id": instance_id,
                "image": spec["image"],
                "base_commit": spec.get("base_commit"),
                "group_id": spec.get("group_id"),
            }
        )
    files = {
        "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
        "pilot_manifest": {"path": str(pilot_manifest_path), "sha256": sha256_file(pilot_manifest_path)},
        "execution_specs": {"path": str(execution_specs_path), "sha256": sha256_file(execution_specs_path)},
        "workflows": {"path": str(pilot_workflows_path), "sha256": sha256_file(pilot_workflows_path)},
        "profiles": {"path": str(args.profiles), "sha256": sha256_file(args.profiles)},
    }
    for relative in SOURCE_FILES:
        path = Path(relative)
        files[relative] = {"path": relative, "sha256": sha256_file(path)}
    return {
        "tasks": tasks,
        "profiles": profiles,
        "parameters": PARAMETERS,
        "files": files,
        "dataset_revision": pilot_manifest.get("source_revision"),
        "selection_rule": pilot_manifest.get("selection_rule"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        bindings = build_bindings(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.verify:
        if not args.out.is_file():
            raise SystemExit(f"protocol manifest is missing: {args.out}")
        existing = read_json(args.out)
        if existing.get("bindings") != bindings:
            raise SystemExit("R002 protocol drift detected; refusing to run")
        print(json.dumps({"verified": True, "protocol_id": existing.get("protocol_id"), "tasks": 11, "branches": 33}, indent=2))
        return
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {args.out}")
    canonical = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "share_swebench_r002_controlled_protocol",
        "protocol_id": "r002_protocol_v1_" + hashlib.sha256(canonical).hexdigest()[:16],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "environment": {
            "python": platform.python_version(),
            "swebench": command_output([".venv-swebench/bin/python", "-c", "import swebench; print(swebench.__version__)"]),
            "docker_server": command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
        },
        "analysis_rule": "Count every fixed single attempt. Generation errors, empty patches, and non-applicable patches are unresolved; no retry substitution.",
        "evidence_boundary": "Controlled AI-backbone comparison on historical SWE-bench tasks; Human gold is a proxy and contemporaneous human time is unavailable.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "protocol_id": manifest["protocol_id"], "tasks": 11, "branches": 33}, indent=2))


if __name__ == "__main__":
    main()
