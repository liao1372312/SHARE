from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOUND_SOURCES = (
    "scripts/refine_share_swebench_ai_patch.py",
    "scripts/run_share_swebench_r002b_refinement.sh",
    "scripts/freeze_share_swebench_r002b_protocol.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build(args: argparse.Namespace) -> dict:
    r002_manifest = args.r002_pilot / "protocol_manifest.json"
    r002 = load_json(r002_manifest)
    if r002.get("kind") != "share_swebench_r002_controlled_protocol":
        raise ValueError("input is not the frozen R002 protocol")
    tasks = r002.get("bindings", {}).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 11:
        raise ValueError("R002b requires all 11 frozen R002 tasks")
    initial_patches = []
    for task in tasks:
        instance = task["instance_id"]
        patch = args.r002_pilot / "observed_r002_ai" / instance / "primary" / "raw" / "generation_model.patch"
        if not patch.is_file() or not patch.read_text(encoding="utf-8").strip():
            raise ValueError(f"missing non-empty primary R002 patch: {instance}")
        initial_patches.append({"instance_id": instance, "path": str(patch), "sha256": sha256_file(patch)})
    profiles = load_json(args.profiles)
    primary = next((row for row in profiles.get("profiles", []) if row.get("name") == "primary"), None)
    if primary is None:
        raise ValueError("primary profile is missing")
    bindings = {
        "parent_protocol_id": r002["protocol_id"],
        "parent_protocol_sha256": sha256_file(r002_manifest),
        "tasks": tasks,
        "initial_patches": initial_patches,
        "profile": {key: primary.get(key) for key in ("name", "model", "reasoning_effort")},
        "parameters": {
            "refinement_attempts_per_task": 1,
            "max_turns": 20,
            "max_output_tokens": 4096,
            "api_timeout_seconds": 180,
            "command_timeout_seconds": 300,
            "evaluation_timeout_seconds": 1800,
            "temperature": 0.0,
            "requires_successful_test_after_latest_edit": True,
            "requires_diff_inspection_after_latest_edit": True,
        },
        "files": {
            str(path): {"path": str(path), "sha256": sha256_file(path)}
            for path in [r002_manifest, args.profiles, *(ROOT / value for value in BOUND_SOURCES)]
        },
        "analysis_rule": "Run the same second-stage refiner on all 11 tasks. Count generation errors, empty patches, non-admissible finals, non-applicable patches, and evaluation failures as unresolved. No retry substitution.",
        "feedback_boundary": "The refiner may inspect repository files and run repository tests, but receives no R002 evaluation summary, benchmark test labels, gold patch, or test patch.",
    }
    protocol_id = "r002b_protocol_v2_" + hashlib.sha256(canonical(bindings)).hexdigest()[:16]
    return {
        "schema_version": 1,
        "kind": "share_swebench_r002b_verified_refinement_protocol",
        "protocol_id": protocol_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "environment": {
            "python": platform.python_version(),
            "docker_server": subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], text=True, capture_output=True, check=True).stdout.strip(),
        },
        "evidence_boundary": "Exploratory same-task system refinement after R002 outcomes were inspected; not a confirmatory backbone ranking and not compute-matched to one-attempt baselines.",
    }


def comparable(value: dict) -> dict:
    copy = dict(value)
    copy.pop("frozen_at_utc", None)
    return copy


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the exploratory R002b refinement protocol.")
    parser.add_argument("--r002-pilot", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = build(args)
    if args.verify:
        observed = load_json(args.out)
        if comparable(observed) != comparable(expected):
            raise SystemExit("R002b protocol drift detected; refusing to run")
        print(json.dumps({"verified": str(args.out), "protocol_id": observed["protocol_id"]}, indent=2))
        return
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite R002b protocol: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "protocol_id": expected["protocol_id"], "tasks": 11}, indent=2))


if __name__ == "__main__":
    main()
