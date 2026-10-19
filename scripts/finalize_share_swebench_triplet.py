from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_branch(root: Path, instance_id: str, branch: str) -> tuple[Mapping[str, Any], str]:
    branch_root = root / branch
    artifact_path = branch_root / "artifact.json"
    manifest_path = branch_root / "manifest.json"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"{branch} branch archive is incomplete")
    artifact = read_json(artifact_path)
    manifest = read_json(manifest_path)
    digest = sha256_file(artifact_path)
    if manifest.get("artifact_file") != "artifact.json" or manifest.get("artifact_sha256") != digest:
        raise ValueError(f"{branch} branch manifest digest mismatch")
    if artifact.get("instance_id") != instance_id or artifact.get("branch") != branch:
        raise ValueError(f"{branch} branch identity mismatch")
    if not isinstance(artifact.get("resolved"), bool):
        raise ValueError(f"{branch} branch lacks a Boolean resolved outcome")
    return artifact, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind archived Human, Stop, and AI SWE-bench outcomes into one triplet manifest.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite triplet manifest: {args.out}")
    root = args.pilot / "observed_branches" / args.instance
    try:
        artifacts: dict[str, Mapping[str, Any]] = {}
        digests: dict[str, str] = {}
        for branch in ("human", "stop", "ai"):
            artifacts[branch], digests[branch] = load_branch(root, args.instance, branch)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if artifacts["stop"].get("solution_patch_applied") is not False or artifacts["stop"]["resolved"] is not False:
        raise SystemExit("Stop branch does not represent an unresolved no-solution-patch baseline")
    image_ids = {artifact.get("image", {}).get("id") for artifact in artifacts.values()}
    if len(image_ids) != 1 or None in image_ids:
        raise SystemExit("branch archives do not bind to the same image ID")
    human_time = artifacts["human"].get("human_time")
    human_time_observed = isinstance(human_time, Mapping) and human_time.get("observed") is True
    triplet = {
        "schema_version": 1,
        "kind": "share_swebench_executed_branch_triplet_manifest",
        "instance_id": args.instance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_id": next(iter(image_ids)),
        "branches": {
            branch: {
                "artifact_file": f"observed_branches/{args.instance}/{branch}/artifact.json",
                "artifact_sha256": digests[branch],
                "resolved": artifacts[branch]["resolved"],
            }
            for branch in ("human", "stop", "ai")
        },
        "outcomes_executed": True,
        "ready_for_repair_success_analysis": True,
        "human_time_observed": human_time_observed,
        "ready_for_human_time_claims": human_time_observed,
        "repair_episode_created": False,
        "evidence_status": "executed_triplet_human_time_missing" if not human_time_observed else "executed_triplet_ready_for_normalization",
        "warning": None if human_time_observed else "Historical Human repair time is unavailable; do not impute it or use this triplet for human-time claims without a declared external model.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(triplet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "outcomes": {key: value["resolved"] for key, value in artifacts.items()}, "human_time_observed": human_time_observed}, indent=2))


if __name__ == "__main__":
    main()
