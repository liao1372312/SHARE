from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EASY_DIFFICULTY = "<15 min fix"
OUTPUT_FILES = ("workflows.jsonl", "human_reference_patches.jsonl", "execution_specs.jsonl", "pilot_manifest.json")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing input file: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def select_pilot(rows: Iterable[Mapping[str, Any]], tasks: int) -> list[Mapping[str, Any]]:
    by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("difficulty") == EASY_DIFFICULTY:
            repo = row.get("repo")
            instance_id = row.get("instance_id")
            if isinstance(repo, str) and repo and isinstance(instance_id, str) and instance_id:
                by_repo[repo].append(row)
    selected = []
    for repo in sorted(by_repo):
        selected.append(sorted(by_repo[repo], key=lambda row: str(row["instance_id"]))[0])
        if len(selected) == tasks:
            break
    if len(selected) < tasks:
        raise ValueError(f"only {len(selected)} repositories contain an eligible {EASY_DIFFICULTY!r} task")
    return selected


def select_explicit_instances(rows: Iterable[Mapping[str, Any]], instance_ids: list[str]) -> list[Mapping[str, Any]]:
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("explicit replacement instance IDs must be unique")
    by_id = {row.get("instance_id"): row for row in rows}
    selected = []
    for instance_id in instance_ids:
        row = by_id.get(instance_id)
        if row is None:
            raise ValueError(f"explicit instance is absent from the pinned Parquet: {instance_id}")
        if row.get("difficulty") != EASY_DIFFICULTY:
            raise ValueError(f"explicit instance is not an eligible {EASY_DIFFICULTY!r} task: {instance_id}")
        selected.append(row)
    repositories = [row.get("repo") for row in selected]
    if len(repositories) != len(set(repositories)):
        raise ValueError("explicit replacement instances must come from distinct repositories")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an auditable, cross-repository SWE-bench pilot.")
    parser.add_argument("--parquet", type=Path, required=True, help="Pinned SWE-bench Verified Parquet file.")
    parser.add_argument("--staging", type=Path, required=True, help="Output from prepare_share_swebench.py.")
    parser.add_argument("--out", type=Path, required=True, help="New pilot directory.")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--instance", action="append", default=[], help="Select an explicit eligible instance; repeat for a replacement set.")
    args = parser.parse_args()
    if not 1 <= args.tasks <= 12:
        parser.error("--tasks must be between 1 and 12")
    return args


def main() -> None:
    args = parse_args()
    if not args.parquet.is_file():
        raise SystemExit("--parquet must be an existing file")
    existing = [args.out / name for name in OUTPUT_FILES if (args.out / name).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing pilot output: {existing[0]}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("pilot creation requires pyarrow") from error
    try:
        staging_report = json.loads((args.staging / "preparation_report.json").read_text(encoding="utf-8"))
        workflows = read_jsonl(args.staging / "workflows.jsonl")
        references = read_jsonl(args.staging / "human_reference_patches.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if staging_report.get("source_fingerprint") != sha256_file(args.parquet):
        raise SystemExit("Parquet digest does not match the staging preparation report")
    if staging_report.get("workflows_sha256") != sha256_file(args.staging / "workflows.jsonl"):
        raise SystemExit("staging workflow digest mismatch")
    if staging_report.get("human_references_sha256") != sha256_file(args.staging / "human_reference_patches.jsonl"):
        raise SystemExit("staging human-reference digest mismatch")

    table_rows = parquet.read_table(args.parquet).to_pylist()
    try:
        selected = select_explicit_instances(table_rows, args.instance) if args.instance else select_pilot(table_rows, args.tasks)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    workflow_by_id = {row.get("workflow_id"): row for row in workflows}
    reference_by_id = {row.get("workflow_id"): row for row in references}
    selected_ids = [str(row["instance_id"]) for row in selected]
    if any(instance_id not in workflow_by_id or instance_id not in reference_by_id for instance_id in selected_ids):
        raise SystemExit("selected pilot task is missing from the bound staging files")

    args.out.mkdir(parents=True, exist_ok=True)
    input_root = args.out / "branch_inputs"
    input_root.mkdir()
    specs = []
    for row in selected:
        instance_id = str(row["instance_id"])
        required = ("image", "eval_script", "test_patch", "patch", "eval_type", "log_parser")
        if any(not isinstance(row.get(field), str) or not row[field] for field in required):
            raise SystemExit(f"{instance_id} lacks an executable field in the pinned Parquet")
        task_dir = input_root / instance_id
        task_dir.mkdir()
        files = {
            "human_patch": (task_dir / "human.patch", row["patch"]),
            "test_patch": (task_dir / "test.patch", row["test_patch"]),
            "eval_script": (task_dir / "eval.sh", row["eval_script"]),
        }
        for path, content in files.values():
            path.write_text(content, encoding="utf-8")
        specs.append(
            {
                "schema_version": 1,
                "kind": "share_swebench_pilot_execution_spec",
                "instance_id": instance_id,
                "repo": row["repo"],
                "difficulty": row["difficulty"],
                "image": row["image"],
                "eval_type": row["eval_type"],
                "log_parser": row["log_parser"],
                "branches": {
                    "human": {"patch_file": str(files["human_patch"][0].relative_to(args.out)), "patch_sha256": sha256_file(files["human_patch"][0])},
                    "stop": {"patch_file": None, "semantics": "execute_at_base_commit_without_solution_patch"},
                    "ai": {"patch_file": None, "status": "awaiting_model_patch"},
                },
                "test_patch_file": str(files["test_patch"][0].relative_to(args.out)),
                "test_patch_sha256": sha256_file(files["test_patch"][0]),
                "eval_script_file": str(files["eval_script"][0].relative_to(args.out)),
                "eval_script_sha256": sha256_file(files["eval_script"][0]),
            }
        )

    workflow_path = args.out / "workflows.jsonl"
    reference_path = args.out / "human_reference_patches.jsonl"
    specs_path = args.out / "execution_specs.jsonl"
    write_jsonl(workflow_path, (workflow_by_id[instance_id] for instance_id in selected_ids))
    write_jsonl(reference_path, (reference_by_id[instance_id] for instance_id in selected_ids))
    write_jsonl(specs_path, specs)
    manifest = {
        "schema_version": 1,
        "kind": "share_swebench_pilot_manifest",
        "selection_rule": "explicit_eligible_instance_ids_v1" if args.instance else "lexicographically_first_easy_task_per_repository_v1",
        "difficulty": EASY_DIFFICULTY,
        "tasks": len(selected_ids),
        "repositories": len({row["repo"] for row in selected}),
        "instance_ids": selected_ids,
        "source_parquet_sha256": sha256_file(args.parquet),
        "source_dataset": staging_report.get("dataset"),
        "source_revision": staging_report.get("source_revision"),
        "workflows_sha256": sha256_file(workflow_path),
        "human_references_sha256": sha256_file(reference_path),
        "execution_specs_sha256": sha256_file(specs_path),
        "evidence_status": "execution_ready_pilot_inputs_not_outcomes",
    }
    (args.out / "pilot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "tasks": len(selected_ids), "instance_ids": selected_ids}, indent=2))


if __name__ == "__main__":
    main()
