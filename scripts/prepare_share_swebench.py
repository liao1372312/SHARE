from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
OUTPUT_FILES = ("workflows.jsonl", "human_reference_patches.jsonl", "preparation_report.json")
REQUIRED_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_test_ids(value: object, *, field: str, instance_id: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{instance_id} has non-JSON {field}") from error
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{instance_id} has invalid {field}; expected a JSON array of non-empty strings")
    return value


def validate_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"SWE-bench row is missing {', '.join(missing)}")
    instance_id = raw["instance_id"]
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("SWE-bench row has an invalid instance_id")
    for field in ("repo", "base_commit", "problem_statement", "patch", "test_patch"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"{instance_id} has an empty {field}")
    return {
        "instance_id": instance_id.strip(),
        "repo": raw["repo"].strip(),
        "base_commit": raw["base_commit"].strip(),
        "problem_statement": raw["problem_statement"].strip(),
        "patch": raw["patch"],
        "test_patch": raw["test_patch"],
        "fail_to_pass": parse_test_ids(raw["FAIL_TO_PASS"], field="FAIL_TO_PASS", instance_id=instance_id),
        "pass_to_pass": parse_test_ids(raw["PASS_TO_PASS"], field="PASS_TO_PASS", instance_id=instance_id),
        "version": str(raw.get("version", "")).strip(),
        "environment_setup_commit": str(raw.get("environment_setup_commit", "")).strip(),
    }


def load_local_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError("--input must be an existing JSON, JSONL, or Parquet file")
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ValueError("Parquet input requires the optional 'pyarrow' package") from error
        try:
            table = parquet.read_table(path)
        except Exception as error:
            raise ValueError(f"could not read Parquet input: {error}") from error
        for row in table.to_pylist():
            if not isinstance(row, Mapping):
                raise ValueError("Parquet input contains a non-object row")
            yield row
        return
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"input line {line_number} is not JSON") from error
                if not isinstance(row, Mapping):
                    raise ValueError(f"input line {line_number} must be a JSON object")
                yield row
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("--input is not valid JSON") from error
    if isinstance(value, Mapping) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list):
        raise ValueError("JSON input must be an array or an object with a data array")
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"input item {index} must be a JSON object")
        yield row


def load_huggingface_rows(
    dataset_name: str,
    split: str,
    revision: str | None = None,
    *,
    endpoint: str | None = None,
    download_timeout: int = 120,
) -> Any:
    # huggingface_hub reads these settings when it is first imported.
    if endpoint is not None:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(download_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("Hugging Face loading requires the optional 'datasets' package") from error
    try:
        dataset = load_dataset(dataset_name, split=split, revision=revision)
    except Exception as error:
        raise ValueError(f"could not load Hugging Face dataset {dataset_name!r} split {split!r}: {error}") from error
    return dataset


def prepare_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    dataset_split: str,
    limit: int | None,
    dataset_revision: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workflows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = validate_row(raw)
        instance_id = row["instance_id"]
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        source_binding = {
            "dataset": dataset_name,
            "dataset_split": dataset_split,
            "dataset_revision": dataset_revision,
            "instance_id": instance_id,
            "repo": row["repo"],
            "base_commit": row["base_commit"],
        }
        workflows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "workflow_id": instance_id,
                "group_id": row["repo"],
                "source": "swe-bench",
                "synthetic": False,
                "task": row["problem_statement"],
                "context": {
                    "repository": row["repo"],
                    "base_commit": row["base_commit"],
                    "version": row["version"],
                    "environment_setup_commit": row["environment_setup_commit"],
                    "evaluation_test_patch_sha256": sha256_text(row["test_patch"]),
                    "fail_to_pass": row["fail_to_pass"],
                    "pass_to_pass": row["pass_to_pass"],
                },
                "metadata": {
                    "evidence_role": "historical_human_repair_proxy_staging",
                    "source_binding": source_binding,
                    "gold_patch_withheld": True,
                    "not_a_repair_episode": True,
                },
            }
        )
        references.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "share_historical_human_reference_patch",
                "workflow_id": instance_id,
                "source_binding": source_binding,
                "patch": row["patch"],
                "patch_sha256": sha256_text(row["patch"]),
                "test_patch": row["test_patch"],
                "test_patch_sha256": sha256_text(row["test_patch"]),
                "fail_to_pass": row["fail_to_pass"],
                "pass_to_pass": row["pass_to_pass"],
                "evidence_role": "historical_human_repair_proxy",
                "human_minutes": None,
                "execution_status": "not_executed",
            }
        )
        if limit is not None and len(workflows) >= limit:
            break
    if not workflows:
        raise ValueError("no SWE-bench rows were selected")
    return workflows, references


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leak-resistant SWE-bench staging inputs for SHARE branch execution."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Local SWE-bench JSON, JSONL, or Parquet export.")
    source.add_argument("--dataset", default=None, help=f"Hugging Face dataset name (default: {DEFAULT_DATASET}).")
    parser.add_argument("--source-dataset", default=None, help="Dataset ID represented by a local --input export.")
    parser.add_argument("--source-revision", default=None, help="Immutable dataset revision represented by local --input.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split (default: {DEFAULT_SPLIT}).")
    parser.add_argument("--revision", default=None, help="Optional immutable Hugging Face dataset commit or tag.")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HTTPS Hugging Face-compatible mirror endpoint.")
    parser.add_argument("--download-timeout", type=int, default=120, help="Hub metadata and download timeout in seconds.")
    parser.add_argument("--out", type=Path, required=True, help="New or empty staging directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic prefix size for a pilot.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.hf_endpoint is not None and not args.hf_endpoint.startswith("https://"):
        parser.error("--hf-endpoint must use HTTPS")
    if not 10 <= args.download_timeout <= 3600:
        parser.error("--download-timeout must be between 10 and 3600 seconds")
    if args.input is None and (args.source_dataset is not None or args.source_revision is not None):
        parser.error("--source-dataset and --source-revision require --input")
    return args


def main() -> None:
    args = parse_args()
    existing_outputs = [args.out / name for name in OUTPUT_FILES if (args.out / name).exists()]
    if existing_outputs:
        raise SystemExit(f"refusing to overwrite existing staging output: {existing_outputs[0]}")
    dataset_name = args.dataset or DEFAULT_DATASET
    try:
        if args.input is not None:
            rows = load_local_rows(args.input)
            source_kind = "local_export"
            dataset_name = args.source_dataset or "local_swe_bench_export"
            source_revision = args.source_revision
            source_fingerprint = sha256_file(args.input)
        else:
            rows = load_huggingface_rows(
                dataset_name,
                args.split,
                args.revision,
                endpoint=args.hf_endpoint,
                download_timeout=args.download_timeout,
            )
            source_kind = "huggingface"
            source_revision = args.revision
            source_fingerprint = getattr(rows, "_fingerprint", None)
        workflows, references = prepare_rows(
            rows,
            dataset_name=dataset_name,
            dataset_split=args.split,
            limit=args.limit,
            dataset_revision=source_revision,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    args.out.mkdir(parents=True, exist_ok=True)
    workflows_path = args.out / "workflows.jsonl"
    references_path = args.out / "human_reference_patches.jsonl"
    report_path = args.out / "preparation_report.json"
    write_jsonl(workflows_path, workflows)
    write_jsonl(references_path, references)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "share_swebench_staging_preparation_report",
        "source_kind": source_kind,
        "dataset": dataset_name,
        "dataset_split": args.split,
        "source_revision": source_revision,
        "source_fingerprint": source_fingerprint,
        "hub_endpoint": None if source_kind == "local_export" else (args.hf_endpoint or "https://huggingface.co"),
        "selected_tasks": len(workflows),
        "repository_groups": len({row["group_id"] for row in workflows}),
        "tasks_by_repository": dict(sorted(Counter(row["group_id"] for row in workflows).items())),
        "workflows_file": workflows_path.name,
        "workflows_sha256": sha256_file(workflows_path),
        "human_references_file": references_path.name,
        "human_references_sha256": sha256_file(references_path),
        "gold_patch_leakage_check": all("patch" not in row["context"] and "patch" not in row["metadata"] for row in workflows),
        "evidence_status": "staging_only_not_paired_outcomes",
        "evidence_warning": (
            "Historical merged patches are human-repair proxies. These files are not RepairEpisode records and do not "
            "contain executed AI, Human, or Stop outcomes. Human time is unavailable. An immutable --revision should "
            "be supplied for a release dataset."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "selected_tasks": len(workflows), "repository_groups": report["repository_groups"]}, indent=2))


if __name__ == "__main__":
    main()
