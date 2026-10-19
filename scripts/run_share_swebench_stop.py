from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a SWE-bench Stop baseline without applying a solution patch.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error("--dataset must be an existing local file")
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if not 60 <= args.timeout <= 7200:
        parser.error("--timeout must be between 60 and 7200 seconds")
    return args


def main() -> None:
    args = parse_args()
    try:
        from swebench.harness.constants import RUN_EVALUATION_LOG_DIR
        from swebench.harness.reporting import make_run_report
        from swebench.harness.run_evaluation import _docker_client, load_instances, run_instances, write_run_metadata
    except ImportError as error:
        raise SystemExit("run this script with the Python environment containing the official swebench package") from error

    model_name = "share-stop-no-solution-patch"
    log_dir = RUN_EVALUATION_LOG_DIR / args.run_id / model_name / args.instance
    binding_path = args.report_dir / f"{model_name}.{args.run_id}.binding.json"
    if log_dir.exists() or binding_path.exists():
        raise SystemExit("refusing to reuse a Stop run-id or overwrite an existing binding")
    instances = load_instances(str(args.dataset), "test", [args.instance], None)
    if len(instances) != 1 or instances[0].get("instance_id") != args.instance:
        raise SystemExit("local dataset did not resolve exactly the requested instance")

    prediction = {
        "instance_id": args.instance,
        "model_name_or_path": model_name,
        # The harness reporting layer requires a non-empty sentinel. run_instances
        # receives skip_patch=True, so these bytes are never copied or applied.
        "model_patch": "SHARE_STOP_SENTINEL_NOT_APPLIED",
    }
    predictions = {args.instance: prediction}
    write_run_metadata(args.run_id, str(args.dataset), "test", None)
    run_instances(
        predictions,
        instances,
        max_workers=1,
        run_id=args.run_id,
        timeout=args.timeout,
        skip_patch=True,
    )
    report_path = make_run_report(predictions, instances, args.run_id, _docker_client(), str(args.report_dir))
    detail_path = log_dir / "report.json"
    if not detail_path.is_file():
        raise SystemExit("Stop execution did not produce a detailed report; inspect the run log")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    result = detail.get(args.instance, {})
    if result.get("infra_failure") is True:
        raise SystemExit("Stop execution encountered an infrastructure failure")
    binding = {
        "schema_version": 1,
        "kind": "share_swebench_stop_run_binding",
        "instance_id": args.instance,
        "run_id": args.run_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "semantics": "base_commit_tests_without_solution_patch",
        "solution_patch_applied": False,
        "harness_skip_patch": True,
        "report_file": str(report_path),
        "report_sha256": sha256_file(report_path),
        "detail_report_file": str(detail_path),
        "detail_report_sha256": sha256_file(detail_path),
        "resolved": bool(result.get("resolved")),
        "evidence_status": "one_executed_stop_branch_not_paired_outcomes",
        "warning": "The non-empty prediction sentinel is required only by generic SWE-bench reporting and was not applied to the container.",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "binding": str(binding_path), "resolved": binding["resolved"]}, indent=2))


if __name__ == "__main__":
    main()
