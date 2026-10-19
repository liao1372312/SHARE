from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def item(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"missing correction input: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a correction after a verified pre-holdout Plan-A preflight failure.")
    parser.add_argument("--failed-dir", type=Path, default=ROOT / "results/share_plan_a_locked_evaluation_20260826")
    parser.add_argument("--out", type=Path, default=ROOT / "results/share_plan_a_locked_evaluation_final_20260826")
    args = parser.parse_args()
    failed = args.failed_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "UNBLINDING_ATTEMPT.json").exists():
        raise SystemExit("corrected Plan-A evaluation has already been attempted")

    attempt_path = failed / "UNBLINDING_ATTEMPT.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    forbidden_results = [
        "raw_results.json", "primary_20pct_results.json", "aggregate_budget_results.json",
        "primary_paired_comparisons.json", "voh_validity_analysis.json", "evaluation_manifest.json",
    ]
    present = [name for name in forbidden_results if (failed / name).exists()]
    if present:
        raise SystemExit(f"cannot classify as pre-unblinding failure; result files exist: {present}")
    expected_error = "historical identifier index mismatch for development_sha256"
    development = ROOT / "datasets/share_guard_development/episodes.jsonl"
    historical = json.loads((ROOT / "datasets/share_guard_development/historical_identifiers.json").read_text(encoding="utf-8"))
    if sha256_file(development) != historical["development_sha256"]:
        raise SystemExit("corrected development input still does not match historical index")
    runner = ROOT / "scripts/run_share_plan_a_locked_holdout_once.py"
    runner_lines = runner.read_text(encoding="utf-8").splitlines()
    development_line = next(index for index, line in enumerate(runner_lines, 1) if "development = load_episodes" in line)
    holdout_line = next(index for index, line in enumerate(runner_lines, 1) if "holdout = load_episodes" in line)
    if development_line >= holdout_line:
        raise SystemExit("source order does not prove failure occurred before holdout loading")

    failure_audit = {
        "schema_version": 1,
        "kind": "share_plan_a_pre_unblinding_failure_audit",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED_PREFLIGHT_BEFORE_HOLDOUT_READ",
        "failed_process_ledger": item(attempt_path),
        "observed_exception": expected_error,
        "cause": "the command supplied datasets/share_synthetic/episodes.jsonl, while the sealed historical identifier index validates against datasets/share_guard_development/episodes.jsonl",
        "source_order_evidence": {
            "runner": item(runner),
            "development_validation_line": development_line,
            "holdout_load_line": holdout_line,
            "failure_preceded_holdout_load": True,
        },
        "result_files_present": present,
        "method_level_locked_results_accessed": False,
        "locked_episode_payload_parsed": False,
        "policies_executed": False,
        "statistical_analysis_executed": False,
        "ledger_preserved_unmodified": True,
        "correction_rule": "the failed preflight is not an evaluation or unblinding; authorize one and only one holdout read with the corrected, hash-bound development input",
    }
    failure_path = out / "pre_unblinding_failure_audit.json"
    atomic_json(failure_path, failure_audit)

    original_amendment = failed / "protocol_amendment.json"
    original_seal = failed / "combined_pre_evaluation_seal.json"
    updated_manifest = failed / "updated_locked_manifest.json"
    artifact_lock = ROOT / "results/share_locked_evaluation_prereg_20260826/artifact_lock.json"
    holdout = ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl"
    amendment = {
        "schema_version": 2,
        "kind": "share_plan_a_locked_evaluation_preregistration_preflight_correction",
        "protocol_id": "share-plan-a-replayable-locked-eval-v1-correction-1",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEALED_PRE_EVALUATION",
        "reason": "Correct a development-provenance input mismatch detected before holdout parsing; no method-level locked outcome was accessed.",
        "method_and_analysis_changes": "none",
        "corrected_input": item(development),
        "historical_identifier_index": item(ROOT / "datasets/share_guard_development/historical_identifiers.json"),
        "original_amendment": item(original_amendment),
        "original_pre_evaluation_seal": item(original_seal),
        "pre_unblinding_failure_audit": item(failure_path),
        "unchanged_data_contract": "real-source replayable synthetic; all 400 episodes synthetic=true; no human-participant execution claim",
        "unchanged_frozen_plan": json.loads(original_amendment.read_text(encoding="utf-8"))["unchanged_frozen_plan"],
        "method_level_locked_results_accessed": False,
        "locked_evaluator_run": False,
        "authorized_actual_holdout_openings_remaining": 1,
        "prohibited_actions": ["delete or rewrite the failed ledger", "rerun after the corrected holdout opening", "modify any frozen method or analysis setting"],
    }
    amendment_path = out / "protocol_amendment.json"
    atomic_json(amendment_path, amendment)

    bindings = {
        "protocol_amendment": item(amendment_path),
        "updated_locked_manifest": item(updated_manifest),
        "artifact_lock": item(artifact_lock),
        "episodes": item(holdout),
        "plan_a_one_shot_evaluator": item(runner),
        "corrected_development": item(development),
        "historical_identifier_index": item(ROOT / "datasets/share_guard_development/historical_identifiers.json"),
        "pre_unblinding_failure_audit": item(failure_path),
        "original_amendment": item(original_amendment),
        "original_pre_evaluation_seal": item(original_seal),
        "original_failed_ledger": item(attempt_path),
    }
    seal = {
        "schema_version": 2,
        "kind": "share_plan_a_combined_pre_evaluation_seal_after_preflight_correction",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEALED_READY_FOR_EXACTLY_ONE_LOCKED_EVALUATION",
        "bindings": bindings,
        "combined_binding_sha256": sha256_json({name: value["sha256"] for name, value in bindings.items()}),
        "failed_preflight_preserved": True,
        "actual_holdout_openings_so_far": 0,
        "method_level_locked_results_accessed": False,
        "locked_evaluator_run": False,
    }
    seal_path = out / "combined_pre_evaluation_seal.json"
    atomic_json(seal_path, seal)
    checksum_paths = [failure_path, amendment_path, seal_path]
    (out / "PRE_EVALUATION_SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths), encoding="utf-8"
    )
    print(json.dumps({
        "out": str(out),
        "status": seal["status"],
        "failed_preflight_preserved": True,
        "locked_outcomes_accessed": False,
        "actual_holdout_openings_remaining": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
