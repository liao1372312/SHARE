from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


MAX_VIOLATION_RATIO = 1e308


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def failure_mode(candidate: Mapping, human_target: float, unsafe_target: float, max_stopped_rate: float | None = None) -> str:
    metrics = candidate["fold_metrics"]
    human_failure = any(float(row["hir"]) > human_target + 1e-12 for row in metrics)
    unsafe_failure = any(float(row["unsafe_autonomy_cost_rate"]) > unsafe_target + 1e-12 for row in metrics)
    stop_failure = max_stopped_rate is not None and any(float(row.get("stopped_rate", 0.0)) > max_stopped_rate + 1e-12 for row in metrics)
    failures = [name for name, failed in (("human", human_failure), ("unsafe", unsafe_failure), ("stop", stop_failure)) if failed]
    if not failures:
        return "passed"
    return f"{failures[0]}_only" if len(failures) == 1 else "_and_".join(failures)


def constraint_ratio(value: float, target: float) -> float:
    if target > 0.0:
        return value / target
    return 0.0 if value <= 0.0 else MAX_VIOLATION_RATIO


def candidate_summary(candidate: Mapping, human_target: float, unsafe_target: float, max_stopped_rate: float | None = None) -> dict:
    fold_metrics = candidate["fold_metrics"]
    max_hir = max(float(row["hir"]) for row in fold_metrics)
    max_unsafe = max(float(row["unsafe_autonomy_cost_rate"]) for row in fold_metrics)
    max_stopped = max(float(row.get("stopped_rate", 0.0)) for row in fold_metrics)
    ratios = [constraint_ratio(max_hir, human_target), constraint_ratio(max_unsafe, unsafe_target)]
    if max_stopped_rate is not None:
        ratios.append(constraint_ratio(max_stopped, max_stopped_rate))
    return {
        "candidate": candidate["candidate"],
        "stable": bool(candidate.get("stable", False)),
        "failure_mode": failure_mode(candidate, human_target, unsafe_target, max_stopped_rate),
        "max_fold_hir": max_hir,
        "max_fold_unsafe_autonomy_cost_rate": max_unsafe,
        "max_fold_stopped_rate": max_stopped,
        "max_relative_constraint_ratio": max(ratios),
        "mean_metrics": candidate["mean_metrics"],
    }


def summarize_runs(result: Mapping) -> dict:
    runs: Sequence[Mapping] = result.get("runs", [])
    if not runs:
        raise ValueError("result has no completed SHARE-Guard runs")
    config = runs[0]["selection"]["config"]
    human_target = float(config["human_operating_target"])
    unsafe_target = float(config["unsafe_operating_target"])
    max_stopped_rate = float(config["max_stopped_rate"]) if config.get("max_stopped_rate") is not None else None
    method_rows: dict[str, list[dict]] = defaultdict(list)
    failure_modes: Counter[str] = Counter()
    concise_runs = []
    for run in sorted(runs, key=lambda item: (int(item["seed"]), str(item["method"]))):
        selection = run["selection"]
        if selection["config"] != config:
            raise ValueError("runs use inconsistent selection configurations")
        candidates = [candidate_summary(candidate, human_target, unsafe_target, max_stopped_rate) for candidate in selection["candidates"]]
        for candidate in candidates:
            failure_modes[candidate["failure_mode"]] += 1
        least_violation = min(
            candidates,
            key=lambda candidate: (
                candidate["max_relative_constraint_ratio"],
                candidate["max_fold_unsafe_autonomy_cost_rate"],
                candidate["max_fold_hir"],
                candidate["candidate"],
            ),
        )
        row = {
            "method": run["method"],
            "seed": int(run["seed"]),
            "status": run["status"],
            "selected_candidate": run["selected_candidate"],
            "candidate_failure_modes": dict(sorted(Counter(candidate["failure_mode"] for candidate in candidates).items())),
            "least_violation_candidate": least_violation,
            "candidates": candidates,
        }
        concise_runs.append(row)
        method_rows[str(run["method"])].append(row)
    method_summaries = {
        method: {
            "runs": len(rows),
            "selected_runs": sum(row["selected_candidate"] is not None for row in rows),
            "candidate_failure_modes": dict(sorted(Counter(
                mode for row in rows for mode, count in row["candidate_failure_modes"].items() for _ in range(count)
            ).items())),
            "least_violation_candidates": [row["least_violation_candidate"] for row in rows],
        }
        for method, rows in sorted(method_rows.items())
    }
    return {
        "schema_version": 1,
        "evidence_warning": result.get("evidence_warning"),
        "screen_status": "no_stable_candidate" if not any(row["selected_candidate"] is not None for row in concise_runs) else "selection_completed",
        "completed_runs": len(concise_runs),
        "selected_runs": sum(row["selected_candidate"] is not None for row in concise_runs),
        "selection_config": config,
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "method_summaries": method_summaries,
        "runs": concise_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SHARE-Guard selection-screen outcomes without evaluating new episodes.")
    parser.add_argument("--results", type=Path, required=True, help="Completed SHARE-Guard results.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON summary.")
    args = parser.parse_args()
    if not args.results.is_file():
        raise SystemExit("--results must be an existing results.json file")
    source = json.loads(args.results.read_text(encoding="utf-8"))
    summary = summarize_runs(source)
    summary["source_results"] = str(args.results.resolve())
    summary["source_results_sha256"] = sha256_file(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "completed_runs": summary["completed_runs"], "selected_runs": summary["selected_runs"]}, indent=2))


if __name__ == "__main__":
    main()
