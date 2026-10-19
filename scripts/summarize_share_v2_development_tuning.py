from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "share_v2_tuning_summary_20260827"
V1 = ROOT / "results" / "share_final_development_ablation_20pct_20260826" / "aggregate.json"
PAIRED = ROOT / "results" / "share_v2_development_multistart_20260827" / "paired_e30" / "summary.json"
TUNING = ROOT / "results" / "share_v2_stop_penalty_tuning_20260827"
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
COMPARISON_METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def aggregate(rows: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[section][metric]) for row in rows]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    if section == "development_test":
        output["all_runs_respected_cap"] = all(
            bool(row[section]["cap_respected"]) for row in rows
        )
    return output


def concise(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"variant": name, "seeds": int(row["seeds"])}
    for metric in COMPARISON_METRICS:
        output[f"{metric}_mean"] = float(row[f"{metric}_mean"])
        output[f"{metric}_std"] = float(row[f"{metric}_std"])
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_rows = json.loads(V1.read_text(encoding="utf-8"))
    v1 = next(row for row in v1_rows if row["policy"] == "share")
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))["development_test_aggregate"][0]

    candidate_summaries = []
    for name in ("p040", "p060", "p080"):
        payload = json.loads((TUNING / name / "summary.json").read_text(encoding="utf-8"))
        candidate_summaries.append(
            {
                "candidate": name,
                "rollback_penalty": float(payload["rollback_penalty"]),
                "validation": payload["validation_aggregate"],
                "development_test_diagnostic": payload["development_test_aggregate"],
            }
        )

    # Penalty 0.60 was selected using seed-17 validation only: all candidates
    # passed HIR/UAR/Stop gates; maximize Repair.SR, then common-reward Return.
    seed_paths = [TUNING / "p060" / "seed_17.json"] + [
        TUNING / "p060_remaining" / f"seed_{seed}.json"
        for seed in (29, 43, 71, 101)
    ]
    v2_rows = [json.loads(path.read_text(encoding="utf-8")) for path in seed_paths]
    v2_validation = aggregate(v2_rows, "validation")
    v2_development = aggregate(v2_rows, "development_test")

    comparison = [
        concise("V1-standard-replay", v1),
        concise("V2-paired-replay", paired),
        concise("V2-paired-replay-stop0.60", v2_development),
    ]
    v2_minus_v1 = {
        metric: float(v2_development[f"{metric}_mean"]) - float(v1[f"{metric}_mean"])
        for metric in COMPARISON_METRICS
    }
    selected_checkpoints = {
        str(row["seed"]): {
            "path": str(path.relative_to(ROOT)),
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "result_sha256": sha256_file(path),
        }
        for path, row in zip(seed_paths, v2_rows)
    }

    summary = {
        "status": "DEVELOPMENT_CANDIDATE_ONLY",
        "method": "SHARE-V2 paired-replay Double DQN with training rollback_penalty=0.60",
        "locked_holdout_accessed_by_this_tuning_pipeline": False,
        "eligible_to_replace_current_locked_manuscript_result": False,
        "new_untouched_holdout_required": True,
        "selection": {
            "data": "validation only",
            "pilot_seed": 17,
            "candidate_penalties": [0.40, 0.60, 0.80],
            "gate": "HIR<=0.20, UAR<=0.12, Stop<=0.05",
            "ranking": "maximize Repair.SR, then common-original-reward Return",
            "selected_rollback_penalty": 0.60,
            "development_test_used_for_selection": False,
        },
        "candidate_pilot_summaries": candidate_summaries,
        "selected_validation_five_seed": v2_validation,
        "selected_development_test_five_seed": v2_development,
        "comparison": comparison,
        "v2_minus_v1": v2_minus_v1,
        "selected_checkpoints": selected_checkpoints,
        "source_sha256": {
            "scripts/run_share_v2_development_multistart.py": sha256_file(ROOT / "scripts" / "run_share_v2_development_multistart.py"),
            "scripts/run_share_v2_stop_penalty_tuning.py": sha256_file(ROOT / "scripts" / "run_share_v2_stop_penalty_tuning.py"),
            "scripts/summarize_share_v2_development_tuning.py": sha256_file(Path(__file__)),
            "share/dqn.py": sha256_file(ROOT / "share" / "dqn.py"),
        },
    }
    atomic_json(OUT / "summary.json", summary)

    fields = list(comparison[0].keys())
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)

    per_seed_fields = ["seed"]
    for section in ("validation", "development_test"):
        per_seed_fields.extend(f"{section}_{metric}" for metric in METRICS)
    per_seed = []
    for row in v2_rows:
        item: dict[str, Any] = {"seed": row["seed"]}
        for section in ("validation", "development_test"):
            for metric in METRICS:
                item[f"{section}_{metric}"] = row[section][metric]
        per_seed.append(item)
    with (OUT / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(per_seed)

    def f(value: float) -> str:
        return f"{value:.3f}"

    report = f"""# SHARE-V2 Development Tuning Report

Status: **development candidate only; new untouched holdout required**

## Selected configuration

- Controller: paired-replay Double DQN
- Training rollback penalty: 0.60
- Human budget: 0.20
- Seeds: 17, 29, 43, 71, 101
- Selection: seed-17 validation gate/ranking only; development-test was not used to choose the penalty
- Evaluation reward: original frozen RewardConfig for all candidates

## Five-seed development diagnostic

| Variant | Repair.SR | HIR | UAR | Stop | Return | Regret |
|---|---:|---:|---:|---:|---:|---:|
| V1 | {f(v1['repair_sr_mean'])} | {f(v1['hir_mean'])} | {f(v1['uar_mean'])} | {f(v1['stopped_rate_mean'])} | {f(v1['normalized_return_mean'])} | {f(v1['collaboration_regret_mean'])} |
| Paired replay | {f(paired['repair_sr_mean'])} | {f(paired['hir_mean'])} | {f(paired['uar_mean'])} | {f(paired['stopped_rate_mean'])} | {f(paired['normalized_return_mean'])} | {f(paired['collaboration_regret_mean'])} |
| Paired + Stop penalty 0.60 | {f(v2_development['repair_sr_mean'])} | {f(v2_development['hir_mean'])} | {f(v2_development['uar_mean'])} | {f(v2_development['stopped_rate_mean'])} | {f(v2_development['normalized_return_mean'])} | {f(v2_development['collaboration_regret_mean'])} |

V2 minus V1: Repair.SR {v2_minus_v1['repair_sr']:+.3f}, HIR {v2_minus_v1['hir']:+.3f}, UAR {v2_minus_v1['uar']:+.3f}, Stop {v2_minus_v1['stopped_rate']:+.3f}, Return {v2_minus_v1['normalized_return']:+.3f}, Regret {v2_minus_v1['collaboration_regret']:+.3f}.

## Evidence boundary

This tuning started after the previous locked result was known. It used only the allowlisted synthetic train/validation/development dataset and did not load the 400-episode locked holdout. The resulting candidate **cannot replace the current locked manuscript result**. A newly generated, workflow-group-disjoint, untouched holdout must be sealed before confirmatory evaluation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    checksum_paths = [OUT / "summary.json", OUT / "comparison.csv", OUT / "per_seed.csv", OUT / "REPORT.md"]
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
