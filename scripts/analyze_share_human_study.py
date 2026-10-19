from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median


REQUIRED = {
    "participant_id", "task_id", "condition", "task_complexity", "success", "completion_seconds",
    "human_actions", "recommendation_outcome", "workload", "trust", "excluded",
}
CONDITIONS = {"human_only", "always_ai_assisted", "selective_collaboration"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize real SHARE practitioner-study logs without fabricating missing observations.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/share_human_study/summary.json"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate(rows)
    included = [row for row in rows if not parse_bool(row["excluded"])]
    summary = {"participants": len({row["participant_id"] for row in included}), "tasks": len({row["task_id"] for row in included}), "conditions": {}}
    for condition in sorted(CONDITIONS):
        subset = [row for row in included if row["condition"] == condition]
        summary["conditions"][condition] = {
            "n": len(subset),
            "repair_sr": mean(parse_bool(row["success"]) for row in subset) if subset else None,
            "completion_seconds_mean": mean(float(row["completion_seconds"]) for row in subset) if subset else None,
            "completion_seconds_median": median(float(row["completion_seconds"]) for row in subset) if subset else None,
            "human_actions_mean": mean(float(row["human_actions"]) for row in subset) if subset else None,
            "workload_mean": mean(float(row["workload"]) for row in subset) if subset else None,
            "trust_mean": mean(float(row["trust"]) for row in subset) if subset else None,
            "recommendation_outcomes": dict(Counter(row["recommendation_outcome"] for row in subset)),
        }
    summary["paired_permutation"] = paired_permutation_analysis(included)
    summary["mixed_effects_models"] = "preregistered_external_analysis_required_after_sample-size_and_diagnostic_checks"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def validate(rows):
    if not rows:
        raise ValueError("human study log is empty")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    seen = set()
    participant_conditions = {}
    for index, row in enumerate(rows, 2):
        key = (row["participant_id"], row["task_id"])
        if key in seen:
            raise ValueError(f"duplicate participant/task at row {index}")
        seen.add(key)
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"invalid condition at row {index}")
        if float(row["completion_seconds"]) <= 0:
            raise ValueError(f"non-positive completion time at row {index}")
        participant_conditions.setdefault(row["participant_id"], set()).add(row["condition"])
    incomplete = [participant for participant, values in participant_conditions.items() if values != CONDITIONS]
    if incomplete:
        raise ValueError(f"within-subject participants lack conditions: {incomplete[:5]}")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalized in {"1", "true", "yes"}


def paired_permutation_analysis(rows, permutations: int = 10_000, seed: int = 17):
    contrasts = [("selective_collaboration", "human_only"), ("selective_collaboration", "always_ai_assisted")]
    metrics = {
        "success": lambda row: float(parse_bool(row["success"])),
        "completion_seconds": lambda row: float(row["completion_seconds"]),
        "workload": lambda row: float(row["workload"]),
        "trust": lambda row: float(row["trust"]),
    }
    by_participant = {}
    for row in rows:
        by_participant.setdefault(row["participant_id"], {}).setdefault(row["condition"], []).append(row)
    results = []
    rng = random.Random(seed)
    for left, right in contrasts:
        for metric, getter in metrics.items():
            differences = []
            for conditions in by_participant.values():
                if left in conditions and right in conditions:
                    differences.append(mean(getter(row) for row in conditions[left]) - mean(getter(row) for row in conditions[right]))
            observed = mean(differences) if differences else None
            if differences:
                exceed = 0
                for _ in range(permutations):
                    permuted = mean(value * (-1 if rng.random() < 0.5 else 1) for value in differences)
                    exceed += abs(permuted) >= abs(observed)
                p_value = (exceed + 1) / (permutations + 1)
                std = (sum((value - observed) ** 2 for value in differences) / max(1, len(differences) - 1)) ** 0.5
                effect = observed / std if std > 0 else None
            else:
                p_value = effect = None
            results.append({"contrast": f"{left}_minus_{right}", "metric": metric, "participants": len(differences), "mean_difference": observed, "paired_effect_size": effect, "p_value": p_value})
    valid = sorted((item["p_value"], index) for index, item in enumerate(results) if item["p_value"] is not None)
    adjusted = [None] * len(results)
    running = 0.0
    count = len(valid)
    for rank, (value, index) in enumerate(valid):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[index] = running
    for item, value in zip(results, adjusted):
        item["holm_adjusted_p"] = value
    return results


if __name__ == "__main__":
    main()
