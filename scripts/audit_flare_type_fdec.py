"""Reconstruct the archived Type/F.Dec counts from per-example outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/flare_complexity_realistic/predictions.jsonl"
DATA = ROOT / "datasets/flare_complexity"
FAILURE_MAP = {
    "planning_failure": "missing_dependency",
    "service_selection_failure": "api_error",
    "parameter_binding_failure": "parameter_error",
    "tool_execution_failure": "execution_error",
    "supervisor_failure": "unrecovered",
    "answer_grounding_failure": "unsupported_claim",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    predictions = {}
    for line in PREDICTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        if row["method"] == "flare":
            key = (row["dataset"], row["sample_id"])
            if key in predictions:
                raise ValueError(f"duplicate prediction: {key}")
            predictions[key] = row
    result = {}
    for split in ("simple", "complex"):
        counts = Counter()
        mismatches = Counter()
        seen = set()
        for line in (DATA / f"{split}.jsonl").open(encoding="utf-8"):
            case = json.loads(line)
            if not case["repair_label"]["failed"]:
                continue
            key = (split, case["id"])
            prediction = predictions[key]
            seen.add(key)
            gold = case["repair_label"]
            node_correct = bool(prediction["ranked_decisions"]) and (
                prediction["ranked_decisions"][0] == gold["faulty_decision_node"]
            )
            label_correct = prediction["pred_failure_type"] == FAILURE_MAP[gold["failure_type"]]
            counts[(label_correct, node_correct)] += 1
            if node_correct and not label_correct:
                mismatches[(gold["failure_type"], prediction["pred_failure_type"])] += 1
        if seen != {key for key in predictions if key[0] == split}:
            raise ValueError(f"prediction/case coverage mismatch for {split}")
        n = sum(counts.values())
        label_hits = sum(value for (label, _), value in counts.items() if label)
        node_hits = sum(value for (_, node), value in counts.items() if node)
        result[split] = {
            "n": n,
            "failure_label_correct": label_hits,
            "failure_label_accuracy": label_hits / n,
            "faulty_node_correct": node_hits,
            "faulty_node_accuracy": node_hits / n,
            "contingency": {
                f"label_{int(label)}_node_{int(node)}": counts[(label, node)]
                for label in (False, True) for node in (False, True)
            },
            "correct_node_wrong_label_breakdown": {
                f"{gold} -> {pred}": count for (gold, pred), count in sorted(mismatches.items())
            },
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
