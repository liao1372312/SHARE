from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

REQUIRED_TOP = {"id", "dataset", "synthetic", "task", "api_pool_size", "candidate_api_ids", "events", "repair_label", "metadata"}
REQUIRED_LABEL = {"failed", "faulty_agent", "failure_type", "faulty_decision_node", "expected_repair_action", "affected_slice", "annotation_source"}
VALID_FAILURES = {None, "planning_failure", "service_selection_failure", "parameter_binding_failure", "tool_execution_failure", "supervisor_failure", "answer_grounding_failure"}
VALID_ACTIONS = {None, "AddSubtask", "Replan", "ReselectAPI", "RebindParam", "RetryTool", "ReplaceTool", "ReviseRepair", "ReviseAnswer", "MarkUncertain", "Terminate"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated FLARE synthetic datasets.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found under {args.root}")
    summaries = []
    for path in files:
        summaries.append(validate_file(path))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def validate_file(path: Path) -> Dict[str, Any]:
    ids = set()
    failures = 0
    failure_types = Counter()
    actions = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            missing = REQUIRED_TOP - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no} missing top-level keys {missing}")
            if item["id"] in ids:
                raise ValueError(f"{path}:{line_no} duplicate id {item['id']}")
            ids.add(item["id"])
            if item["synthetic"] is not True:
                raise ValueError(f"{path}:{line_no} synthetic must be true")
            if not item["events"]:
                raise ValueError(f"{path}:{line_no} events must be non-empty")
            label = item["repair_label"]
            missing_label = REQUIRED_LABEL - set(label)
            if missing_label:
                raise ValueError(f"{path}:{line_no} missing repair_label keys {missing_label}")
            if label["failure_type"] not in VALID_FAILURES:
                raise ValueError(f"{path}:{line_no} invalid failure_type {label['failure_type']}")
            if label["expected_repair_action"] not in VALID_ACTIONS:
                raise ValueError(f"{path}:{line_no} invalid action {label['expected_repair_action']}")
            event_ids = {event["id"] for event in item["events"]}
            if label["failed"]:
                failures += 1
                failure_types[label["failure_type"]] += 1
                actions[label["expected_repair_action"]] += 1
                if label["faulty_decision_node"] not in event_ids:
                    raise ValueError(f"{path}:{line_no} faulty node not in events")
                if not set(label["affected_slice"]).issubset(event_ids):
                    raise ValueError(f"{path}:{line_no} affected_slice contains unknown ids")
            else:
                if any(label[key] is not None for key in ["faulty_agent", "failure_type", "faulty_decision_node", "expected_repair_action"]):
                    raise ValueError(f"{path}:{line_no} non-failed sample has failure labels")
    return {
        "path": str(path),
        "rows": rows,
        "failures": failures,
        "failure_type_counts": dict(failure_types),
        "repair_action_counts": dict(actions),
    }


if __name__ == "__main__":
    main()
