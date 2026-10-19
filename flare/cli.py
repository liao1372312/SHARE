from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .pipeline import FLARE, FLAREConfig


def load_events(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "events" in data:
        return data["events"]
    if isinstance(data, list):
        return data
    raise ValueError("Input must be a JSON list, JSON object with 'events', or JSONL events")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLARE fault localization and minimal repair on execution events.")
    parser.add_argument("trace", type=Path, help="JSON/JSONL structured execution events")
    parser.add_argument("--budget", type=int, default=3, help="bounded repair budget B")
    args = parser.parse_args()
    result = FLARE(FLAREConfig(repair_budget=args.budget)).run(load_events(args.trace))
    print(json.dumps(_result_to_json(result), ensure_ascii=False, indent=2))


def _result_to_json(result):
    return {
        "initial_failures": sorted(result.initial_failures),
        "final_failures": sorted(result.final_failures),
        "repaired": result.repaired,
        "iterations": [
            {
                "failures": sorted(item.failures),
                "slice_nodes": sorted(item.slice_nodes),
                "action": item.action.value,
                "localization": None
                if item.localization is None
                else {
                    "agent": item.localization.agent,
                    "node_id": item.localization.node_id,
                    "node_type": item.localization.node_type,
                    "failure_type": item.localization.failure_type,
                    "confidence": item.localization.confidence,
                    "affected": sorted(item.localization.affected),
                    "scores": item.localization.scores,
                },
            }
            for item in result.iterations
        ],
    }


if __name__ == "__main__":
    main()
