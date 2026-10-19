from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Split FLARE domain benchmark into simple/complex trace-complexity datasets.")
    parser.add_argument("--data", type=Path, default=Path("datasets/flare_domain"))
    parser.add_argument("--out", type=Path, default=Path("datasets/flare_complexity"))
    parser.add_argument("--simple-max-steps", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    buckets: Dict[str, List[Dict[str, Any]]] = {"simple": [], "complex": []}
    for path in sorted(args.data.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            steps = count_steps(row)
            active_events = count_active_events(row)
            active_agents = count_active_agents(row)
            bucket = "simple" if steps <= args.simple_max_steps else "complex"
            row = dict(row)
            row["dataset"] = bucket
            row["complexity_benchmark"] = True
            row.setdefault("metadata", {})
            row["metadata"] = dict(row["metadata"])
            row["metadata"].update(
                {
                    "source_dataset": path.stem,
                    "complexity_split": bucket,
                    "trace_steps": steps,
                    "active_agents": active_agents,
                    "active_events": active_events,
                    "generator": "scripts/split_flare_complexity_datasets.py",
                }
            )
            buckets[bucket].append(row)

    summaries = []
    for name, rows in buckets.items():
        output_path = args.out / f"{name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                row = dict(row)
                row["id"] = f"{name}_complexity_{index:06d}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries.append(summarize(name, rows, output_path))
    (args.out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_schema(args.out / "SCHEMA.md", args.simple_max_steps)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def count_steps(row: Dict[str, Any]) -> int:
    return sum(1 for event in row["events"] if event["type"] == "plan")


def count_active_events(row: Dict[str, Any]) -> int:
    return sum(1 for event in row["events"] if event["agent"] != "system")


def count_active_agents(row: Dict[str, Any]) -> int:
    return len({event["agent"] for event in row["events"] if event["agent"] != "system"})


def summarize(name: str, rows: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    failures = [row for row in rows if row["repair_label"]["failed"]]
    return {
        "dataset": name,
        "path": str(path),
        "tasks": len(rows),
        "apis": max(row.get("api_pool_size", 0) for row in rows) if rows else 0,
        "failures": len(failures),
        "faulty_decisions": len(failures),
        "failure_type_counts": dict(Counter(row["repair_label"]["failure_type"] for row in failures)),
        "repair_action_counts": dict(Counter(row["repair_label"].get("expected_repair_action") for row in failures)),
        "avg_steps": round(sum(count_steps(row) for row in rows) / len(rows), 2) if rows else 0.0,
        "avg_agents": round(sum(count_active_agents(row) for row in rows) / len(rows), 2) if rows else 0.0,
        "avg_actions": round(sum(count_active_events(row) for row in rows) / len(rows), 2) if rows else 0.0,
        "avg_actions_per_agent": round(sum(count_active_events(row) / max(1, count_active_agents(row)) for row in rows) / len(rows), 2) if rows else 0.0,
    }


def write_schema(path: Path, simple_max_steps: int) -> None:
    path.write_text(
        f"""# FLARE Complexity Benchmark Schema\n\nThis benchmark is split from `datasets/flare_domain` by trace complexity.\n\n- `simple`: traces with at most {simple_max_steps} planned service-composition steps.\n- `complex`: traces with more than {simple_max_steps} planned service-composition steps.\n\nEach row remains compatible with `scripts/run_flare_experiments.py` and preserves the source domain in `metadata.source_dataset`.\n""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
