from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("repair_sr", "hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate", "normalized_return", "collaboration_regret", "human_budget_violation", "unsafe_budget_violation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed SHARE results.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/share_aggregate/summary.json"))
    args = parser.parse_args()
    groups = defaultdict(list)
    run_count = 0
    for path in args.inputs:
        source = path / "summary.json" if path.is_dir() else path
        for row in json.loads(source.read_text(encoding="utf-8")):
            groups[(row["policy"], float(row["target_budget"]))].append(row)
            run_count += 1
    summary = []
    for key, items in sorted(groups.items()):
        policy, budget = key
        aggregate = {"policy": policy, "target_budget": budget, "seeds": len(items)}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items])
            aggregate[metric + "_mean"] = float(values.mean())
            aggregate[metric + "_p2_5"] = float(np.percentile(values, 2.5))
            aggregate[metric + "_p97_5"] = float(np.percentile(values, 97.5))
        passes = [item["human_budget_violation"] <= 1e-12 and item["unsafe_budget_violation"] <= 1e-12 for item in items]
        aggregate["joint_constraint_pass_rate"] = float(np.mean(passes))
        aggregate["synthetic_evidence"] = all(item.get("synthetic_evidence", False) for item in items)
        summary.append(aggregate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "groups": len(summary), "runs": run_count}, indent=2))


if __name__ == "__main__":
    main()
