"""Reconstruct the frozen 232-group component table from per-workflow records."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/share_component_confirmation_e2_232_20260930"
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def main() -> None:
    result = json.loads((OUT / "FINAL_RESULTS.json").read_text())
    groups = defaultdict(list)
    with (OUT / "per_workflow_records.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[(row["method"], row["members"], row["seed"])].append(row)
    if len(groups) != 30:
        raise AssertionError(f"expected 30 policies, got {len(groups)}")
    checks = []
    for (method, members, seed), rows in groups.items():
        if len(rows) != 232 or len({row["episode_id"] for row in rows}) != 232:
            raise AssertionError(f"nonunique or incomplete workflow rows: {method}/{members}/{seed}")
        human = [row for row in rows if row["human_used"]]
        stop = [row for row in rows if row["actions"][-1] == 2]
        autonomous = [row for row in rows if not row["human_used"] and row["actions"][-1] != 2]
        if len(human) > 46 or len(autonomous) == 0:
            raise AssertionError(f"quota or UAR denominator invalid: {method}/{members}/{seed}")
        selected = [v for row in rows for v in row["selected_returns"]]
        oracle = [v for row in rows for v in row["oracle_returns"]]
        actual = {
            "repair_sr": mean(bool(row["success"]) for row in rows),
            "hir": len(human) / len(rows),
            "uar": mean(bool(row["unsafe"]) or not bool(row["success"]) for row in autonomous),
            "stopped_rate": len(stop) / len(rows),
            "normalized_return": mean(sum(row["rewards"]) for row in rows),
            "collaboration_regret": mean(best - chosen for chosen, best in zip(selected, oracle)),
        }
        expected = result["five_member_ensembles"][method] if members == 5 else next(
            row for row in result["single_model_rows"][method] if row["seed"] == seed
        )
        for key in METRICS:
            if abs(actual[key] - expected[key]) > 1e-12:
                raise AssertionError(f"{method}/{members}/{seed}: {key} {actual[key]} != {expected[key]}")
        checks.append({
            "method": method, "members": members, "seed": seed,
            "workflows": len(rows), "success": sum(bool(row["success"]) for row in rows),
            "human": len(human), "stop": len(stop), "autonomous_completed": len(autonomous),
            "autonomous_adverse": sum(bool(row["unsafe"]) or not bool(row["success"]) for row in autonomous),
            "autonomous_success_and_unsafe": sum(bool(row["success"]) and bool(row["unsafe"]) for row in autonomous),
            "metric_reconstruction_pass": True,
        })
    audit = {"result": "PASS", "policies": len(groups), "workflow_records": sum(map(len, groups.values())), "checks": checks}
    (OUT / "RESULT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"result": audit["result"], "policies": audit["policies"], "workflow_records": audit["workflow_records"], "ensembles": [row for row in checks if row["members"] == 5]}, indent=2))


if __name__ == "__main__":
    main()
