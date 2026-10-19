from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.oracle import solve_constrained_oracle, solve_expected_root_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute synthetic SHARE oracle feasibility diagnostics.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_feasibility"))
    parser.add_argument("--human-budgets", default="0,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--max-stopped-rate", type=float, default=None)
    parser.add_argument("--expected-only", action="store_true", help="Skip exact trajectory DP and run the scalable root-action LP only.")
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    splits = {name: [episode for episode in episodes if episode.split == name] for name in ["train", "validation", "test"]}
    rows = []
    details = {}
    for split, items in splits.items():
        if not items:
            continue
        for budget in [float(value) for value in args.human_budgets.split(",")]:
            result = None if args.expected_only else solve_constrained_oracle(items, budget, args.unsafe_budget)
            expected = None
            expected_error = None
            try:
                expected = solve_expected_root_oracle(
                    items,
                    budget,
                    args.unsafe_budget,
                    max_stopped_rate=args.max_stopped_rate,
                )
            except RuntimeError as error:
                expected_error = str(error)
            row = {
                "split": split,
                "human_budget": budget,
                "unsafe_budget": args.unsafe_budget,
                "max_stopped_rate": args.max_stopped_rate,
                "episodes": len(items),
                "human_limit": result.human_limit if result else None,
                "unsafe_limit": result.unsafe_limit if result else None,
                "repair_sr": result.repair_sr if result else None,
                "hir": result.hir if result else None,
                "unsafe_cost_rate": result.unsafe_cost_rate if result else None,
                "uar": result.uar if result else None,
                "stopped_rate": result.stopped_rate if result else None,
                "mean_discounted_return": result.total_return / result.episodes if result else None,
                "feasible": (
                    result.hir <= budget + 1e-12
                    and result.unsafe_cost_rate <= args.unsafe_budget + 1e-12
                ) if result else None,
                "synthetic_evidence": all(episode.synthetic for episode in items),
                "expected_oracle_feasible": expected is not None,
                "expected_oracle_error": expected_error,
                "expected_oracle_success_rate": expected.expected_success_rate if expected else None,
                "expected_oracle_return": expected.expected_return if expected else None,
                "expected_oracle_hir": expected.hir if expected else None,
                "expected_oracle_unsafe_rate": expected.expected_unsafe_rate if expected else None,
                "expected_oracle_stopped_rate": expected.stopped_rate if expected else None,
            }
            rows.append(row)
            if result is not None:
                details[f"{split}:{budget:g}"] = [[action.name.lower() for action in selection.actions] for selection in result.selections]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "selections.json").write_text(json.dumps(details, separators=(",", ":")), encoding="utf-8")
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "manifest.json").write_text(json.dumps({
        "evidence_warning": "Oracle results over synthetic paired outcomes are engineering diagnostics, not practitioner evidence.",
        "algorithm": "exact multiple-choice two-constraint dynamic programming and sparse root-action LP",
        "expected_oracle_stop_constraint": args.max_stopped_rate,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
