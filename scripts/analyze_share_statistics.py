from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.statistics import METRICS, hierarchical_bootstrap_ci, holm_adjust, paired_differences, paired_effect_size, paired_permutation_test


def load_result(runs: Path, policy: str, budget: float, seed: int) -> dict:
    path = runs / f"{policy}_budget_{budget:g}_seed_{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired hierarchical statistics for SHARE policy comparisons.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--candidate", default="share_controller")
    parser.add_argument("--reference", default="unconstrained_double_dqn")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("results/share_statistics.json"))
    args = parser.parse_args()
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise SystemExit(f"unknown metrics: {unknown}")
    seeds = [int(value) for value in args.seeds.split(",")]
    test = [episode for episode in load_episodes(args.episodes) if episode.split == "test"]
    groups = {episode.episode_id: episode.group_id for episode in test}
    rows = []
    for seed in seeds:
        candidate = load_result(args.runs, args.candidate, args.budget, seed)
        reference = load_result(args.runs, args.reference, args.budget, seed)
        for metric in metrics:
            differences = paired_differences(candidate["records"], reference["records"], metric)
            low, high = hierarchical_bootstrap_ci(differences, groups, args.resamples, seed)
            rows.append({
                "candidate": args.candidate,
                "reference": args.reference,
                "metric": metric,
                "seed": seed,
                "episodes": len(differences),
                "mean_difference": float(np.mean(list(differences.values()))),
                "ci95_low": low,
                "ci95_high": high,
                "paired_effect_size": paired_effect_size(differences),
                "p_value": paired_permutation_test(differences, args.resamples, seed),
            })
    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
    output = {
        "evidence_warning": "Synthetic results are engineering evidence only." if all(episode.synthetic for episode in test) else None,
        "difference_direction": "candidate_minus_reference",
        "bootstrap": "workflow-group hierarchical bootstrap, then episodes within sampled groups",
        "permutation": "two-sided paired sign-flip test",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "comparisons": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
