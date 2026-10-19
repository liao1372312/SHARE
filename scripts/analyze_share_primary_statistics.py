from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes


METRICS = ("repair_sr", "hir", "unsafe_autonomy_cost_rate", "normalized_return", "collaboration_regret")


def record_metrics(record: dict) -> dict[str, float]:
    selected = np.asarray(record["selected_returns"], dtype=float)
    oracle = np.asarray(record["oracle_returns"], dtype=float)
    return {
        "repair_sr": float(record["success"]),
        "hir": float(record["human_used"]),
        "unsafe_autonomy_cost_rate": float(record.get("unsafe_actions", int(record["unsafe"]))),
        "normalized_return": float(sum(record["rewards"])),
        "collaboration_regret": float(np.mean(oracle - selected)),
    }


def load_policy_rows(root: Path, policy: str, seeds: list[int], budget: float) -> dict[str, dict[str, float]]:
    values: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seed in seeds:
        path = root / f"{policy}_budget_{budget:g}_seed_{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            values[record["episode_id"]].append(record_metrics(record))
    output = {}
    for episode_id, rows in values.items():
        if len(rows) != len(seeds):
            raise ValueError(f"{policy}/{episode_id}: expected {len(seeds)} seeds, found {len(rows)}")
        output[episode_id] = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    return output


def paired_group_differences(left: dict, right: dict, episode_groups: dict[str, str], metric: str) -> np.ndarray:
    by_group: dict[str, list[float]] = defaultdict(list)
    if left.keys() != right.keys():
        raise ValueError("policies do not cover identical episodes")
    for episode_id in left:
        by_group[episode_groups[episode_id]].append(left[episode_id][metric] - right[episode_id][metric])
    return np.asarray([np.mean(values) for _, values in sorted(by_group.items())], dtype=float)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_p(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(permuted >= observed - 1e-15)) / (draws + 1))


def holm(rows: list[dict]) -> None:
    for metric in METRICS:
        items = sorted((row for row in rows if row["metric"] == metric), key=lambda row: row["p_raw"])
        running = 0.0
        count = len(items)
        for rank, row in enumerate(items):
            running = max(running, min(1.0, (count - rank) * row["p_raw"]))
            row["p_holm"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired group-level statistics for the synthetic primary-budget diagnostic.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=17)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    episodes = [episode for episode in load_episodes(args.episodes) if episode.split == "test"]
    episode_groups = {episode.episode_id: episode.group_id for episode in episodes}
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("episode_sha256") is None or not all(episode.synthetic for episode in episodes):
        raise SystemExit("expected a manifest-bound synthetic development dataset")
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    policies = sorted({row["policy"] for row in summary})
    primary = load_policy_rows(args.results, "share_controller", seeds, args.budget)
    rows = []
    rng = np.random.default_rng(args.random_seed)
    for comparator in policies:
        if comparator == "share_controller":
            continue
        other = load_policy_rows(args.results, comparator, seeds, args.budget)
        for metric in METRICS:
            diffs = paired_group_differences(primary, other, episode_groups, metric)
            lo, hi = bootstrap_ci(diffs, rng, args.draws)
            rows.append({
                "comparison": f"share_controller-minus-{comparator}",
                "comparator": comparator,
                "metric": metric,
                "difference": float(diffs.mean()),
                "ci95_group_bootstrap": [lo, hi],
                "p_raw": permutation_p(diffs, rng, args.draws),
                "groups": int(len(diffs)),
            })
    holm(rows)
    payload = {
        "evidence_warning": "Synthetic development-test comparison averaged over five seeds; not independent locked-holdout or practitioner evidence.",
        "estimand": "SHARE minus comparator, averaged over seeds and paired by workflow group",
        "bootstrap": {"unit": "workflow_group", "draws": args.draws, "seed": args.random_seed},
        "permutation": {"unit": "workflow_group", "draws": args.draws, "two_sided": True, "holm_family": "comparators within metric"},
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "comparisons": len(rows), "groups": len(set(episode_groups.values()))}, indent=2))


if __name__ == "__main__":
    main()
