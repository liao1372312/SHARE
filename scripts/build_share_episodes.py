from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import build_synthetic_episodes, read_flare_rows, write_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicitly synthetic paired SHARE repair episodes from FLARE traces.")
    parser.add_argument("--data", type=Path, default=Path("datasets/flare_complexity"))
    parser.add_argument("--out", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repair-budget", type=int, default=3)
    parser.add_argument("--human-margin", type=float, default=0.1)
    args = parser.parse_args()
    episodes = build_synthetic_episodes(read_flare_rows(args.data, args.limit), args.seed, args.repair_budget, args.human_margin)
    write_episodes(args.out, episodes)
    summary = {
        "output": str(args.out),
        "source": "share-synthetic-v1",
        "episodes": len(episodes),
        "splits": dict(Counter(episode.split for episode in episodes)),
        "decision_states": sum(len(episode.steps) for episode in episodes),
        "multi_step_episodes": sum(len(episode.steps) > 1 for episode in episodes),
        "human_beneficial_states": sum(step.human_beneficial for episode in episodes for step in episode.steps),
    }
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

