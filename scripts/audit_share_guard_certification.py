from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.protocol import guard_partitions
from share.safety import certify_group_mean, maximum_certifiable_unsafe_actions, minimum_groups_for_zero_mean


def binomial_acceptance_power(groups: int, true_rate: float, max_accepted_events: int) -> float:
    return sum(comb(groups, events) * true_rate**events * (1.0 - true_rate) ** (groups - events) for events in range(max_accepted_events + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether SHARE-Guard can certify group-level constraints before training.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--family-confidence", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=Path("results/share_guard_protocol/certification_feasibility.json"))
    args = parser.parse_args()
    partitions = guard_partitions(load_episodes(args.episodes), args.seed)
    groups = defaultdict(list)
    for episode in partitions.certification:
        groups[episode.group_id].append(episode)
    count = len(groups)
    unsafe_horizon = max(len(episode.steps) for episode in partitions.certification)
    group_incident_confidence = args.family_confidence
    zero_unsafe = certify_group_mean(0.0, count, args.unsafe_budget, group_incident_confidence, unsafe_horizon)
    maximum_incidents = maximum_certifiable_unsafe_actions(count, args.unsafe_budget, group_incident_confidence)
    report = {
        "independent_unit": "workflow_group",
        "certification_episodes": len(partitions.certification),
        "certification_groups": count,
        "episodes_per_group": {"min": min(map(len, groups.values())), "max": max(map(len, groups.values()))},
        "group_incident_confidence": group_incident_confidence,
        "unsafe_budget": args.unsafe_budget,
        "human_budget": args.human_budget,
        "cluster_mean_cumulative_j_u": {
            "definition": "equal-weight mean of each workflow group's within-group episode-average cumulative unsafe action count",
            "range": [0, unsafe_horizon],
            "zero_observation_upper_bound": zero_unsafe.upper_bound,
            "zero_observation_certified": zero_unsafe.certified,
            "minimum_groups_for_zero_observation": minimum_groups_for_zero_mean(args.unsafe_budget, group_incident_confidence, unsafe_horizon),
            "warning": "This cluster-mean diagnostic is not substituted for the manuscript's episode-weighted J_U; primary J_U requires cluster-aware inference.",
        },
        "group_incident_endpoint": {
            "definition": "probability that a workflow group contains at least one unsafe action",
            "max_certifiable_incident_groups": maximum_incidents,
            "acceptance_power": {
                f"true_rate_{rate:g}": binomial_acceptance_power(count, rate, maximum_incidents)
                for rate in (0.01, 0.02, 0.03, 0.05)
            },
            "warning": "This endpoint is distinct from the manuscript's episode-level cumulative J_U.",
        },
        "conclusion": "no_strict_5_percent_cumulative_J_U_certificate_is_possible_with_current_validation_groups" if not zero_unsafe.certified else "zero_event_certificate_is_possible",
        "evidence_warning": "Synthetic protocol diagnostic only; this does not establish deployment safety.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
