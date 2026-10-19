from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import AutonomousPolicy
from share.protocol import GuardPartitions, guard_development_partitions
from share.replay import FrozenLoggedReplay, collect_frozen_logged_replay
from share.support import SupportCheckedPolicy, SupportKey, support_count, support_key
from share.types import AuthorityAction, RepairEpisode, RepairState

DEFAULT_EPISODES = ROOT / "datasets" / "share_synthetic_augmented_guard_development" / "episodes.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "share_guard_augmented_support_audit" / "support_audit.json"
AUDIT_VERSION = "share-guard-support-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_summary(partitions: GuardPartitions) -> dict[str, dict[str, int]]:
    return {
        name: {"episodes": len(items), "groups": len({episode.group_id for episode in items})}
        for name, items in (
            ("fitting", partitions.fitting),
            ("selection", partitions.selection),
            ("certification", partitions.certification),
            ("test", partitions.test),
        )
    }


def format_support_key(key: SupportKey) -> dict[str, object]:
    return {
        "gap_type": key[0],
        "root_type": key[1],
        "domain": key[2],
        "complexity_bin": key[3],
        "drift_risk_bin": key[4],
    }


def summarize_support_coverage(
    episodes: Sequence[RepairEpisode],
    counts: Mapping[tuple[SupportKey, int], int],
    minimum_ai_support: int,
    top_buckets: int,
) -> dict:
    """Report fitting-replay support for every feasible-AI selection state.

    This is a static coverage audit. It deliberately walks all recorded selection
    states instead of a learned trajectory, so it exposes support gaps separately
    from a policy's preference for Stop or Human.
    """
    if minimum_ai_support <= 0:
        raise ValueError("minimum_ai_support must be positive")
    if top_buckets <= 0:
        raise ValueError("top_buckets must be positive")
    buckets: dict[SupportKey, dict] = {}
    histogram: Counter[str] = Counter()
    total = supported = 0
    groups: set[str] = set()
    for episode in episodes:
        for paired_step in episode.steps:
            state = paired_step.state
            if state.remaining_budget <= 0:
                continue
            total += 1
            groups.add(episode.group_id)
            key = support_key(state)
            count = support_count(counts, state, AuthorityAction.AI)
            supported_here = count >= minimum_ai_support
            supported += int(supported_here)
            histogram["%d" % count if count < minimum_ai_support else ">=%d" % minimum_ai_support] += 1
            bucket = buckets.setdefault(
                key,
                {
                    "support_key": format_support_key(key),
                    "selection_state_visits": 0,
                    "selection_workflow_groups": set(),
                    "ai_fitting_support": count,
                    "supported": supported_here,
                },
            )
            bucket["selection_state_visits"] += 1
            bucket["selection_workflow_groups"].add(episode.group_id)

    rows = []
    for row in buckets.values():
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "selection_workflow_groups"},
                "selection_workflow_groups": len(row["selection_workflow_groups"]),
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row["supported"]),
            -int(row["selection_state_visits"]),
            str(row["support_key"]),
        )
    )
    return {
        "selection_ai_feasible_state_visits": total,
        "selection_workflow_groups_with_feasible_ai": len(groups),
        "selection_ai_supported_state_visits": supported,
        "selection_ai_unsupported_state_visits": total - supported,
        "selection_ai_unsupported_state_rate": (total - supported) / total if total else 0.0,
        "minimum_ai_support": minimum_ai_support,
        "ai_support_count_histogram": dict(sorted(histogram.items())),
        "unique_support_buckets_reached": len(rows),
        "top_support_buckets": rows[:top_buckets],
        "all_support_buckets": rows,
    }


class AuditedSupportPolicy(SupportCheckedPolicy):
    """Support wrapper that records only fallback mechanism diagnostics."""

    def __init__(self, policy: AutonomousPolicy, counts: Mapping[tuple[SupportKey, int], int], minimum_count: int):
        super().__init__(policy, counts, minimum_count=minimum_count)
        self.fallback_states: list[RepairState] = []

    def reset_diagnostics(self) -> None:
        super().reset_diagnostics()
        self.fallback_states.clear()

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        proposed = self.policy.select(state, feasible)
        if proposed in self.checked_actions and support_count(self.counts, state, proposed) < self.minimum_count:
            if AuthorityAction.STOP not in feasible:
                raise ValueError("support fallback requires Stop to be feasible")
            self.fallback_actions += 1
            self.fallback_states.append(state)
            return AuthorityAction.STOP
        return proposed


def summarize_fallback_states(
    states: Sequence[RepairState],
    episodes: Sequence[RepairEpisode],
    counts: Mapping[tuple[SupportKey, int], int],
    minimum_ai_support: int,
    top_buckets: int,
) -> dict:
    by_episode = {episode.episode_id: episode for episode in episodes}
    buckets: dict[SupportKey, dict] = {}
    support_histogram: Counter[str] = Counter()
    successful_ai = unsafe_ai = human_beneficial = 0
    groups: set[str] = set()
    for state in states:
        episode = by_episode[state.episode_id]
        paired_step = episode.steps[state.step]
        ai = paired_step.outcome(AuthorityAction.AI)
        key = support_key(state)
        count = support_count(counts, state, AuthorityAction.AI)
        groups.add(episode.group_id)
        support_histogram[str(count)] += 1
        successful_ai += int(ai.success)
        unsafe_ai += int(ai.unsafe)
        human_beneficial += int(paired_step.human_beneficial)
        bucket = buckets.setdefault(
            key,
            {
                "support_key": format_support_key(key),
                "fallback_state_visits": 0,
                "workflow_groups": set(),
                "ai_fitting_support": count,
                "simulator_ai_successes": 0,
                "simulator_ai_unsafe": 0,
                "human_beneficial_labels": 0,
            },
        )
        bucket["fallback_state_visits"] += 1
        bucket["workflow_groups"].add(episode.group_id)
        bucket["simulator_ai_successes"] += int(ai.success)
        bucket["simulator_ai_unsafe"] += int(ai.unsafe)
        bucket["human_beneficial_labels"] += int(paired_step.human_beneficial)

    rows = []
    for row in buckets.values():
        visits = int(row["fallback_state_visits"])
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "workflow_groups"},
                "workflow_groups": len(row["workflow_groups"]),
                "simulator_ai_success_rate": row["simulator_ai_successes"] / visits,
                "simulator_ai_unsafe_rate": row["simulator_ai_unsafe"] / visits,
                "human_beneficial_rate": row["human_beneficial_labels"] / visits,
            }
        )
    rows.sort(key=lambda row: (-int(row["fallback_state_visits"]), str(row["support_key"])))
    total = len(states)
    return {
        "fallback_state_visits": total,
        "fallback_workflow_groups": len(groups),
        "minimum_ai_support": minimum_ai_support,
        "ai_support_count_histogram": dict(sorted(support_histogram.items())),
        "simulator_ai_branch_summary": {
            "warning": "These are paired synthetic branch descriptors for fallback states, not observed deployment outcomes or learned-policy counterfactuals.",
            "success_rate": successful_ai / total if total else 0.0,
            "unsafe_rate": unsafe_ai / total if total else 0.0,
            "human_beneficial_rate": human_beneficial / total if total else 0.0,
        },
        "top_fallback_buckets": rows[:top_buckets],
        "all_fallback_buckets": rows,
    }


def summarize_evaluation(result: Mapping) -> dict:
    return {key: value for key, value in result.items() if key != "records"}


def run_audit(
    episodes: Sequence[RepairEpisode],
    *,
    partition_seed: int,
    selection_fraction: float,
    behavior_seed: int,
    minimum_ai_support: int,
    top_buckets: int,
) -> dict:
    partitions = guard_development_partitions(episodes, seed=partition_seed, selection_fraction=selection_fraction)
    encoder = FeatureEncoder.fit([step.state for episode in partitions.fitting for step in episode.steps])
    replay = collect_frozen_logged_replay(partitions.fitting, encoder, behavior_seed=behavior_seed)
    coverage = summarize_support_coverage(partitions.selection, replay.support_counts, minimum_ai_support, top_buckets)

    raw_policy = AutonomousPolicy()
    raw_result = evaluate_policy(raw_policy, partitions.selection)
    supported_policy = AuditedSupportPolicy(raw_policy, replay.support_counts, minimum_ai_support)
    supported_policy.reset_diagnostics()
    supported_result = evaluate_policy(supported_policy, partitions.selection)
    fallback = summarize_fallback_states(
        supported_policy.fallback_states,
        partitions.selection,
        replay.support_counts,
        minimum_ai_support,
        top_buckets,
    )
    decisions = max(1, int(supported_result["decision_states"]))
    return {
        "audit_version": AUDIT_VERSION,
        "evidence_warning": "Synthetic train-only mechanism diagnostic. It is not a paper result, deployment-safety evidence, real paired authority data, or a locked-holdout evaluation.",
        "scope": {
            "input_requirement": "train-only SHARE-Guard development artifact",
            "selection_partition_role": "development-only; created from original train groups",
            "historical_validation_test_loaded": False,
            "learning_candidates_trained": False,
            "probe_policy": "AutonomousPolicy (always propose feasible AI)",
            "comparison_interpretation": "The probe isolates the mechanics of the AI support fallback. It does not estimate a learned controller's performance.",
        },
        "configuration": {
            "partition_seed": partition_seed,
            "selection_fraction": selection_fraction,
            "behavior_seed": behavior_seed,
            "behavior_policy": replay.behavior_policy,
            "minimum_ai_support": minimum_ai_support,
        },
        "partitions": partition_summary(partitions),
        "replay": {
            "transitions": len(replay.transitions),
            "unique_support_observations": replay.unique_support_observations,
            "content_sha256": replay.content_sha256,
        },
        "static_selection_coverage": coverage,
        "autonomous_probe": {
            "raw": summarize_evaluation(raw_result),
            "support_checked": summarize_evaluation(supported_result),
            "fallback_actions": supported_policy.fallback_actions,
            "fallback_rate_over_executed_decisions": supported_policy.fallback_actions / decisions,
            "fallback_state_breakdown": fallback,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SHARE-Guard support coverage on a train-only development artifact.")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partition-seed", type=int, default=17)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--behavior-seed", type=int, default=17)
    parser.add_argument("--minimum-ai-support", type=int, default=3)
    parser.add_argument("--top-buckets", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes.is_file():
        raise ValueError("--episodes must be one train-only JSONL artifact")
    episodes = load_episodes(args.episodes)
    report = run_audit(
        episodes,
        partition_seed=args.partition_seed,
        selection_fraction=args.selection_fraction,
        behavior_seed=args.behavior_seed,
        minimum_ai_support=args.minimum_ai_support,
        top_buckets=args.top_buckets,
    )
    report["input"] = {
        "episodes_path": str(args.episodes.resolve()),
        "episodes_sha256": sha256_file(args.episodes),
        "episodes": len(episodes),
        "workflow_groups": len({episode.group_id for episode in episodes}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "unsupported_static_rate": report["static_selection_coverage"]["selection_ai_unsupported_state_rate"],
        "probe_fallback_rate": report["autonomous_probe"]["fallback_rate_over_executed_decisions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
