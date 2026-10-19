"""Reconstruct raw unsafe flags on the frozen 250-group policy trajectories.

This is a read-only, post-hoc audit. It replays frozen policies in the original
episode order and reports branch-level flags separately from the evaluator's
derived adverse-outcome flag.
"""

from __future__ import annotations

import json
import math

from scripts.analyze_share_acceptance_experiments import (
    COMPARATOR_ARTIFACTS,
    COMPARATOR_CALIBRATION,
    COMPARATOR_CHECKPOINTS,
    V4,
    load_models,
)
from scripts.run_share_online_hard_cap import (
    BINARY_POLICIES,
    OnlineHardHumanCapPolicy,
    load_frozen_policies,
)
from share.data import load_episodes
from share.evaluation import evaluate_policy
from share.types import AuthorityAction, RewardConfig


def audit_policy(policy, episodes, reward: RewardConfig) -> dict[str, float | int]:
    guard = OnlineHardHumanCapPolicy(
        policy, math.floor(0.20 * len(episodes)), policy.name in BINARY_POLICIES
    )
    evaluation = evaluate_policy(guard, episodes, reward, 0.20, None)
    flagged = before_human = before_stop = 0
    for episode, record in zip(episodes, evaluation["records"], strict=True):
        step_index = 0
        events: list[tuple[AuthorityAction, bool]] = []
        for action_id in record["actions"]:
            action = AuthorityAction(action_id)
            outcome = episode.steps[step_index].outcome(action)
            events.append((action, bool(outcome.unsafe)))
            if action == AuthorityAction.AI and not outcome.terminal:
                assert outcome.next_step is not None
                step_index = outcome.next_step
        flagged += int(any(unsafe for _, unsafe in events))
        prior_flag = any(unsafe for _, unsafe in events[:-1])
        before_human += int(prior_flag and events[-1][0] == AuthorityAction.HUMAN)
        before_stop += int(prior_flag and events[-1][0] == AuthorityAction.STOP)
    count = len(episodes)
    return {
        "episodes": count,
        "raw_unsafe_flag_episodes": flagged,
        "raw_unsafe_flag_rate": flagged / count,
        "flag_before_human_episodes": before_human,
        "flag_before_stop_episodes": before_stop,
        "repair_sr": evaluation["repair_sr"],
        "hir": evaluation["hir"],
        "uar": evaluation["uar"],
        "stop": evaluation["stopped_rate"],
    }


def main() -> None:
    episodes = load_episodes(V4)
    encoder, _, share, _ = load_models()
    reward = RewardConfig()
    result = {"SHARE_ensemble": audit_policy(share, episodes, reward)}
    for seed in (17, 29, 43, 71, 101):
        policies = load_frozen_policies(
            seed, COMPARATOR_ARTIFACTS, COMPARATOR_CHECKPOINTS,
            COMPARATOR_CALIBRATION, encoder, reward,
        )
        for policy in policies:
            if policy.name == "share_h":
                continue
            result[f"{policy.name}_seed_{seed}"] = audit_policy(policy, episodes, reward)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
