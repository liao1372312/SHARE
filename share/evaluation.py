from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

from .environment import RepairAuthorityEnv, base_reward
from .policies import Policy
from .types import AuthorityAction, EpisodeRecord, RepairEpisode, RewardConfig


def evaluate_policy(policy: Policy, episodes: Sequence[RepairEpisode], reward_config: RewardConfig | None = None, human_budget: float | None = None, unsafe_budget: float | None = None) -> Dict[str, Any]:
    config = reward_config or RewardConfig()
    records = [run_episode(policy, episode, config) for episode in episodes]
    successes = [record.success for record in records]
    autonomous_completed = [record for record in records if not record.human_used and record.actions[-1] != AuthorityAction.STOP]
    labels = [label for record in records for label in record.human_labels]
    scores = [score for record in records for score in record.escalation_scores]
    decisions = [int(action == AuthorityAction.HUMAN) for record in records for action in record.actions]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, decisions, average="binary", zero_division=0)
    hir = mean([record.human_used for record in records])
    uar = mean([record.unsafe or not record.success for record in autonomous_completed])
    unsafe_cost_rate = float(sum(record.unsafe_actions for record in records) / len(records)) if records else 0.0
    stopped_rate = mean([record.actions[-1] == AuthorityAction.STOP for record in records])
    selected = [value for record in records for value in record.selected_returns]
    oracle = [value for record in records for value in record.oracle_returns]
    finite = [(label, score) for label, score in zip(labels, scores) if score is not None and math.isfinite(score)]
    result = {
        "policy": policy.name,
        "episodes": len(records),
        "decision_states": len(labels),
        "repair_sr": mean(successes),
        "hir": hir,
        "uar": uar,
        "unsafe_autonomy_cost_rate": unsafe_cost_rate,
        "stopped_rate": stopped_rate,
        "normalized_return": mean([sum(record.rewards) for record in records]),
        "human_minutes_per_100": mean([record.human_minutes for record in records]) * 100.0,
        "latency": mean([record.latency for record in records]),
        "execution_cost": mean([record.execution_cost for record in records]),
        "escalation_precision": float(precision),
        "escalation_recall": float(recall),
        "escalation_f1": float(f1),
        "decision_accuracy": float(accuracy_score(labels, decisions)) if labels else 0.0,
        "collaboration_regret": mean([best - chosen for chosen, best in zip(selected, oracle)]),
        "human_budget_violation": max(0.0, hir - human_budget) if human_budget is not None else 0.0,
        "unsafe_budget_violation": max(0.0, unsafe_cost_rate - unsafe_budget) if unsafe_budget is not None else 0.0,
        "records": [asdict(record) for record in records],
    }
    if finite and len({label for label, _ in finite}) == 2:
        finite_labels, finite_scores = zip(*finite)
        result["auroc"] = float(roc_auc_score(finite_labels, finite_scores))
        result["auprc"] = float(average_precision_score(finite_labels, finite_scores))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def run_episode(policy: Policy, episode: RepairEpisode, config: RewardConfig, gamma: float = 0.95) -> EpisodeRecord:
    env = RepairAuthorityEnv(episode, config)
    state = env.reset()
    actions: List[int] = []
    rewards: List[float] = []
    labels: List[int] = []
    scores: List[float | None] = []
    selected_returns: List[float] = []
    oracle_returns: List[float] = []
    visited_steps: List[int] = []
    success = human_used = False
    # Preserve flags from the original trace independently of terminal success.
    unsafe = bool(episode.metadata.get("initial_unsafe", False))
    unsafe_actions = 0
    human_minutes = latency = execution_cost = 0.0
    while not env.done:
        paired = episode.steps[state.step]
        feasible = env.feasible_actions()
        action = policy.select(state, feasible)
        score = policy.escalation_score(state)
        result = env.step(action)
        actions.append(int(action))
        rewards.append(result.reward)
        labels.append(int(paired.human_beneficial))
        scores.append(score if math.isfinite(score) else None)
        visited_steps.append(state.step)
        success = result.outcome.success
        action_unsafe = result.outcome.unsafe or (result.done and action == AuthorityAction.AI and not result.outcome.success)
        # Ordinary unsuccessful repair contributes to UAR through ``success``;
        # it must not create a raw executor-event flag.
        unsafe = unsafe or result.outcome.unsafe
        unsafe_actions += int(action_unsafe)
        human_used = human_used or action == AuthorityAction.HUMAN
        human_minutes += result.outcome.human_minutes
        latency += result.outcome.latency
        execution_cost += result.outcome.execution_cost
        if result.next_state is not None:
            state = result.next_state
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        selected_returns.append(running)
    selected_returns.reverse()
    oracle_values = oracle_state_values(episode, config, gamma)
    oracle_returns = [oracle_values[index] for index in visited_steps]
    return EpisodeRecord(episode.episode_id, actions, rewards, success, unsafe, unsafe_actions, human_used, human_minutes, latency, execution_cost, labels, scores, selected_returns, oracle_returns)


def oracle_state_values(episode: RepairEpisode, config: RewardConfig, gamma: float = 0.95) -> List[float]:
    values = [0.0] * len(episode.steps)
    for index in range(len(episode.steps) - 1, -1, -1):
        candidates = []
        for action in AuthorityAction:
            outcome = episode.steps[index].outcome(action)
            value = base_reward(outcome, action, config)
            if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None:
                value += gamma * values[outcome.next_step]
            candidates.append(value)
        values[index] = max(candidates)
    return values


def immediate_reward_tensor(episodes: Sequence[RepairEpisode], config: RewardConfig | None = None) -> List[List[List[float]]]:
    reward_config = config or RewardConfig()
    return [
        [[base_reward(step.outcome(action), action, reward_config) for action in AuthorityAction] for step in episode.steps]
        for episode in episodes
    ]


def stratified_evaluation(policy: Policy, episodes: Sequence[RepairEpisode], reward_config: RewardConfig | None = None) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[RepairEpisode]] = {}
    for episode in episodes:
        first = episode.steps[0].state
        dimensions = {
            "failure_type": first.gap_type,
            "complexity": "simple" if first.complexity < 0.25 else "complex",
            "domain": first.domain,
            "backbone": str(episode.metadata.get("backbone", "default")),
        }
        for dimension, value in dimensions.items():
            groups.setdefault((dimension, value), []).append(episode)
    rows = []
    for (dimension, value), items in sorted(groups.items()):
        result = evaluate_policy(policy, items, reward_config)
        rows.append({**{key: item for key, item in result.items() if key != "records"}, "dimension": dimension, "stratum": value})
    return rows


def mean(values: Sequence[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0
