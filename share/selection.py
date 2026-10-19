from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import stable_uniform
from .evaluation import evaluate_policy
from .policies import Policy
from .support import JointCalibratedPolicy, SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy
from .types import RepairEpisode, RewardConfig


@dataclass(frozen=True)
class StableSelectionConfig:
    """Predeclared group-stability selection rule for learned policy families."""

    human_operating_target: float = 0.15
    unsafe_operating_target: float = 0.03
    folds: int = 3
    seed: int = 17
    max_stopped_rate: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.human_operating_target <= 1.0:
            raise ValueError("human_operating_target must be in [0, 1]")
        if self.unsafe_operating_target < 0.0 or self.folds < 2:
            raise ValueError("unsafe_operating_target must be non-negative and folds at least two")
        if self.max_stopped_rate is not None and not 0.0 <= self.max_stopped_rate <= 1.0:
            raise ValueError("max_stopped_rate must be in [0, 1] when supplied")


@dataclass(frozen=True)
class CandidateSelection:
    candidate: str
    stable: bool
    fold_metrics: tuple[dict[str, float], ...]
    mean_metrics: dict[str, float]


@dataclass(frozen=True)
class StableSelectionResult:
    config: StableSelectionConfig
    selected: str | None
    candidates: tuple[CandidateSelection, ...]

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "selected": self.selected,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


def group_folds(episodes: Sequence[RepairEpisode], config: StableSelectionConfig) -> tuple[tuple[RepairEpisode, ...], ...]:
    """Assign full workflow groups to deterministic, non-empty selection folds."""
    by_group: dict[str, list[RepairEpisode]] = {}
    for episode in episodes:
        by_group.setdefault(episode.group_id, []).append(episode)
    if len(by_group) < config.folds:
        raise ValueError("selection split has fewer workflow groups than folds")
    ordered_groups = sorted(by_group, key=lambda group: (stable_uniform(f"share-guard-fold:{config.seed}:{group}"), group))
    folds: list[list[RepairEpisode]] = [[] for _ in range(config.folds)]
    for index, group in enumerate(ordered_groups):
        folds[index % config.folds].extend(by_group[group])
    return tuple(tuple(fold) for fold in folds)


def evaluate_selection_policy(policy: Policy, episodes: Sequence[RepairEpisode], reward_config: RewardConfig | None = None) -> dict[str, float]:
    if isinstance(policy, (SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy, JointCalibratedPolicy)):
        policy.reset_diagnostics()
    result = evaluate_policy(policy, episodes, reward_config)
    decisions = max(1, int(result["decision_states"]))
    fallback_actions = policy.fallback_actions if isinstance(policy, (SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy, JointCalibratedPolicy)) else 0
    masked_actions = policy.masked_ai_actions if isinstance(policy, (SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy, JointCalibratedPolicy)) else 0
    routed_actions = policy.routed_ai_actions if isinstance(policy, SupportRoutedPolicy) else 0
    routed_human_actions = policy.routed_human_actions if isinstance(policy, SupportRoutedPolicy) else 0
    routed_stop_actions = policy.routed_stop_actions if isinstance(policy, SupportRoutedPolicy) else 0
    joint_actions = policy.joint_calibrated_actions if isinstance(policy, JointCalibratedPolicy) else 0
    joint_action_counts = policy.joint_action_counts if isinstance(policy, JointCalibratedPolicy) else {}
    return {
        "episodes": float(result["episodes"]),
        "hir": float(result["hir"]),
        "unsafe_autonomy_cost_rate": float(result["unsafe_autonomy_cost_rate"]),
        "normalized_return": float(result["normalized_return"]),
        "stopped_rate": float(result["stopped_rate"]),
        "unsupported_action_fallback_rate": fallback_actions / decisions,
        "unsupported_ai_mask_rate": masked_actions / decisions,
        "unsupported_ai_route_rate": routed_actions / decisions,
        "unsupported_ai_route_human_rate": routed_human_actions / decisions,
        "unsupported_ai_route_stop_rate": routed_stop_actions / decisions,
        "joint_calibrated_action_rate": joint_actions / decisions,
        "joint_calibrated_ai_rate": int(joint_action_counts.get(0, 0)) / decisions,
        "joint_calibrated_human_rate": int(joint_action_counts.get(1, 0)) / decisions,
        "joint_calibrated_stop_rate": int(joint_action_counts.get(2, 0)) / decisions,
    }


def select_stable_candidate(candidates: Mapping[str, Policy], episodes: Sequence[RepairEpisode], config: StableSelectionConfig | None = None, reward_config: RewardConfig | None = None) -> StableSelectionResult:
    """Select only a candidate passing every fixed group-isolated selection fold.

    This is a conservative model-selection rule, not a confidence certificate.
    All candidates, including scalar and vector critics, must enter through this
    identical function with the same action-support wrapper.
    """
    if not candidates:
        raise ValueError("candidate family must be non-empty")
    resolved = config or StableSelectionConfig()
    folds = group_folds(episodes, resolved)
    ledger = []
    for name in sorted(candidates):
        metrics = tuple(evaluate_selection_policy(candidates[name], fold, reward_config) for fold in folds)
        stable = all(
            row["hir"] <= resolved.human_operating_target + 1e-12
            and row["unsafe_autonomy_cost_rate"] <= resolved.unsafe_operating_target + 1e-12
            and (resolved.max_stopped_rate is None or row["stopped_rate"] <= resolved.max_stopped_rate + 1e-12)
            for row in metrics
        )
        keys = metrics[0].keys()
        total_episodes = sum(row["episodes"] for row in metrics)
        mean_metrics = {
            key: (
                float(sum(row[key] * row["episodes"] for row in metrics) / total_episodes)
                if key != "episodes"
                else float(total_episodes)
            )
            for key in keys
        }
        ledger.append(CandidateSelection(name, stable, metrics, mean_metrics))
    feasible = [candidate for candidate in ledger if candidate.stable]
    selected = None
    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                -candidate.mean_metrics["normalized_return"],
                candidate.mean_metrics["stopped_rate"],
                candidate.mean_metrics["unsafe_autonomy_cost_rate"],
                candidate.mean_metrics["hir"],
                candidate.candidate,
            ),
        ).candidate
    return StableSelectionResult(resolved, selected, tuple(ledger))
