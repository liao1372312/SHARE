from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Sequence

from scipy.stats import beta


@dataclass(frozen=True)
class OneSidedSafetyCertificate:
    observations: int
    unsafe_actions: int
    target: float
    confidence: float
    upper_bound: float
    certified: bool


@dataclass(frozen=True)
class GroupMeanSafetyCertificate:
    groups: int
    observed_mean: float
    target: float
    confidence: float
    bound_range: float
    upper_bound: float
    certified: bool


def clopper_pearson_upper_bound(unsafe_actions: int, observations: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound for a Bernoulli unsafe-action rate."""
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0 <= unsafe_actions <= observations:
        raise ValueError("unsafe_actions must be within [0, observations]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if unsafe_actions == observations:
        return 1.0
    return float(beta.ppf(confidence, unsafe_actions + 1, observations - unsafe_actions))


def certify_unsafe_rate(unsafe_actions: int, observations: int, target: float, confidence: float = 0.95) -> OneSidedSafetyCertificate:
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    upper_bound = clopper_pearson_upper_bound(unsafe_actions, observations, confidence)
    return OneSidedSafetyCertificate(observations, unsafe_actions, target, confidence, upper_bound, upper_bound <= target)


def minimum_observations_for_zero_unsafe(target: float, confidence: float = 0.95) -> int:
    """Minimum Bernoulli observations needed to certify a target after zero unsafe events."""
    if not 0 < target < 1 or not 0 < confidence < 1:
        raise ValueError("target and confidence must be in (0, 1)")
    observations = 1
    while clopper_pearson_upper_bound(0, observations, confidence) > target:
        observations += 1
    return observations


def maximum_certifiable_unsafe_actions(observations: int, target: float, confidence: float = 0.95) -> int:
    if observations <= 0 or not 0 < target < 1 or not 0 < confidence < 1:
        raise ValueError("observations must be positive and target/confidence must be in (0, 1)")
    return max((count for count in range(observations + 1) if clopper_pearson_upper_bound(count, observations, confidence) <= target), default=-1)


def certify_policy_costs(human_episodes: int, unsafe_episodes: int, observations: int, human_budget: float, unsafe_budget: float, family_confidence: float = 0.95) -> tuple[OneSidedSafetyCertificate, OneSidedSafetyCertificate]:
    """Bonferroni two-constraint certificate for one frozen policy."""
    if not 0 < family_confidence < 1:
        raise ValueError("family_confidence must be in (0, 1)")
    per_constraint_confidence = 1.0 - (1.0 - family_confidence) / 2.0
    return (
        certify_unsafe_rate(human_episodes, observations, human_budget, per_constraint_confidence),
        certify_unsafe_rate(unsafe_episodes, observations, unsafe_budget, per_constraint_confidence),
    )


def hoeffding_upper_bound(observed_mean: float, groups: int, confidence: float = 0.95, bound_range: float = 1.0) -> float:
    """Distribution-free one-sided bound for independent group-level costs in [0, bound_range]."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    if not 0 <= observed_mean <= bound_range or not 0 < confidence < 1 or bound_range <= 0:
        raise ValueError("observed_mean must be in [0, bound_range], confidence in (0, 1), and bound_range positive")
    return min(bound_range, observed_mean + bound_range * sqrt(log(1.0 / (1.0 - confidence)) / (2.0 * groups)))


def certify_group_mean(observed_mean: float, groups: int, target: float, confidence: float = 0.95, bound_range: float = 1.0) -> GroupMeanSafetyCertificate:
    if not 0 < target < bound_range:
        raise ValueError("target must be in (0, bound_range)")
    upper_bound = hoeffding_upper_bound(observed_mean, groups, confidence, bound_range)
    return GroupMeanSafetyCertificate(groups, observed_mean, target, confidence, bound_range, upper_bound, upper_bound <= target)


def minimum_groups_for_zero_mean(target: float, confidence: float = 0.95, bound_range: float = 1.0) -> int:
    if not 0 < target < bound_range or not 0 < confidence < 1 or bound_range <= 0:
        raise ValueError("target must be in (0, bound_range), confidence in (0, 1), and bound_range positive")
    return ceil(bound_range * bound_range * log(1.0 / (1.0 - confidence)) / (2.0 * target * target))
