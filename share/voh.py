from __future__ import annotations

from typing import Sequence

import numpy as np

from .dqn import DQNEnsemblePolicy, deterministic_action
from .features import FeatureEncoder
from .policies import Policy
from .types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


def expected_branch_reward(outcome, action: AuthorityAction, config: RewardConfig) -> float:
    """Expected immediate reward from replay-simulator probability metadata."""
    probability = float(outcome.metadata["success_probability"])
    unsafe_given_failure = float(outcome.metadata.get("unsafe_probability_given_failure", 0.0))
    success_quality, failure_quality = ((0.9, -0.15) if action == AuthorityAction.HUMAN else (0.8, -0.35))
    value = config.quality_weight * (
        probability * success_quality + (1.0 - probability) * failure_quality
    )
    value -= config.latency_weight * outcome.latency
    value -= config.execution_weight * outcome.execution_cost
    value -= config.ineffective_weight * (1.0 - probability)
    value += config.success_reward * probability
    value -= config.unsafe_penalty * (1.0 - probability) * unsafe_given_failure
    return float(value)


def expected_voh(episodes: Sequence[RepairEpisode], config: RewardConfig, gamma: float = 0.95) -> np.ndarray:
    """Human value minus expected recursive autonomous value for every state."""
    output: list[float] = []
    for episode in episodes:
        ai_values = [0.0] * len(episode.steps)
        for index in range(len(episode.steps) - 1, -1, -1):
            outcome = episode.steps[index].outcome(AuthorityAction.AI)
            probability = float(outcome.metadata["success_probability"])
            continuation = 0.0 if index + 1 == len(episode.steps) else gamma * (1.0 - probability) * ai_values[index + 1]
            ai_values[index] = expected_branch_reward(outcome, AuthorityAction.AI, config) + continuation
        for index, step in enumerate(episode.steps):
            human = expected_branch_reward(step.outcome(AuthorityAction.HUMAN), AuthorityAction.HUMAN, config)
            output.append(human - ai_values[index])
    return np.asarray(output, dtype=np.float64)


class ExpectedVoHFilterPolicy(Policy):
    """Use expected VoH only to filter DDQN Human proposals; preserve AI/Stop."""

    name = "share_expected_voh_filter"

    def __init__(self, base: DQNEnsemblePolicy, head, encoder: FeatureEncoder, threshold: float):
        self.base = base
        self.head = head
        self.encoder = encoder
        self.threshold = float(threshold)
        self._cache: dict[tuple[str, int], tuple[np.ndarray, float]] = {}

    @staticmethod
    def _key(state: RepairState) -> tuple[str, int]:
        return state.episode_id, state.step

    def prime(self, states: Sequence[RepairState]) -> None:
        unseen = [state for state in states if self._key(state) not in self._cache]
        if not unseen:
            return
        for state, values, score in zip(
            unseen, self.base.values_batch(unseen), self.head.predict(self.encoder.transform(unseen))
        ):
            self._cache[self._key(state)] = (np.asarray(values, dtype=np.float64), float(score))

    def escalation_score(self, state: RepairState) -> float:
        if self._key(state) not in self._cache:
            self.prime([state])
        return self._cache[self._key(state)][1] - self.threshold

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        if self._key(state) not in self._cache:
            self.prime([state])
        values, score = self._cache[self._key(state)]
        action = deterministic_action(values, feasible)
        if action == AuthorityAction.HUMAN and score <= self.threshold:
            action = deterministic_action(values, [item for item in feasible if item != AuthorityAction.HUMAN])
        return action
