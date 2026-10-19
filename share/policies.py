from __future__ import annotations

import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from .features import FeatureEncoder
from .types import AuthorityAction, RepairEpisode, RepairState


class Policy(ABC):
    name = "policy"

    @abstractmethod
    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        raise NotImplementedError

    def escalation_score(self, state: RepairState) -> float:
        return float("nan")


class AutonomousPolicy(Policy):
    name = "autonomous_flare"

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP


class AlwaysHumanPolicy(Policy):
    name = "always_human"

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        return AuthorityAction.HUMAN


class RandomBudgetPolicy(Policy):
    name = "random_budget"

    def __init__(self, intervention_rate: float, seed: int = 0):
        self.intervention_rate = intervention_rate
        self.rng = random.Random(seed)
        self.decisions: dict[str, bool] = {}

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        escalate = self.decisions.setdefault(state.episode_id, self.rng.random() < self.intervention_rate)
        if escalate:
            return AuthorityAction.HUMAN
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP

    def escalation_score(self, state: RepairState) -> float:
        return self.intervention_rate


class ThresholdPolicy(Policy):
    def __init__(self, feature: str, threshold: float, high_means_human: bool = True, name: str = "threshold"):
        self.feature = feature
        self.threshold = threshold
        self.high_means_human = high_means_human
        self.name = name

    def escalation_score(self, state: RepairState) -> float:
        value = float(getattr(state, self.feature))
        return value if self.high_means_human else -value

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        value = float(getattr(state, self.feature))
        escalate = value >= self.threshold if self.high_means_human else value <= self.threshold
        if escalate:
            return AuthorityAction.HUMAN
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP


class SupervisedBenefitPolicy(Policy):
    name = "static_benefit_classifier"

    def __init__(self, encoder: FeatureEncoder, intervention_rate: float = 0.2, seed: int = 0):
        self.encoder = encoder
        self.intervention_rate = intervention_rate
        self.model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=seed, class_weight="balanced")
        self.threshold = 0.5

    def fit(self, episodes: Sequence[RepairEpisode], validation: Sequence[RepairEpisode] = ()) -> "SupervisedBenefitPolicy":
        states, labels = training_states_labels(episodes)
        self.model.fit(self.encoder.transform(states), labels)
        calibration = [step.state for episode in validation for step in episode.steps] or states
        scores = (
            np.asarray([max(self._score(step.state) for step in episode.steps) for episode in validation])
            if validation else self.model.predict_proba(self.encoder.transform(calibration))[:, self._positive_column()]
        )
        self.threshold = budget_threshold(scores, self.intervention_rate)
        return self

    def _score(self, state: RepairState) -> float:
        return float(self.model.predict_proba(self.encoder.transform([state]))[0, self._positive_column()])

    def _positive_column(self) -> int:
        classes = list(self.model.classes_)
        return classes.index(1) if 1 in classes else 0

    def escalation_score(self, state: RepairState) -> float:
        return self._score(state)

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        if self.escalation_score(state) >= self.threshold:
            return AuthorityAction.HUMAN
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP


class ContextualBanditPolicy(Policy):
    name = "contextual_bandit"

    def __init__(self, encoder: FeatureEncoder, intervention_rate: float = 0.2, alpha: float = 1.0):
        self.encoder = encoder
        self.intervention_rate = intervention_rate
        self.models = [Ridge(alpha=alpha) for _ in AuthorityAction]
        self.human_penalty = 0.0

    def fit(self, episodes: Sequence[RepairEpisode], rewards: Sequence[Sequence[Sequence[float]]], validation: Sequence[RepairEpisode] = ()) -> "ContextualBanditPolicy":
        states = [step.state for episode in episodes for step in episode.steps]
        x = self.encoder.transform(states)
        flat_rewards = [row for episode_rows in rewards for row in episode_rows]
        for action in AuthorityAction:
            self.models[action].fit(x, [row[action] for row in flat_rewards])
        advantages = (
            [max(self._advantage(step.state) for step in episode.steps) for episode in validation]
            if validation else [self._advantage(state) for state in states]
        )
        self.human_penalty = budget_threshold(np.asarray(advantages), self.intervention_rate)
        return self

    def _advantage(self, state: RepairState) -> float:
        values = self._raw_values(state)
        return float(values[AuthorityAction.HUMAN] - max(values[AuthorityAction.AI], values[AuthorityAction.STOP]))

    def _raw_values(self, state: RepairState) -> np.ndarray:
        x = self.encoder.transform([state])
        return np.asarray([model.predict(x)[0] for model in self.models])

    def values(self, state: RepairState) -> np.ndarray:
        values = self._raw_values(state)
        values[AuthorityAction.HUMAN] -= self.human_penalty
        return values

    def escalation_score(self, state: RepairState) -> float:
        raw = self._raw_values(state)
        return float(raw[AuthorityAction.HUMAN] - raw[AuthorityAction.AI])

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        values = self.values(state)
        return max(feasible, key=lambda action: values[action])


class LoggedAssessmentPolicy(Policy):
    """Evaluation-only adapter for pre-recorded LLM escalation scores."""

    name = "llm_self_assessment"

    def __init__(self, scores: dict[tuple[str, int], float], threshold: float):
        self.scores = scores
        self.threshold = threshold

    @classmethod
    def from_jsonl(cls, path, threshold: float = 0.5) -> "LoggedAssessmentPolicy":
        scores = {}
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["episode_id"]), int(row["step"]))
                value = float(row["human_needed_score"])
                if not 0 <= value <= 1 or key in scores:
                    raise ValueError(f"invalid or duplicate assessment at line {line_no}")
                scores[key] = value
        return cls(scores, threshold)

    def escalation_score(self, state: RepairState) -> float:
        key = (state.episode_id, state.step)
        if key not in self.scores:
            raise KeyError(f"missing logged LLM assessment for {key}")
        return self.scores[key]

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        if self.escalation_score(state) >= self.threshold:
            return AuthorityAction.HUMAN
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP


def training_states_labels(episodes: Sequence[RepairEpisode]) -> tuple[List[RepairState], List[int]]:
    states = [step.state for episode in episodes for step in episode.steps]
    labels = [int(step.human_beneficial) for episode in episodes for step in episode.steps]
    if len(set(labels)) < 2:
        raise ValueError("benefit classifier requires both positive and negative training labels")
    return states, labels


def budget_threshold(scores: np.ndarray, intervention_rate: float) -> float:
    if intervention_rate <= 0:
        return float("inf")
    if intervention_rate >= 1:
        return float("-inf")
    return float(np.quantile(scores, 1.0 - intervention_rate, method="higher"))
