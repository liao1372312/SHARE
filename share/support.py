from __future__ import annotations

import hashlib
from collections import Counter
from typing import Mapping, Sequence, TypeAlias

import numpy as np
from sklearn.linear_model import LogisticRegression

from .environment import RepairAuthorityEnv, base_reward
from .features import FeatureEncoder
from .policies import Policy
from .types import AuthorityAction, RepairEpisode, RepairState


SupportKey: TypeAlias = tuple[str, str, str, int, int]
NEVER_HUMAN_THRESHOLD = 1.1


def support_key(state: RepairState) -> SupportKey:
    """Coarsen a state before counting logged state-action support.

    Episode IDs are deliberately excluded: they are unique and would make every
    state unsupported. The bins are fixed protocol fields, not fitted features.
    """
    return (
        state.gap_type,
        state.root_type,
        state.domain,
        min(3, int(max(0.0, state.complexity) * 4.0)),
        min(3, int(max(0.0, state.drift_risk) * 4.0)),
    )


def support_count(counts: Mapping[tuple[SupportKey, int], int], state: RepairState, action: AuthorityAction) -> int:
    return int(counts.get((support_key(state), int(action)), 0))


def mask_unsupported_ai_actions(
    feasible: Sequence[AuthorityAction],
    state: RepairState,
    counts: Mapping[tuple[SupportKey, int], int],
    minimum_count: int,
) -> tuple[AuthorityAction, ...]:
    """Remove only unsupported AI from an already feasible action set."""
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if AuthorityAction.AI not in feasible or support_count(counts, state, AuthorityAction.AI) >= minimum_count:
        return tuple(feasible)
    masked = tuple(action for action in feasible if action != AuthorityAction.AI)
    if not masked:
        raise ValueError("support mask removed every feasible action")
    return masked


def mask_unsupported_ai_next_actions(
    masks: np.ndarray,
    next_keys: Sequence[SupportKey | None],
    counts: Mapping[tuple[SupportKey, int], int],
    minimum_count: int,
) -> np.ndarray:
    """Apply the pre-decision AI support rule to fixed-replay Bellman masks."""
    if len(masks) != len(next_keys):
        raise ValueError("support keys must align with replay masks")
    result = np.asarray(masks, dtype=bool).copy()
    for index, key in enumerate(next_keys):
        if key is not None and result[index, AuthorityAction.AI] and counts.get((key, int(AuthorityAction.AI)), 0) < minimum_count:
            result[index, AuthorityAction.AI] = False
    return result


class SupportCheckedPolicy(Policy):
    """Apply the prespecified unsupported-action fallback to a learned policy.

    The wrapper is part of the evaluation protocol and must be used for every
    learned candidate in a comparison. It does not change the environment: an
    unsupported learned action becomes the existing Stop action before execution.
    """

    def __init__(
        self,
        policy: Policy,
        counts: Mapping[tuple[SupportKey, int], int],
        minimum_count: int = 3,
        checked_actions: Sequence[AuthorityAction] = (AuthorityAction.AI,),
    ):
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive")
        if not checked_actions:
            raise ValueError("checked_actions must be non-empty")
        self.policy = policy
        self.counts = Counter(counts)
        self.minimum_count = minimum_count
        self.checked_actions = frozenset(checked_actions)
        self.name = f"{policy.name}_supported"
        self.fallback_actions = 0
        self.masked_ai_actions = 0

    def reset_diagnostics(self) -> None:
        self.fallback_actions = 0
        self.masked_ai_actions = 0

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        proposed = self.policy.select(state, feasible)
        if proposed not in self.checked_actions or support_count(self.counts, state, proposed) >= self.minimum_count:
            return proposed
        if AuthorityAction.STOP not in feasible:
            raise ValueError("support fallback requires Stop to be feasible")
        self.fallback_actions += 1
        return AuthorityAction.STOP

    def escalation_score(self, state: RepairState) -> float:
        return self.policy.escalation_score(state)


class SupportMaskedPolicy(Policy):
    """Make action support part of action selection instead of a post-hoc Stop.

    The wrapped policy compares Human and Stop whenever AI is infeasible under
    the fixed replay support rule. This mode is a new protocol option; it does
    not alter the legacy post-hoc fallback behavior.
    """

    def __init__(
        self,
        policy: Policy,
        counts: Mapping[tuple[SupportKey, int], int],
        minimum_count: int = 3,
    ):
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive")
        self.policy = policy
        self.counts = Counter(counts)
        self.minimum_count = minimum_count
        self.name = f"{policy.name}_support_masked"
        self.fallback_actions = 0
        self.masked_ai_actions = 0

    def reset_diagnostics(self) -> None:
        self.fallback_actions = 0
        self.masked_ai_actions = 0

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        masked = mask_unsupported_ai_actions(feasible, state, self.counts, self.minimum_count)
        self.masked_ai_actions += int(len(masked) != len(feasible))
        return self.policy.select(state, masked)

    def escalation_score(self, state: RepairState) -> float:
        return self.policy.escalation_score(state)


class SupportRoutedPolicy(Policy):
    """Route unsupported proposed AI actions to calibrated Human or Stop.

    Unlike ``SupportMaskedPolicy``, the base policy remains responsible for all
    supported decisions. This router intervenes only after that policy proposes
    AI in a bucket with insufficient fixed-replay support. Its Human/Stop model
    and operating threshold are fit exclusively on the fitting partition.
    """

    def __init__(
        self,
        policy: Policy,
        counts: Mapping[tuple[SupportKey, int], int],
        encoder: FeatureEncoder,
        *,
        human_budget: float,
        unsafe_budget: float,
        stop_penalty: float = 0.25,
        threshold_grid: int = 33,
        minimum_count: int = 3,
        seed: int = 17,
    ):
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive")
        if not 0.0 <= human_budget <= 1.0 or not 0.0 <= unsafe_budget <= 1.0:
            raise ValueError("router budgets must be in [0, 1]")
        if stop_penalty < 0.0 or threshold_grid < 2:
            raise ValueError("stop_penalty must be non-negative and threshold_grid at least two")
        self.policy = policy
        self.counts = Counter(counts)
        self.encoder = encoder
        self.human_budget = human_budget
        self.unsafe_budget = unsafe_budget
        self.stop_penalty = stop_penalty
        self.threshold_grid = threshold_grid
        self.minimum_count = minimum_count
        self.seed = seed
        self.name = f"{policy.name}_support_routed"
        self.model: LogisticRegression | None = None
        self.constant_score: float | None = None
        self.threshold = NEVER_HUMAN_THRESHOLD
        self.calibration: dict[str, float | int | bool] = {}
        self._calibration_records: tuple[dict[str, object], ...] = ()
        self.fallback_actions = 0
        self.masked_ai_actions = 0
        self.routed_ai_actions = 0
        self.routed_human_actions = 0
        self.routed_stop_actions = 0

    def fit(self, episodes: Sequence[RepairEpisode]) -> "SupportRoutedPolicy":
        if not episodes:
            raise ValueError("support router requires fitting episodes")
        states = [step.state for episode in episodes for step in episode.steps]
        labels = np.asarray([int(step.human_beneficial) for episode in episodes for step in episode.steps], dtype=np.int64)
        if not states:
            raise ValueError("support router requires fitting states")
        if len(np.unique(labels)) == 1:
            self.constant_score = float(labels[0])
            self.model = None
        else:
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=self.seed,
            ).fit(self.encoder.transform(states), labels)
            self.constant_score = None
        self._calibration_records = tuple(self._collect_calibration_records(episodes))
        self.threshold, metrics, feasible = self._calibrate_threshold()
        self.calibration = {
            "fitting_episodes": len(episodes),
            "fitting_workflow_groups": len({episode.group_id for episode in episodes}),
            "threshold": self.threshold,
            "threshold_grid": self.threshold_grid,
            "human_budget": self.human_budget,
            "unsafe_budget": self.unsafe_budget,
            "stop_penalty": self.stop_penalty,
            "feasible": feasible,
            **metrics,
        }
        self.reset_diagnostics()
        return self

    def reset_diagnostics(self) -> None:
        self.fallback_actions = 0
        self.masked_ai_actions = 0
        self.routed_ai_actions = 0
        self.routed_human_actions = 0
        self.routed_stop_actions = 0

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        proposed = self.policy.select(state, feasible)
        if proposed != AuthorityAction.AI or support_count(self.counts, state, AuthorityAction.AI) >= self.minimum_count:
            return proposed
        self.routed_ai_actions += 1
        if self._human_score(state) >= self.threshold and AuthorityAction.HUMAN in feasible:
            self.routed_human_actions += 1
            return AuthorityAction.HUMAN
        if AuthorityAction.STOP not in feasible:
            raise ValueError("support router requires Stop to be feasible")
        self.routed_stop_actions += 1
        return AuthorityAction.STOP

    def escalation_score(self, state: RepairState) -> float:
        return self._human_score(state)

    def _human_score(self, state: RepairState) -> float:
        if self.model is None:
            if self.constant_score is None:
                raise RuntimeError("support router must be fit before selection")
            return self.constant_score
        classes = list(self.model.classes_)
        positive = classes.index(1) if 1 in classes else 0
        return float(self.model.predict_proba(self.encoder.transform([state]))[0, positive])

    def _calibrate_threshold(self) -> tuple[float, dict[str, float], bool]:
        potential_scores = [float(record["score"]) for record in self._calibration_records if bool(record["routed"])]
        if not potential_scores:
            metrics = self._operating_metrics(NEVER_HUMAN_THRESHOLD)
            return NEVER_HUMAN_THRESHOLD, metrics, self._within_budgets(metrics)
        quantiles = np.linspace(0.0, 1.0, self.threshold_grid)
        thresholds = sorted({float(np.quantile(potential_scores, quantile, method="higher")) for quantile in quantiles}, reverse=True)
        thresholds.insert(0, NEVER_HUMAN_THRESHOLD)
        best_feasible: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
        best_infeasible: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
        for threshold in thresholds:
            metrics = self._operating_metrics(threshold)
            objective = metrics["normalized_return"] - self.stop_penalty * metrics["stopped_rate"]
            feasible = self._within_budgets(metrics)
            if feasible:
                key = (objective, -metrics["hir"], -metrics["unsafe_autonomy_cost_rate"], -metrics["stopped_rate"])
                if best_feasible is None or key > best_feasible[0]:
                    best_feasible = (key, threshold, metrics)
            else:
                violation = max(0.0, metrics["hir"] - self.human_budget) + max(0.0, metrics["unsafe_autonomy_cost_rate"] - self.unsafe_budget)
                key = (-violation, objective, -metrics["hir"], -metrics["stopped_rate"])
                if best_infeasible is None or key > best_infeasible[0]:
                    best_infeasible = (key, threshold, metrics)
        if best_feasible is not None:
            return best_feasible[1], best_feasible[2], True
        assert best_infeasible is not None
        return best_infeasible[1], best_infeasible[2], False

    def _collect_calibration_records(self, episodes: Sequence[RepairEpisode]) -> list[dict[str, object]]:
        """Freeze each fitting trajectory through its first router intervention.

        Human and Stop both terminate in the current environment. Therefore a
        threshold can alter only the first unsupported-AI proposal on a base
        trajectory; preserving its prefix and the two terminal branches is
        equivalent to replaying the full fitting set for every threshold.
        """
        records: list[dict[str, object]] = []
        for episode in episodes:
            env = RepairAuthorityEnv(episode)
            state = env.reset()
            total_reward = 0.0
            unsafe_actions = 0
            decisions = 0
            human_used = stopped = False
            while not env.done:
                proposed = self.policy.select(state, env.feasible_actions())
                if proposed == AuthorityAction.AI and support_count(self.counts, state, AuthorityAction.AI) < self.minimum_count:
                    paired = episode.steps[state.step]
                    human = paired.outcome(AuthorityAction.HUMAN)
                    stop = paired.outcome(AuthorityAction.STOP)
                    common = {
                        "routed": True,
                        "score": self._human_score(state),
                        "decisions": decisions + 1,
                    }
                    records.append({
                        **common,
                        "human_used": True,
                        "stopped": False,
                        "unsafe_actions": unsafe_actions,
                        "total_reward": total_reward + base_reward(human, AuthorityAction.HUMAN, env.reward_config),
                    })
                    records.append({
                        **common,
                        "human_used": human_used,
                        "stopped": True,
                        "unsafe_actions": unsafe_actions,
                        "total_reward": total_reward + base_reward(stop, AuthorityAction.STOP, env.reward_config),
                    })
                    break
                result = env.step(proposed)
                decisions += 1
                total_reward += result.reward
                human_used = human_used or proposed == AuthorityAction.HUMAN
                stopped = stopped or proposed == AuthorityAction.STOP
                unsafe_actions += int(result.outcome.unsafe or (result.done and proposed == AuthorityAction.AI and not result.outcome.success))
                if result.next_state is not None:
                    state = result.next_state
            else:
                records.append({
                    "routed": False,
                    "score": float("nan"),
                    "decisions": decisions,
                    "human_used": human_used,
                    "stopped": stopped,
                    "unsafe_actions": unsafe_actions,
                    "total_reward": total_reward,
                })
        # A routed episode stores Human then Stop alternatives consecutively.
        return records

    def _operating_metrics(self, threshold: float) -> dict[str, float]:
        human_episodes = unsafe_actions = stopped_episodes = decisions = routed = 0
        total_return = 0.0
        index = 0
        episodes = 0
        while index < len(self._calibration_records):
            record = self._calibration_records[index]
            if bool(record["routed"]):
                score = float(record["score"])
                record = self._calibration_records[index if score >= threshold else index + 1]
                routed += 1
                index += 2
            else:
                index += 1
            episodes += 1
            human_episodes += int(bool(record["human_used"]))
            stopped_episodes += int(bool(record["stopped"]))
            unsafe_actions += int(record["unsafe_actions"])
            decisions += int(record["decisions"])
            total_return += float(record["total_reward"])
        total = max(1, episodes)
        return {
            "hir": human_episodes / total,
            "unsafe_autonomy_cost_rate": unsafe_actions / total,
            "stopped_rate": stopped_episodes / total,
            "normalized_return": total_return / total,
            "decision_states": float(decisions),
            "routed_ai_rate": routed / max(1, decisions),
        }

    def _within_budgets(self, metrics: Mapping[str, float]) -> bool:
        return (
            metrics["hir"] <= self.human_budget + 1e-12
            and metrics["unsafe_autonomy_cost_rate"] <= self.unsafe_budget + 1e-12
        )


class JointCalibratedPolicy(Policy):
    """Fit-only joint calibration over AI, Human, and Stop operating choices.

    The frozen-replay critic supplies action values. On the fitting partition,
    this wrapper searches non-negative AI and Human action penalties. Unsupported
    AI stays infeasible, while Stop remains available and is discouraged by a
    predeclared Stop term in the fitting-only objective. Selection data are not
    used for either the score grid or the chosen penalty pair.
    """

    def __init__(
        self,
        policy: Policy,
        counts: Mapping[tuple[SupportKey, int], int],
        *,
        human_budget: float,
        unsafe_budget: float,
        max_stopped_rate: float,
        stop_penalty: float = 0.25,
        grid_size: int = 9,
        minimum_count: int = 3,
        human_reserve_rate: float = 0.0,
        reserve_seed: int = 17,
        human_reserve_mode: str = "hash",
        calibration_folds: int = 1,
        human_reserve_grid_size: int = 1,
    ):
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive")
        if not 0.0 <= human_budget <= 1.0 or not 0.0 <= unsafe_budget <= 1.0 or not 0.0 <= max_stopped_rate <= 1.0:
            raise ValueError("joint calibration budgets must be in [0, 1]")
        if stop_penalty < 0.0 or grid_size < 2:
            raise ValueError("joint stop penalty must be non-negative and grid size at least two")
        if not 0.0 <= human_reserve_rate <= 1.0:
            raise ValueError("joint human reserve rate must be in [0, 1]")
        if human_reserve_mode not in {"hash", "risk"}:
            raise ValueError("joint human reserve mode must be hash or risk")
        if calibration_folds < 1:
            raise ValueError("joint calibration folds must be positive")
        if human_reserve_grid_size < 1:
            raise ValueError("joint human reserve grid size must be positive")
        self.policy = policy
        self.counts = Counter(counts)
        self.human_budget = human_budget
        self.unsafe_budget = unsafe_budget
        self.max_stopped_rate = max_stopped_rate
        self.stop_penalty = stop_penalty
        self.grid_size = grid_size
        self.minimum_count = minimum_count
        self.human_reserve_rate = human_reserve_rate
        self.reserve_seed = reserve_seed
        self.human_reserve_mode = human_reserve_mode
        self.calibration_folds = calibration_folds
        self.human_reserve_grid_size = human_reserve_grid_size
        self.name = f"{policy.name}_joint_calibrated"
        self.ai_penalty = 0.0
        self.human_penalty = 0.0
        self.calibration: dict[str, float | int | bool] = {}
        self.fallback_actions = 0
        self.masked_ai_actions = 0
        self.joint_calibrated_actions = 0
        self.joint_action_counts: Counter[int] = Counter()
        self._value_cache: dict[tuple[str, int], np.ndarray] = {}
        self._reserve_cache: dict[tuple[str, int], bool] = {}
        self._risk_model: LogisticRegression | None = None
        self._risk_constant: float | None = None
        self.risk_reserve_threshold = float("inf")
        self._risk_score_by_state: dict[tuple[str, int], float] = {}
        self._risk_episode_scores = np.asarray([], dtype=np.float64)

    def fit(self, episodes: Sequence[RepairEpisode]) -> "JointCalibratedPolicy":
        if not episodes:
            raise ValueError("joint calibration requires fitting episodes")
        self._fit_human_reserve(episodes)
        states = [step.state for episode in episodes for step in episode.steps]
        batched_values = self._base_values_batch(states)
        self._value_cache = {
            (state.episode_id, state.step): value
            for state, value in zip(states, batched_values)
        }
        ai_penalties, human_penalties = self._penalty_grid(episodes)
        calibration_folds = self._group_calibration_folds(episodes)
        reserve_rates = self._human_reserve_grid()
        best_feasible = None
        best_infeasible = None
        for reserve_rate in reserve_rates:
            self._set_human_reserve_rate(reserve_rate)
            for ai_penalty in ai_penalties:
                for human_penalty in human_penalties:
                    metrics = self._operating_metrics(episodes, ai_penalty, human_penalty)
                    fold_metrics = tuple(self._operating_metrics(fold, ai_penalty, human_penalty) for fold in calibration_folds)
                    objective = metrics["normalized_return"] - self.stop_penalty * metrics["stopped_rate"]
                    key = (objective, -metrics["stopped_rate"], -metrics["hir"], -metrics["unsafe_autonomy_cost_rate"])
                    if all(self._within_budgets(row) for row in fold_metrics):
                        if best_feasible is None or key > best_feasible[0]:
                            best_feasible = (key, reserve_rate, ai_penalty, human_penalty, metrics, fold_metrics)
                    else:
                        violation = max(
                            max(0.0, row["hir"] - self.human_budget)
                            + max(0.0, row["unsafe_autonomy_cost_rate"] - self.unsafe_budget)
                            + max(0.0, row["stopped_rate"] - self.max_stopped_rate)
                            for row in fold_metrics
                        )
                        fallback_key = (-violation, *key)
                        if best_infeasible is None or fallback_key > best_infeasible[0]:
                            best_infeasible = (fallback_key, reserve_rate, ai_penalty, human_penalty, metrics, fold_metrics)
        feasible = best_feasible is not None
        selected = best_feasible or best_infeasible
        if selected is None:
            raise RuntimeError("joint calibration grid is empty")
        _, reserve_rate, self.ai_penalty, self.human_penalty, metrics, fold_metrics = selected
        self._set_human_reserve_rate(reserve_rate)
        self.calibration = {
            "fitting_episodes": len(episodes),
            "fitting_workflow_groups": len({episode.group_id for episode in episodes}),
            "ai_penalty": self.ai_penalty,
            "human_penalty": self.human_penalty,
            "grid_size": self.grid_size,
            "ai_penalty_candidates": len(ai_penalties),
            "human_penalty_candidates": len(human_penalties),
            "human_budget": self.human_budget,
            "unsafe_budget": self.unsafe_budget,
            "max_stopped_rate": self.max_stopped_rate,
            "stop_penalty": self.stop_penalty,
            "human_reserve_rate": self.human_reserve_rate,
            "human_reserve_rate_cap": max(reserve_rates),
            "human_reserve_rate_candidates": len(reserve_rates),
            "reserve_seed": self.reserve_seed,
            "human_reserve_mode": self.human_reserve_mode,
            "risk_reserve_threshold": self.risk_reserve_threshold if np.isfinite(self.risk_reserve_threshold) else None,
            "calibration_folds": self.calibration_folds,
            "max_fold_hir": max(row["hir"] for row in fold_metrics),
            "max_fold_unsafe_autonomy_cost_rate": max(row["unsafe_autonomy_cost_rate"] for row in fold_metrics),
            "max_fold_stopped_rate": max(row["stopped_rate"] for row in fold_metrics),
            "feasible": feasible,
            **metrics,
        }
        self.reset_diagnostics()
        return self

    def reset_diagnostics(self) -> None:
        self.fallback_actions = 0
        self.masked_ai_actions = 0
        self.joint_calibrated_actions = 0
        self.joint_action_counts.clear()

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        action, masked = self._select_with_penalties(state, feasible, self.ai_penalty, self.human_penalty)
        self.masked_ai_actions += int(masked)
        self.joint_calibrated_actions += 1
        self.joint_action_counts[int(action)] += 1
        return action

    def escalation_score(self, state: RepairState) -> float:
        if self._is_human_reserved(state):
            return float("inf")
        values = self._base_values(state)
        return float(values[AuthorityAction.HUMAN] - self.human_penalty - (values[AuthorityAction.AI] - self.ai_penalty))

    def _base_values(self, state: RepairState) -> np.ndarray:
        if hasattr(self.policy, "values"):
            values = self.policy.values(state)
        elif hasattr(self.policy, "scalar_values"):
            values = self.policy.scalar_values(state)
        else:
            raise TypeError("joint calibration requires a policy exposing values or scalar_values")
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(AuthorityAction),):
            raise ValueError("joint calibration policy values must contain AI, Human, and Stop scores")
        return result

    def _base_values_batch(self, states: Sequence[RepairState]) -> np.ndarray:
        if hasattr(self.policy, "values_batch"):
            values = self.policy.values_batch(states)
        elif hasattr(self.policy, "scalar_values_batch"):
            values = self.policy.scalar_values_batch(states)
        else:
            values = np.stack([self._base_values(state) for state in states])
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(states), len(AuthorityAction)):
            raise ValueError("joint calibration batched policy values must be states by actions")
        return result

    def _cached_values(self, state: RepairState) -> np.ndarray:
        key = (state.episode_id, state.step)
        if key in self._value_cache:
            return self._value_cache[key]
        return self._base_values(state)

    def _penalty_grid(self, episodes: Sequence[RepairEpisode]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        # Each calibrated action competes with *both* alternatives.  Using only
        # its advantage over the currently best alternative misses operating
        # regions that become reachable after the other action is penalized.
        # For example, Human may initially trail AI and Stop, yet become optimal
        # after an AI penalty; its Human-vs-Stop breakpoint must still be in the
        # grid.  Include all pairwise non-negative Q gaps before taking the
        # bounded quantile grid.
        ai_breakpoints = []
        human_breakpoints = []
        for episode in episodes:
            for paired_step in episode.steps:
                state = paired_step.state
                values = self._cached_values(state)
                ai_breakpoints.extend((
                    max(0.0, float(values[AuthorityAction.AI] - values[AuthorityAction.HUMAN])),
                    max(0.0, float(values[AuthorityAction.AI] - values[AuthorityAction.STOP])),
                ))
                human_breakpoints.extend((
                    max(0.0, float(values[AuthorityAction.HUMAN] - values[AuthorityAction.AI])),
                    max(0.0, float(values[AuthorityAction.HUMAN] - values[AuthorityAction.STOP])),
                ))
        return self._nonnegative_quantiles(ai_breakpoints), self._nonnegative_quantiles(human_breakpoints)

    def _nonnegative_quantiles(self, values: Sequence[float]) -> tuple[float, ...]:
        maximum = max(values, default=0.0)
        quantiles = np.linspace(0.0, 1.0, self.grid_size)
        candidates = {0.0, maximum + 1.0}
        candidates.update(float(np.quantile(values, quantile, method="higher")) for quantile in quantiles)
        return tuple(sorted(candidates))

    def _select_with_penalties(
        self,
        state: RepairState,
        feasible: Sequence[AuthorityAction],
        ai_penalty: float,
        human_penalty: float,
    ) -> tuple[AuthorityAction, bool]:
        if AuthorityAction.HUMAN in feasible and self._is_human_reserved(state):
            return AuthorityAction.HUMAN, False
        masked_feasible = mask_unsupported_ai_actions(feasible, state, self.counts, self.minimum_count)
        values = self._cached_values(state).copy()
        values[AuthorityAction.AI] -= ai_penalty
        values[AuthorityAction.HUMAN] -= human_penalty
        tie_break = {AuthorityAction.STOP: 0, AuthorityAction.HUMAN: 1, AuthorityAction.AI: 2}
        action = max(masked_feasible, key=lambda item: (float(values[item]), -tie_break[item]))
        return action, len(masked_feasible) != len(feasible)

    def _is_human_reserved(self, state: RepairState) -> bool:
        if self.human_reserve_rate <= 0.0:
            return False
        if self.human_reserve_rate >= 1.0:
            return True
        cache_key = (state.episode_id, state.step)
        if cache_key in self._reserve_cache:
            return self._reserve_cache[cache_key]
        if self.human_reserve_mode == "risk":
            return self._risk_score(state) >= self.risk_reserve_threshold
        digest = hashlib.sha256(
            f"share-joint-human-reserve:{self.reserve_seed}:{state.episode_id}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return unit < self.human_reserve_rate

    def _fit_human_reserve(self, episodes: Sequence[RepairEpisode]) -> None:
        self._reserve_cache = {}
        if self.human_reserve_rate <= 0.0 or self.human_reserve_mode == "hash":
            return
        encoder = getattr(self.policy, "encoder", None)
        if not isinstance(encoder, FeatureEncoder):
            raise TypeError("risk-directed Human reserve requires a policy FeatureEncoder")
        states = [step.state for episode in episodes for step in episode.steps]
        labels = [
            int(step.outcome(AuthorityAction.AI).unsafe or (
                step.outcome(AuthorityAction.AI).terminal and not step.outcome(AuthorityAction.AI).success
            ))
            for episode in episodes
            for step in episode.steps
        ]
        if len(set(labels)) < 2:
            self._risk_constant = float(labels[0]) if labels else 0.0
            self._risk_model = None
        else:
            self._risk_constant = None
            self._risk_model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=self.reserve_seed
            ).fit(encoder.transform(states), labels)
        state_scores = self._risk_scores(states)
        score_by_state = {
            (state.episode_id, state.step): float(score)
            for state, score in zip(states, state_scores)
        }
        episode_scores = np.asarray([
            max(score_by_state[(step.state.episode_id, step.state.step)] for step in episode.steps)
            for episode in episodes
        ])
        self._risk_score_by_state = score_by_state
        self._risk_episode_scores = episode_scores
        self._set_human_reserve_rate(self.human_reserve_rate)

    def _human_reserve_grid(self) -> tuple[float, ...]:
        if self.human_reserve_mode != "risk" or self.human_reserve_grid_size == 1:
            return (self.human_reserve_rate,)
        return tuple(float(value) for value in np.linspace(0.0, self.human_reserve_rate, self.human_reserve_grid_size))

    def _set_human_reserve_rate(self, rate: float) -> None:
        self.human_reserve_rate = rate
        self._reserve_cache = {}
        if self.human_reserve_mode != "risk" or rate <= 0.0:
            self.risk_reserve_threshold = float("inf")
            return
        self.risk_reserve_threshold = (
            float("-inf") if rate >= 1.0
            else float(np.quantile(self._risk_episode_scores, 1.0 - rate, method="higher"))
        )
        self._reserve_cache = {
            key: score >= self.risk_reserve_threshold
            for key, score in self._risk_score_by_state.items()
        }

    def _risk_score(self, state: RepairState) -> float:
        return float(self._risk_scores([state])[0])

    def _risk_scores(self, states: Sequence[RepairState]) -> np.ndarray:
        if self._risk_constant is not None:
            return np.full(len(states), self._risk_constant, dtype=np.float64)
        if self._risk_model is None:
            raise RuntimeError("risk-directed Human reserve must be fit before use")
        encoder = getattr(self.policy, "encoder")
        classes = list(self._risk_model.classes_)
        positive = classes.index(1)
        return np.asarray(self._risk_model.predict_proba(encoder.transform(states))[:, positive], dtype=np.float64)

    def _group_calibration_folds(self, episodes: Sequence[RepairEpisode]) -> tuple[tuple[RepairEpisode, ...], ...]:
        if self.calibration_folds == 1:
            return (tuple(episodes),)
        by_group: dict[str, list[RepairEpisode]] = {}
        for episode in episodes:
            by_group.setdefault(episode.group_id, []).append(episode)
        if len(by_group) < self.calibration_folds:
            raise ValueError("joint calibration has fewer groups than folds")
        ordered = sorted(
            by_group,
            key=lambda group: (hashlib.sha256(f"share-joint-fold:{self.reserve_seed}:{group}".encode()).digest(), group),
        )
        folds: list[list[RepairEpisode]] = [[] for _ in range(self.calibration_folds)]
        for index, group in enumerate(ordered):
            folds[index % self.calibration_folds].extend(by_group[group])
        return tuple(tuple(fold) for fold in folds)

    def _operating_metrics(self, episodes: Sequence[RepairEpisode], ai_penalty: float, human_penalty: float) -> dict[str, float]:
        human_episodes = unsafe_actions = stopped_episodes = decisions = 0
        total_return = 0.0
        masked_ai_actions = 0
        action_counts: Counter[int] = Counter()
        for episode in episodes:
            env = RepairAuthorityEnv(episode)
            state = env.reset()
            human_used = stopped = False
            while not env.done:
                action, masked = self._select_with_penalties(state, env.feasible_actions(), ai_penalty, human_penalty)
                result = env.step(action)
                decisions += 1
                masked_ai_actions += int(masked)
                action_counts[int(action)] += 1
                total_return += result.reward
                human_used = human_used or action == AuthorityAction.HUMAN
                stopped = stopped or action == AuthorityAction.STOP
                unsafe_actions += int(result.outcome.unsafe or (result.done and action == AuthorityAction.AI and not result.outcome.success))
                if result.next_state is not None:
                    state = result.next_state
            human_episodes += int(human_used)
            stopped_episodes += int(stopped)
        total = max(1, len(episodes))
        return {
            "hir": human_episodes / total,
            "unsafe_autonomy_cost_rate": unsafe_actions / total,
            "stopped_rate": stopped_episodes / total,
            "normalized_return": total_return / total,
            "decision_states": float(decisions),
            "unsupported_ai_mask_rate": masked_ai_actions / max(1, decisions),
            "ai_action_rate": action_counts[int(AuthorityAction.AI)] / max(1, decisions),
            "human_action_rate": action_counts[int(AuthorityAction.HUMAN)] / max(1, decisions),
            "stop_action_rate": action_counts[int(AuthorityAction.STOP)] / max(1, decisions),
        }

    def _within_budgets(self, metrics: Mapping[str, float]) -> bool:
        return (
            metrics["hir"] <= self.human_budget + 1e-12
            and metrics["unsafe_autonomy_cost_rate"] <= self.unsafe_budget + 1e-12
            and metrics["stopped_rate"] <= self.max_stopped_rate + 1e-12
        )
