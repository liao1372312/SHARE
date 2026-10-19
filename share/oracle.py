from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .environment import base_reward
from .types import AuthorityAction, RepairEpisode, RewardConfig


@dataclass(frozen=True)
class TrajectoryOption:
    actions: Tuple[AuthorityAction, ...]
    discounted_return: float
    success: bool
    unsafe_actions: int
    human_used: bool
    human_minutes: float

    @property
    def unsafe(self) -> bool:
        return self.unsafe_actions > 0


@dataclass(frozen=True)
class OracleResult:
    episodes: int
    human_limit: int
    unsafe_limit: int
    total_return: float
    repair_sr: float
    hir: float
    unsafe_cost_rate: float
    uar: float
    stopped_rate: float
    selections: Tuple[TrajectoryOption, ...]


@dataclass(frozen=True)
class ExpectedOracleResult:
    episodes: int
    human_limit: float
    unsafe_limit: float
    expected_return: float
    expected_success_rate: float
    hir: float
    expected_unsafe_rate: float
    stopped_rate: float


def enumerate_trajectory_options(episode: RepairEpisode, config: RewardConfig | None = None, gamma: float = 0.95) -> List[TrajectoryOption]:
    reward_config = config or RewardConfig()
    options: List[TrajectoryOption] = []

    def visit(step_index: int, actions: Tuple[AuthorityAction, ...], rewards: Tuple[float, ...], unsafe_actions: int) -> None:
        step = episode.steps[step_index]
        for action in AuthorityAction:
            outcome = step.outcome(action)
            next_actions = actions + (action,)
            next_rewards = rewards + (base_reward(outcome, action, reward_config),)
            next_unsafe_actions = unsafe_actions + int(outcome.unsafe or (action == AuthorityAction.AI and outcome.terminal and not outcome.success))
            if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None:
                visit(outcome.next_step, next_actions, next_rewards, next_unsafe_actions)
                continue
            discounted = sum((gamma**index) * value for index, value in enumerate(next_rewards))
            options.append(
                TrajectoryOption(
                    actions=next_actions,
                    discounted_return=discounted,
                    success=outcome.success,
                    unsafe_actions=next_unsafe_actions,
                    human_used=AuthorityAction.HUMAN in next_actions,
                    human_minutes=outcome.human_minutes if action == AuthorityAction.HUMAN else 0.0,
                )
            )

    visit(0, (), (), False)
    unique = {}
    for option in options:
        key = (option.actions, option.success, option.unsafe_actions, option.human_used)
        if key not in unique or option.discounted_return > unique[key].discounted_return:
            unique[key] = option
    return list(unique.values())


def solve_constrained_oracle(episodes: Sequence[RepairEpisode], human_budget: float, unsafe_budget: float, config: RewardConfig | None = None, gamma: float = 0.95) -> OracleResult:
    if not 0 <= human_budget <= 1 or not 0 <= unsafe_budget <= 1:
        raise ValueError("budgets must be in [0, 1]")
    count = len(episodes)
    if count == 0:
        raise ValueError("oracle requires at least one episode")
    human_limit = int(np.floor(human_budget * count + 1e-12))
    unsafe_limit = int(np.floor(unsafe_budget * count + 1e-12))
    option_sets = [enumerate_trajectory_options(episode, config, gamma) for episode in episodes]
    neg_inf = -np.inf
    dp = np.full((human_limit + 1, unsafe_limit + 1), neg_inf, dtype=np.float64)
    dp[0, 0] = 0.0
    backpointers: List[np.ndarray] = []
    previous_costs: List[np.ndarray] = []
    for options in option_sets:
        next_dp = np.full_like(dp, neg_inf)
        chosen = np.full(dp.shape, -1, dtype=np.int16)
        previous = np.full(dp.shape + (2,), -1, dtype=np.int16)
        for human_used in range(human_limit + 1):
            for unsafe_used in range(unsafe_limit + 1):
                current = dp[human_used, unsafe_used]
                if not np.isfinite(current):
                    continue
                for option_index, option in enumerate(options):
                    next_human = human_used + int(option.human_used)
                    next_unsafe = unsafe_used + option.unsafe_actions
                    if next_human > human_limit or next_unsafe > unsafe_limit:
                        continue
                    value = current + option.discounted_return
                    if value > next_dp[next_human, next_unsafe]:
                        next_dp[next_human, next_unsafe] = value
                        chosen[next_human, next_unsafe] = option_index
                        previous[next_human, next_unsafe] = (human_used, unsafe_used)
        if not np.isfinite(next_dp).any():
            raise RuntimeError("no feasible trajectory combination; STOP should make this impossible")
        dp = next_dp
        backpointers.append(chosen)
        previous_costs.append(previous)
    final_human, final_unsafe = np.unravel_index(np.nanargmax(dp), dp.shape)
    selections: List[TrajectoryOption] = []
    human_cursor, unsafe_cursor = int(final_human), int(final_unsafe)
    for episode_index in range(count - 1, -1, -1):
        option_index = int(backpointers[episode_index][human_cursor, unsafe_cursor])
        if option_index < 0:
            raise RuntimeError("oracle backpointer is incomplete")
        selections.append(option_sets[episode_index][option_index])
        prior = previous_costs[episode_index][human_cursor, unsafe_cursor]
        human_cursor, unsafe_cursor = int(prior[0]), int(prior[1])
    selections.reverse()
    autonomous = [item for item in selections if not item.human_used and item.actions[-1] != AuthorityAction.STOP]
    return OracleResult(
        episodes=count,
        human_limit=human_limit,
        unsafe_limit=unsafe_limit,
        total_return=float(max(dp.flat)),
        repair_sr=float(np.mean([item.success for item in selections])),
        hir=float(np.mean([item.human_used for item in selections])),
        unsafe_cost_rate=float(sum(item.unsafe_actions for item in selections) / count),
        uar=float(np.mean([item.unsafe or not item.success for item in autonomous])) if autonomous else 0.0,
        stopped_rate=float(np.mean([item.actions[-1] == AuthorityAction.STOP for item in selections])),
        selections=tuple(selections),
    )


def solve_expected_root_oracle(
    episodes: Sequence[RepairEpisode],
    human_budget: float,
    unsafe_budget: float,
    config: RewardConfig | None = None,
    max_stopped_rate: float | None = None,
) -> ExpectedOracleResult:
    """Non-clairvoyant root-action LP over recorded branch probabilities.

    The optional Stop bound makes this a useful feasibility diagnostic for a
    deployment policy that is not allowed to satisfy safety by stopping every
    workflow. It remains an upper bound because synthetic paired outcomes are
    available to the diagnostic.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    reward_config = config or RewardConfig()
    count = len(episodes)
    if count == 0:
        raise ValueError("expected oracle requires episodes")
    if not 0.0 <= human_budget <= 1.0 or not 0.0 <= unsafe_budget <= 1.0:
        raise ValueError("human and unsafe budgets must be in [0, 1]")
    if max_stopped_rate is not None and not 0.0 <= max_stopped_rate <= 1.0:
        raise ValueError("max_stopped_rate must be in [0, 1] when supplied")
    actions = list(AuthorityAction)
    rewards = []
    successes = []
    human_costs = []
    unsafe_costs = []
    stop_costs = []
    for episode in episodes:
        step = episode.steps[0]
        for action in actions:
            outcome = step.outcome(action)
            probability = float(outcome.metadata.get("probability", float(outcome.success)))
            expected_quality = 0.8 * probability - 0.35 * (1.0 - probability) if action == AuthorityAction.AI else (0.9 * probability - 0.15 * (1.0 - probability) if action == AuthorityAction.HUMAN else -0.1)
            expected = expected_quality - reward_config.latency_weight * outcome.latency - reward_config.execution_weight * outcome.execution_cost
            expected -= reward_config.ineffective_weight * (1.0 - probability)
            expected += reward_config.success_reward * probability
            risk_probability = float(outcome.metadata.get("unsafe_probability", 0.0))
            if action == AuthorityAction.AI:
                risk_probability = max(risk_probability, 1.0 - probability)
                expected -= reward_config.unsafe_penalty * risk_probability
            if action == AuthorityAction.STOP:
                expected -= reward_config.rollback_penalty
            rewards.append(expected)
            successes.append(probability)
            human_costs.append(float(action == AuthorityAction.HUMAN))
            unsafe_costs.append(risk_probability if action == AuthorityAction.AI else 0.0)
            stop_costs.append(float(action == AuthorityAction.STOP))
    variables = count * len(actions)
    a_eq = csr_matrix(
        (
            np.ones(variables),
            (np.repeat(np.arange(count), len(actions)), np.arange(variables)),
        ),
        shape=(count, variables),
    )
    constraints = [human_costs, unsafe_costs]
    limits = [human_budget * count, unsafe_budget * count]
    if max_stopped_rate is not None:
        constraints.append(stop_costs)
        limits.append(max_stopped_rate * count)
    result = linprog(
        c=-np.asarray(rewards),
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(limits),
        A_eq=a_eq,
        b_eq=np.ones(count),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"expected oracle LP failed: {result.message}")
    weights = result.x
    return ExpectedOracleResult(
        episodes=count,
        human_limit=human_budget * count,
        unsafe_limit=unsafe_budget * count,
        expected_return=float(np.dot(weights, rewards) / count),
        expected_success_rate=float(np.dot(weights, successes) / count),
        hir=float(np.dot(weights, human_costs) / count),
        expected_unsafe_rate=float(np.dot(weights, unsafe_costs) / count),
        stopped_rate=float(np.dot(weights, stop_costs) / count),
    )
