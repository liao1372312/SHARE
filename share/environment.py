from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .types import AuthorityAction, BranchOutcome, RepairEpisode, RepairState, RewardConfig


@dataclass
class StepResult:
    state: RepairState
    action: AuthorityAction
    outcome: BranchOutcome
    reward: float
    next_state: RepairState | None
    done: bool


class RepairAuthorityEnv:
    def __init__(self, episode: RepairEpisode, reward_config: RewardConfig | None = None):
        if not episode.steps:
            raise ValueError("episode must contain at least one step")
        self.episode = episode
        self.reward_config = reward_config or RewardConfig()
        self._step = 0
        self.done = False

    def reset(self) -> RepairState:
        self._step = 0
        self.done = False
        return self.episode.steps[0].state

    def feasible_actions(self) -> Tuple[AuthorityAction, ...]:
        state = self.episode.steps[self._step].state
        actions = [AuthorityAction.HUMAN, AuthorityAction.STOP]
        if state.remaining_budget > 0:
            actions.insert(0, AuthorityAction.AI)
        return tuple(actions)

    def step(self, action: AuthorityAction) -> StepResult:
        if self.done:
            raise RuntimeError("episode already terminated")
        if action not in self.feasible_actions():
            raise ValueError(f"infeasible action {action.name} at step {self._step}")
        paired_step = self.episode.steps[self._step]
        outcome = paired_step.outcome(action)
        reward = base_reward(outcome, action, self.reward_config)
        next_state = None
        if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None:
            if not 0 <= outcome.next_step < len(self.episode.steps):
                raise ValueError("branch outcome points outside the episode")
            self._step = outcome.next_step
            next_state = self.episode.steps[self._step].state
        else:
            self.done = True
        return StepResult(paired_step.state, action, outcome, reward, next_state, self.done)


def base_reward(outcome: BranchOutcome, action: AuthorityAction, config: RewardConfig) -> float:
    value = config.quality_weight * outcome.quality_delta
    value -= config.latency_weight * outcome.latency
    value -= config.execution_weight * outcome.execution_cost
    if not outcome.success:
        value -= config.ineffective_weight
    if outcome.success:
        value += config.success_reward
    if outcome.unsafe:
        value -= config.unsafe_penalty
    if action == AuthorityAction.STOP:
        value -= config.rollback_penalty
    return float(value)


def branch_rewards(episode: RepairEpisode, step: int, config: RewardConfig) -> Dict[AuthorityAction, float]:
    paired = episode.steps[step]
    return {action: base_reward(paired.outcome(action), action, config) for action in AuthorityAction}

