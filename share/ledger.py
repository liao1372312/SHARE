"""Online intervention accounting shared by paper evaluation and legacy helpers."""
from __future__ import annotations

from typing import Any, Sequence

from .policies import Policy
from .types import AuthorityAction, RepairState


class OnlineHardHumanCapPolicy(Policy):
    """Count selected Human episodes without inspecting branch outcomes."""

    def __init__(self, base: Policy, cap: int, binary_fallback: bool = False):
        if cap < 0:
            raise ValueError("the Human cap must be non-negative")
        self.base = base
        self.name = base.name
        self.cap = int(cap)
        self.binary_fallback = binary_fallback
        self.human_episode_ids: set[str] = set()
        self.attempted_human_after_exhaustion = 0
        self.forced_autonomous_fallbacks = 0
        self.value_policy_replans = 0
        self.events: list[dict[str, Any]] = []

    @property
    def human_interventions(self) -> int:
        return len(self.human_episode_ids)

    def escalation_score(self, state: RepairState) -> float:
        return self.base.escalation_score(state)

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        proposed = self.base.select(state, feasible)
        before = self.human_interventions
        executed = proposed
        fallback = None
        if proposed == AuthorityAction.HUMAN and before >= self.cap:
            self.attempted_human_after_exhaustion += 1
            remaining = [action for action in feasible if action != AuthorityAction.HUMAN]
            if not remaining:
                raise ValueError("no action remains after applying the Human cap")
            if self.binary_fallback:
                executed = AuthorityAction.AI if AuthorityAction.AI in remaining else AuthorityAction.STOP
                self.forced_autonomous_fallbacks += 1
                fallback = "binary_autonomous"
            else:
                executed = self.base.select(state, remaining)
                self.value_policy_replans += 1
                fallback = "value_best_non_human"
        if executed not in feasible:
            raise ValueError("policy selected an infeasible action")
        if executed == AuthorityAction.HUMAN:
            self.human_episode_ids.add(state.episode_id)
        if self.human_interventions > self.cap:
            raise RuntimeError("hard Human cap violated")
        self.events.append({
            "episode_id": state.episode_id, "step": state.step,
            "proposed_action": int(proposed), "executed_action": int(executed),
            "budget_before": before, "budget_after": self.human_interventions,
            "exhausted_before_decision": before >= self.cap, "fallback": fallback,
        })
        return executed
