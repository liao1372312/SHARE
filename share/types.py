from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Sequence


class AuthorityAction(IntEnum):
    AI = 0
    HUMAN = 1
    STOP = 2


@dataclass(frozen=True)
class RepairState:
    episode_id: str
    step: int
    gap_type: str
    root_type: str
    localization_score: float
    repair_utility: float
    affected_ratio: float
    downstream_impact: float
    drift_risk: float
    previous_failures: float
    uncertainty: float
    evidence_conflict: float
    remaining_budget: float
    complexity: float = 0.0
    domain: str = "unknown"

    def continuous(self) -> List[float]:
        return [
            self.localization_score,
            self.repair_utility,
            self.affected_ratio,
            self.downstream_impact,
            self.drift_risk,
            self.previous_failures,
            self.uncertainty,
            self.evidence_conflict,
            self.remaining_budget,
            self.complexity,
        ]


@dataclass(frozen=True)
class BranchOutcome:
    success: bool
    unsafe: bool
    quality_delta: float
    latency: float
    execution_cost: float
    human_minutes: float = 0.0
    next_step: int | None = None
    terminal: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedRepairStep:
    state: RepairState
    outcomes: Mapping[str, BranchOutcome]
    human_beneficial: bool

    def outcome(self, action: AuthorityAction) -> BranchOutcome:
        return self.outcomes[action.name.lower()]


@dataclass(frozen=True)
class RepairEpisode:
    episode_id: str
    group_id: str
    split: str
    source: str
    synthetic: bool
    steps: Sequence[PairedRepairStep]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepairEpisode":
        steps = []
        for raw_step in value["steps"]:
            state = RepairState(**raw_step["state"])
            outcomes = {key: BranchOutcome(**outcome) for key, outcome in raw_step["outcomes"].items()}
            steps.append(PairedRepairStep(state, outcomes, bool(raw_step["human_beneficial"])))
        return cls(
            episode_id=str(value["episode_id"]),
            group_id=str(value["group_id"]),
            split=str(value["split"]),
            source=str(value["source"]),
            synthetic=bool(value["synthetic"]),
            steps=steps,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 1.0
    latency_weight: float = 0.1
    execution_weight: float = 0.1
    ineffective_weight: float = 0.25
    success_reward: float = 1.0
    unsafe_penalty: float = 2.0
    rollback_penalty: float = 0.25

    def __post_init__(self) -> None:
        if not self.unsafe_penalty > self.rollback_penalty > 0:
            raise ValueError("unsafe_penalty must be greater than rollback_penalty and both must be positive")


@dataclass
class EpisodeRecord:
    episode_id: str
    actions: List[int]
    rewards: List[float]
    success: bool
    unsafe: bool
    unsafe_actions: int
    human_used: bool
    human_minutes: float
    latency: float
    execution_cost: float
    human_labels: List[int]
    escalation_scores: List[float | None]
    selected_returns: List[float]
    oracle_returns: List[float]
