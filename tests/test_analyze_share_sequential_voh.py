from __future__ import annotations

from scripts.analyze_share_sequential_voh import complexity_bin, exact_sign_flip, forced_ai_return
from share.environment import base_reward
from share.types import AuthorityAction, BranchOutcome, PairedRepairStep, RepairEpisode, RepairState, RewardConfig


def _outcome(success: bool, terminal: bool = True, next_step: int | None = None) -> BranchOutcome:
    return BranchOutcome(
        success=success, unsafe=not success, quality_delta=0.5 if success else -0.5,
        latency=1.0, execution_cost=0.2, terminal=terminal, next_step=next_step,
    )


def _state(step: int) -> RepairState:
    return RepairState(
        episode_id="e", step=step, gap_type="g", root_type="r", localization_score=0.5,
        repair_utility=0.5, affected_ratio=0.5, downstream_impact=0.5, drift_risk=0.5,
        previous_failures=0.0, uncertainty=0.5, evidence_conflict=0.5, remaining_budget=1.0,
    )


def test_preregistered_complexity_boundaries() -> None:
    assert complexity_bin(0.49) == "C_low"
    assert complexity_bin(0.50) == "C_mid"
    assert complexity_bin(0.79) == "C_mid"
    assert complexity_bin(0.80) == "C_high"


def test_forced_ai_return_recursively_continues_ai() -> None:
    first_ai = _outcome(False, terminal=False, next_step=1)
    second_ai = _outcome(True)
    human = _outcome(True)
    stop = _outcome(False)
    episode = RepairEpisode(
        episode_id="e", group_id="g", split="test", source="unit", synthetic=True,
        steps=[
            PairedRepairStep(_state(0), {"ai": first_ai, "human": human, "stop": stop}, True),
            PairedRepairStep(_state(1), {"ai": second_ai, "human": human, "stop": stop}, False),
        ],
    )
    reward = RewardConfig()
    value, success, unsafe = forced_ai_return(episode, 0, reward, gamma=0.95)
    expected = base_reward(first_ai, AuthorityAction.AI, reward) + 0.95 * base_reward(second_ai, AuthorityAction.AI, reward)
    assert value == expected
    assert success is True
    assert unsafe is True


def test_exact_sign_flip_five_positive_seeds_has_minimum_one_sided_p() -> None:
    assert exact_sign_flip([1, 1, 1, 1, 1], alternative="greater") == 1 / 32
