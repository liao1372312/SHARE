from __future__ import annotations

from scripts.run_share_final_ablation import ablation_config, holm_adjust


def test_myopic_changes_only_gamma_relative_to_no_risk_training_config() -> None:
    myopic = ablation_config("share_myopic", seed=17, epochs=30, device="cpu")
    no_risk = ablation_config("share_wo_risk", seed=17, epochs=30, device="cpu")
    assert myopic.gamma == 0.0
    assert no_risk.gamma == 0.95
    left = vars(myopic).copy()
    right = vars(no_risk).copy()
    left.pop("gamma")
    right.pop("gamma")
    assert left == right


def test_trainable_ablations_retain_human_only_constraint() -> None:
    for name in ("share_myopic", "share_wo_risk"):
        config = ablation_config(name, seed=29, epochs=30, device="cpu")
        assert config.constrained
        assert config.human_constraint
        assert not config.unsafe_constraint
        assert config.human_budget == 0.20


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.05, 0.01, 0.2, 0.03]
    adjusted = holm_adjust(raw)
    ordered = sorted(range(len(raw)), key=raw.__getitem__)
    assert all(adjusted[left] <= adjusted[right] for left, right in zip(ordered, ordered[1:]))
    assert all(adjusted[index] >= raw[index] for index in range(len(raw)))
