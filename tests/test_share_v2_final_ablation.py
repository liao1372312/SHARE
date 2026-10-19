from dataclasses import asdict

from scripts.run_share_v2_development_multistart import config_for
from scripts.run_share_v2_final_ablation import ABLATIONS, training_config, training_reward


def differences(variant: str) -> set[str]:
    base = asdict(config_for(17, "paired_e30", "cpu"))
    changed = asdict(training_config(variant, 17, "cpu"))
    return {key for key in base if base[key] != changed[key]}


def test_v2_ablation_config_changes_are_narrow() -> None:
    assert differences("share_v2_myopic") == {"gamma"}
    assert differences("share_v2_wo_risk") == set()
    assert differences("share_v2_wo_budget") == {"constrained", "human_constraint"}
    assert differences("share_v2_wo_paired_replay") == {"paired_replay"}
    assert differences("share_v2_rollback_025") == set()


def test_only_rollback_ablation_changes_training_reward() -> None:
    for variant in ABLATIONS:
        expected = 0.25 if variant == "share_v2_rollback_025" else 0.60
        assert training_reward(variant).rollback_penalty == expected
