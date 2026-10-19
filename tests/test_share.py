from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from share.data import build_synthetic_episodes, label_human_benefit, load_episodes, validate_splits, write_episodes
from share.dqn import DQNConfig, DQNEnsemblePolicy, DQNPolicy, feasible_mask, masked_conservative_argmax
from share.deployment import canonical_digest, load_deployment_events, sha256_file
from share.evidence_store import ValidatedCorrection, ValidatedEvidenceStore, correction_digest, parse_timestamp
from share.environment import RepairAuthorityEnv, base_reward
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import AlwaysHumanPolicy, AutonomousPolicy, RandomBudgetPolicy
from share.robustness import corrupt_human_feedback
from share.safety import certify_unsafe_rate, minimum_observations_for_zero_unsafe
from share.safety import certify_policy_costs, maximum_certifiable_unsafe_actions
from share.safety import certify_group_mean, minimum_groups_for_zero_mean
from share.protocol import guard_development_partitions, guard_partitions
from share.replay import collect_frozen_logged_replay
from share.selection import StableSelectionConfig, group_folds, select_stable_candidate
from share.statistics import cluster_bootstrap_ci, hierarchical_bootstrap_ci, holm_adjust, paired_differences, paired_permutation_test
from share.support import JointCalibratedPolicy, SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy, mask_unsupported_ai_next_actions, support_key
from share.vector_dqn import VectorDQNConfig, VectorDQNPolicy
from share.oracle import enumerate_trajectory_options, solve_constrained_oracle, solve_expected_root_oracle
from share.types import AuthorityAction, BranchOutcome, PairedRepairStep, RepairEpisode, RepairState, RewardConfig
from scripts.run_share_guard_experiments import compatible_protocol_signature, load_historical_identifier_index, paired_comparison, validate_locked_holdout
from scripts.run_share_controller_rescue import dqn_config as rescue_dqn_config
from scripts.summarize_share_guard_selection import MAX_VIOLATION_RATIO, constraint_ratio, summarize_runs
from scripts.audit_share_guard_support import summarize_support_coverage
from scripts.import_share_real_paired_episodes import load_artifact_ledger, outcome_sha256, validate_real_paired_episodes
from share.llm_profiles import load_llm_profiles, public_profile_report


def sample_episode(split="train", group="g1", episode_id="e1"):
    state0 = RepairState(episode_id, 0, "parameter", "parameter", 0.8, 0.8, 0.2, 0.3, 0.2, 0.0, 0.2, 0.0, 1.0, 0.4, "finance")
    state1 = RepairState(episode_id, 1, "parameter", "parameter", 0.7, 0.6, 0.2, 0.3, 0.3, 0.3, 0.3, 0.0, 0.5, 0.4, "finance")
    stop = BranchOutcome(False, False, -0.1, 0.1, 0.0)
    steps = [
        PairedRepairStep(state0, {
            "ai": BranchOutcome(False, False, -0.2, 0.2, 0.2, next_step=1, terminal=False),
            "human": BranchOutcome(True, False, 0.9, 0.5, 0.2, human_minutes=5),
            "stop": stop,
        }, True),
        PairedRepairStep(state1, {
            "ai": BranchOutcome(True, False, 0.8, 0.2, 0.2),
            "human": BranchOutcome(True, False, 0.9, 0.5, 0.2, human_minutes=5),
            "stop": stop,
        }, False),
    ]
    return RepairEpisode(episode_id, group, split, "test", True, steps)


def real_paired_episode(split="test", group="real-group", episode_id="real-episode"):
    base = sample_episode(split=split, group=group, episode_id=episode_id)
    digest = "a" * 64
    provenance = {
        "collection_id": "externally-attested-collection-1",
        "ai": {"execution_artifact_sha256": digest, "model_id": "model-a"},
        "human": {"execution_artifact_sha256": "b" * 64},
        "stop": {"execution_artifact_sha256": "c" * 64},
    }
    return replace(base, synthetic=False, source="externally-collected-paired-outcomes", metadata={"paired_outcome_provenance": provenance})


def bind_paired_artifacts(tmp_path, episodes):
    artifact_root = tmp_path / "paired_artifacts"
    ledger_path = tmp_path / "paired_artifact_ledger.jsonl"
    artifact_root.mkdir()
    ledger_entries = []
    bound_episodes = []
    collection_id = "externally-attested-collection-1"
    for episode in episodes:
        episode = replace(
            episode,
            steps=label_human_benefit(episode.steps, RewardConfig(), margin=0.1, gamma=0.95),
        )
        provenance = {
            "collection_id": collection_id,
            "human_benefit_label_rule": "share_label_human_benefit_v1",
            "ai": {"model_id": "model-a", "execution_artifact_sha256_by_step": {}},
            "human": {"execution_artifact_sha256_by_step": {}},
            "stop": {"execution_artifact_sha256_by_step": {}},
        }
        for step_index, step in enumerate(episode.steps):
            for branch in ("ai", "human", "stop"):
                outcome = asdict(step.outcomes[branch])
                digest_outcome = outcome_sha256(step.outcomes[branch])
                artifact = {
                    "schema_version": 1,
                    "kind": "share_paired_outcome_branch_artifact",
                    "collection_id": collection_id,
                    "episode_id": episode.episode_id,
                    "step": step_index,
                    "branch": branch,
                    "model_id": "model-a" if branch == "ai" else None,
                    "outcome": outcome,
                    "outcome_sha256": digest_outcome,
                }
                artifact_path = artifact_root / f"{episode.episode_id}-{step_index}-{branch}.json"
                artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
                digest = sha256_file(artifact_path)
                entry = {
                    "schema_version": 1,
                    "kind": "share_paired_outcome_artifact_ledger_entry",
                    "collection_id": collection_id,
                    "episode_id": episode.episode_id,
                    "step": step_index,
                    "branch": branch,
                    "execution_artifact_file": artifact_path.name,
                    "execution_artifact_sha256": digest,
                    "outcome_sha256": digest_outcome,
                }
                if branch == "ai":
                    entry["model_id"] = "model-a"
                provenance[branch]["execution_artifact_sha256_by_step"][str(step_index)] = digest
                ledger_entries.append(entry)
        bound_episodes.append(replace(episode, metadata={"paired_outcome_provenance": provenance}))
    ledger_path.write_text("".join(json.dumps(entry) + "\n" for entry in ledger_entries), encoding="utf-8")
    return bound_episodes, ledger_path, artifact_root


def test_episode_round_trip_and_group_leakage(tmp_path):
    path = tmp_path / "episodes.jsonl"
    episode = sample_episode()
    write_episodes(path, [episode])
    loaded = load_episodes(path)
    assert loaded[0] == episode
    with pytest.raises(ValueError, match="group leakage"):
        validate_splits([episode, sample_episode("test", "g1", "e2")])


def test_environment_is_sequential_and_masks_exhausted_ai():
    env = RepairAuthorityEnv(sample_episode())
    assert env.reset().step == 0
    result = env.step(AuthorityAction.AI)
    assert not result.done and result.next_state.step == 1
    result = env.step(AuthorityAction.AI)
    assert result.done and result.outcome.success
    exhausted = sample_episode().steps[-1].state
    exhausted = RepairState(**{**exhausted.__dict__, "remaining_budget": 0.0})
    assert not feasible_mask(exhausted)[AuthorityAction.AI]


def test_random_budget_decision_is_fixed_per_episode():
    policy = RandomBudgetPolicy(0.5, seed=3)
    state0, state1 = sample_episode().steps[0].state, sample_episode().steps[1].state
    first = policy.select(state0, [AuthorityAction.AI, AuthorityAction.HUMAN])
    second = policy.select(state1, [AuthorityAction.AI, AuthorityAction.HUMAN])
    assert first == second


def test_reward_prefers_safe_rollback_to_unsafe_completion():
    config = RewardConfig()
    unsafe = base_reward(BranchOutcome(False, True, -0.1, 0, 0), AuthorityAction.AI, config)
    rollback = base_reward(BranchOutcome(False, False, -0.1, 0, 0), AuthorityAction.STOP, config)
    assert rollback > unsafe


def test_human_benefit_uses_autonomous_return_to_go():
    episode = sample_episode()
    labeled = label_human_benefit(episode.steps, RewardConfig(), margin=0.7)
    immediate_ai = base_reward(episode.steps[0].outcomes["ai"], AuthorityAction.AI, RewardConfig())
    immediate_human = base_reward(episode.steps[0].outcomes["human"], AuthorityAction.HUMAN, RewardConfig())
    assert immediate_human > immediate_ai
    assert labeled[0].human_beneficial is False  # AI recovers on its next transition.


def test_feature_encoder_unknown_categories_and_dqn_training():
    episodes = [sample_episode(episode_id=f"e{i}", group=f"g{i}") for i in range(8)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    unknown = RepairState("x", 0, "new", "new", 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, "new")
    assert encoder.transform_one(unknown).shape == (encoder.dimension,)
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=2, batch_size=4, epsilon_decay_steps=5, seed=3))
    before = [parameter.detach().clone() for parameter in policy.online.parameters()]
    policy.fit(episodes)
    assert any(not np.array_equal(old.numpy(), new.detach().numpy()) for old, new in zip(before, policy.online.parameters()))
    assert np.isfinite(policy.escalation_score(episodes[0].steps[0].state))


def test_dqn_records_epoch_action_q_and_dual_diagnostics():
    episodes = [sample_episode(episode_id=f"diag-{i}", group=f"diag-{i}") for i in range(6)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    config = rescue_dqn_config(
        "share_h", seed=17, epochs=2, human_budget=0.2, unsafe_budget=0.05, device="cpu"
    )
    policy = DQNPolicy(encoder, replace(config, batch_size=4, epsilon_decay_steps=5)).fit(
        episodes, episodes
    )
    assert len(policy.training_history) == 2
    assert policy.beta_u == 0.0
    assert policy.selected_checkpoint_diagnostics is not None
    for epoch, row in enumerate(policy.training_history, 1):
        assert row["epoch"] == epoch
        assert row["phase"] == "epoch_end"
        assert row["p_ai"] + row["p_human"] + row["p_stop"] == pytest.approx(1.0)
        assert all(np.isfinite(row[key]) for key in ("q_ai_raw_mean", "q_human_raw_mean", "q_stop_raw_mean", "beta_h", "beta_u"))


def test_paired_replay_contains_every_training_state_action():
    episode = sample_episode()
    encoder = FeatureEncoder.fit([step.state for step in episode.steps])
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=0, batch_size=2, seed=6, paired_replay=True))
    policy._populate_paired_replay([episode])
    assert len(policy.replay) == 6
    assert {transition[1] for transition in policy.replay} == {0, 1, 2}


def test_frozen_logged_replay_is_candidate_independent_and_support_is_unique():
    episodes = [sample_episode(episode_id=f"logged-{index}", group=f"logged-{index}") for index in range(4)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    first = collect_frozen_logged_replay(episodes, encoder, behavior_seed=23)
    second = collect_frozen_logged_replay(episodes, encoder, behavior_seed=23)
    assert first.content_sha256 == second.content_sha256
    assert first.support_counts == second.support_counts
    assert len(first.transitions) == len(first.next_support_keys)
    scalar = DQNPolicy(encoder, DQNConfig(constrained=False, training_epochs=0, batch_size=2, fixed_human_dual=1.0, fixed_unsafe_dual=2.0, seed=4)).fit_replay(first, updates=2)
    vector = VectorDQNPolicy(encoder, VectorDQNConfig(DQNConfig(training_epochs=0, batch_size=2, seed=4), 1.0, 2.0, 0.95, 0.95)).fit_replay(first, updates=2)
    assert scalar.logged_action_support == vector.logged_action_support == first.support_counts


def test_predecision_support_mask_keeps_human_available_and_masks_bellman_ai():
    state = sample_episode().steps[0].state
    class HumanOverStopPolicy:
        name = "human_over_stop"

        def select(self, _state, feasible):
            return AuthorityAction.HUMAN if AuthorityAction.HUMAN in feasible else AuthorityAction.STOP

        def escalation_score(self, _state):
            return 0.0

    policy = SupportMaskedPolicy(HumanOverStopPolicy(), {}, minimum_count=3)
    assert policy.select(state, [AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP]) == AuthorityAction.HUMAN
    assert policy.masked_ai_actions == 1
    mask = np.asarray([[True, True, True], [True, True, True]], dtype=bool)
    masked = mask_unsupported_ai_next_actions(mask, [support_key(state), None], {}, minimum_count=3)
    assert not masked[0, AuthorityAction.AI]
    assert masked[0, AuthorityAction.HUMAN]
    assert masked[1, AuthorityAction.AI]


def test_calibrated_support_router_only_routes_unsupported_ai_from_fitting_data():
    fitting = sample_episode(episode_id="router-fitting", group="router-fitting")
    selection = sample_episode(episode_id="router-selection", group="router-selection")
    encoder = FeatureEncoder.fit([step.state for step in fitting.steps])
    router = SupportRoutedPolicy(
        AutonomousPolicy(),
        {},
        encoder,
        human_budget=1.0,
        unsafe_budget=1.0,
        threshold_grid=3,
        seed=17,
    ).fit([fitting])
    assert router.calibration["fitting_episodes"] == 1
    assert router.calibration["fitting_workflow_groups"] == 1
    first = router.select(selection.steps[0].state, list(AuthorityAction))
    assert first in {AuthorityAction.HUMAN, AuthorityAction.STOP}
    assert router.routed_ai_actions == 1
    assert router.routed_human_actions + router.routed_stop_actions == 1

    supported = SupportRoutedPolicy(
        AutonomousPolicy(),
        {(support_key(fitting.steps[0].state), int(AuthorityAction.AI)): 3},
        encoder,
        human_budget=1.0,
        unsafe_budget=1.0,
    ).fit([fitting])
    assert supported.select(fitting.steps[0].state, list(AuthorityAction)) == AuthorityAction.AI
    assert supported.routed_ai_actions == 0


def test_dqn_config_accepts_execution_only_calibrated_router_mode():
    assert DQNConfig(support_mode="calibrated_router").support_mode == "calibrated_router"


def test_joint_calibration_uses_fitting_budget_and_masks_unsupported_ai():
    fitting = sample_episode(episode_id="joint-fitting", group="joint-fitting")

    class FixedValuePolicy:
        name = "fixed_values"

        def values(self, _state):
            return np.asarray([1.0, 0.7, 0.0])

        def select(self, _state, feasible):
            return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.HUMAN

        def escalation_score(self, _state):
            return 0.0

    policy = JointCalibratedPolicy(
        FixedValuePolicy(), {}, human_budget=1.0, unsafe_budget=1.0, max_stopped_rate=1.0, grid_size=3
    ).fit([fitting])
    assert policy.calibration["fitting_episodes"] == 1
    assert policy.calibration["feasible"] is True
    assert policy.select(fitting.steps[0].state, list(AuthorityAction)) == AuthorityAction.HUMAN
    assert policy.masked_ai_actions == 1
    assert policy.joint_action_counts[int(AuthorityAction.HUMAN)] == 1


def test_joint_selection_rejects_stop_collapse_and_keeps_legacy_positional_order():
    configuration = StableSelectionConfig(0.0, 0.0, 3, 17, max_stopped_rate=0.5)
    assert configuration.folds == 3 and configuration.seed == 17
    episodes = [sample_episode(episode_id=f"stop-{index}", group=f"stop-{index}") for index in range(3)]

    class StopPolicy:
        name = "stop"

        def select(self, _state, _feasible):
            return AuthorityAction.STOP

        def escalation_score(self, _state):
            return 0.0

    result = select_stable_candidate({"stop": StopPolicy()}, episodes, configuration)
    assert result.selected is None
    assert not result.candidates[0].stable


def test_joint_penalty_grid_keeps_pairwise_breakpoints_for_latent_actions():
    episode = sample_episode(episode_id="pairwise-grid", group="pairwise-grid")

    class PairwiseValuePolicy:
        name = "pairwise-values"

        def values(self, _state):
            # Human is not initially the best action, but it can become best
            # when AI is penalized.  Its gap over Stop must therefore survive
            # construction of the joint calibration grid.
            return np.asarray([3.0, 2.0, 0.0])

        def select(self, _state, _feasible):
            return AuthorityAction.AI

    support = {
        (support_key(step.state), int(AuthorityAction.AI)): 3
        for step in episode.steps
    }
    policy = JointCalibratedPolicy(
        PairwiseValuePolicy(), support, human_budget=1.0,
        unsafe_budget=1.0, max_stopped_rate=1.0, grid_size=5,
    )
    policy._value_cache = {
        (step.state.episode_id, step.state.step): PairwiseValuePolicy().values(step.state)
        for step in episode.steps
    }
    ai_penalties, human_penalties = policy._penalty_grid([episode])
    assert 1.0 in ai_penalties
    assert 3.0 in ai_penalties
    assert 2.0 in human_penalties


def test_joint_human_reserve_is_deterministic_per_episode():
    episode = sample_episode(episode_id="reserved", group="reserved")

    class AIValuePolicy:
        name = "ai-values"

        def values(self, _state):
            return np.asarray([3.0, 1.0, 0.0])

        def select(self, _state, _feasible):
            return AuthorityAction.AI

    policy = JointCalibratedPolicy(
        AIValuePolicy(), {}, human_budget=1.0, unsafe_budget=1.0,
        max_stopped_rate=1.0, human_reserve_rate=1.0,
    )
    actions = [policy.select(step.state, list(AuthorityAction)) for step in episode.steps]
    assert actions == [AuthorityAction.HUMAN, AuthorityAction.HUMAN]
    assert all(np.isinf(policy.escalation_score(step.state)) for step in episode.steps)


def test_joint_risk_reserve_and_group_robust_calibration_use_fitting_only():
    episodes = []
    for index in range(6):
        episode = sample_episode(episode_id=f"risk-{index}", group=f"risk-group-{index}")
        state = replace(episode.steps[0].state, drift_risk=index / 5.0, uncertainty=index / 5.0)
        ai = replace(
            episode.steps[0].outcomes["ai"],
            success=index < 3,
            unsafe=index >= 3,
            terminal=True,
            next_step=None,
        )
        step = replace(episode.steps[0], state=state, outcomes={**episode.steps[0].outcomes, "ai": ai})
        episodes.append(replace(episode, steps=[step]))
    encoder = FeatureEncoder.fit([episode.steps[0].state for episode in episodes])
    base = DQNPolicy(encoder, DQNConfig(training_epochs=0, seed=13))
    support = {
        (support_key(episode.steps[0].state), int(AuthorityAction.AI)): 3
        for episode in episodes
    }
    policy = JointCalibratedPolicy(
        base, support, human_budget=1.0, unsafe_budget=1.0,
        max_stopped_rate=1.0, grid_size=3, human_reserve_rate=0.5,
        human_reserve_mode="risk", calibration_folds=3, reserve_seed=13,
        human_reserve_grid_size=3,
    ).fit(episodes)
    assert policy.calibration["human_reserve_mode"] == "risk"
    assert policy.calibration["calibration_folds"] == 3
    assert policy.calibration["human_reserve_rate_candidates"] == 3
    assert policy.calibration["human_reserve_rate"] in {0.0, 0.25, 0.5}
    assert (policy.calibration["risk_reserve_threshold"] is None) == (
        policy.calibration["human_reserve_rate"] == 0.0
    )
    assert len(policy._group_calibration_folds(episodes)) == 3
    low = policy._risk_score(episodes[0].steps[0].state)
    high = policy._risk_score(episodes[-1].steps[0].state)
    assert high > low


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dqn_and_vector_dqn_cuda_smoke():
    episodes = [sample_episode(episode_id=f"cuda-{index}", group=f"cuda-{index}") for index in range(4)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    replay = collect_frozen_logged_replay(episodes, encoder, behavior_seed=17)
    dqn = DQNConfig(training_epochs=0, batch_size=2, device="cuda", seed=17)
    scalar = DQNPolicy(encoder, replace(dqn, constrained=False)).fit_replay(replay, updates=2)
    vector = VectorDQNPolicy(encoder, VectorDQNConfig(dqn, 0.0, 4.0)).fit_replay(replay, updates=2)
    assert scalar.online.layers[0].weight.is_cuda
    assert vector.online.layers[0].weight.is_cuda
    assert np.isfinite(scalar.values(episodes[0].steps[0].state)).all()
    assert np.isfinite(vector.scalar_values(episodes[0].steps[0].state)).all()


def test_legacy_posthoc_protocol_signature_remains_resumable_but_router_does_not():
    legacy = {"paired_replay": False}
    posthoc = {"paired_replay": False, "support_mode": "posthoc_stop", "minimum_ai_support": 3}
    router = {**posthoc, "support_mode": "calibrated_router", "router_human_budget": 0.15}
    assert compatible_protocol_signature(legacy, posthoc)
    assert not compatible_protocol_signature(legacy, router)


def test_support_audit_separates_static_coverage_from_policy_actions():
    episode = sample_episode()
    changed_state = replace(episode.steps[1].state, gap_type="service_selection")
    changed_step = replace(episode.steps[1], state=changed_state)
    episode = replace(episode, steps=[episode.steps[0], changed_step])
    counts = {(support_key(episode.steps[0].state), int(AuthorityAction.AI)): 3}
    report = summarize_support_coverage([episode], counts, minimum_ai_support=3, top_buckets=10)
    assert report["selection_ai_feasible_state_visits"] == 2
    assert report["selection_ai_supported_state_visits"] == 1
    assert report["selection_ai_unsupported_state_visits"] == 1
    assert report["selection_ai_unsupported_state_rate"] == 0.5


def test_primary_dqn_defaults_to_logged_exploratory_replay():
    assert DQNConfig().paired_replay is False


def test_constraint_switches_control_calibration_penalties():
    episode = sample_episode(split="validation")
    encoder = FeatureEncoder.fit([step.state for step in episode.steps])
    human_only = DQNPolicy(encoder, DQNConfig(training_epochs=0, human_constraint=True, unsafe_constraint=False, seed=8))
    human_only.calibrate_constraints([episode], human_budget=0.0, unsafe_budget=0.0, grid_size=5)
    assert human_only.ai_calibration_penalty == 0.0
    unsafe_only = DQNPolicy(encoder, DQNConfig(training_epochs=0, human_constraint=False, unsafe_constraint=True, seed=9))
    unsafe_only.calibrate_constraints([episode], human_budget=0.0, unsafe_budget=0.0, grid_size=5)
    assert unsafe_only.human_calibration_penalty == 0.0

    first = replace(episode.steps[0].outcomes["ai"], unsafe=True)
    second = replace(episode.steps[1].outcomes["ai"], success=False, unsafe=True)
    unsafe_episode = replace(episode, steps=[
        replace(episode.steps[0], outcomes={**episode.steps[0].outcomes, "ai": first}),
        replace(episode.steps[1], outcomes={**episode.steps[1].outcomes, "ai": second}),
    ])
    human_only.calibrate_constraints([unsafe_episode], human_budget=1.0, unsafe_budget=0.0, grid_size=5)
    assert human_only.feasible_checkpoint_found


def test_human_budget_calibration_records_finite_validation_metrics():
    episode = sample_episode(split="validation")
    encoder = FeatureEncoder.fit([step.state for step in episode.steps])
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=0, constrained=False, seed=10))
    policy.calibrate_human_budget([episode], target_rate=0.5)
    assert np.isfinite(policy.calibration_validation_hir)
    assert np.isfinite(policy.calibration_validation_unsafe)
    assert policy.calibration_unsafe_target == 1.0


def test_encoder_and_dqn_checkpoint_round_trip(tmp_path):
    episodes = [sample_episode(episode_id=f"e{i}", group=f"g{i}") for i in range(4)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    encoder_path = tmp_path / "encoder.json"
    encoder.save(encoder_path)
    loaded_encoder = FeatureEncoder.load(encoder_path)
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=1, batch_size=2, seed=4)).fit(episodes)
    policy.human_calibration_penalty = 0.3
    policy.ai_calibration_penalty = 0.4
    checkpoint = tmp_path / "model.pt"
    policy.save(str(checkpoint))
    loaded = DQNPolicy.load(str(checkpoint), loaded_encoder)
    state = episodes[0].steps[0].state
    assert np.allclose(policy.values(state), loaded.values(state))
    assert loaded.human_calibration_penalty == 0.3
    assert loaded.ai_calibration_penalty == 0.4


def test_joint_constraint_calibration_can_route_to_stop():
    base = sample_episode()
    failed = BranchOutcome(False, False, -0.3, 0.1, 0.1, terminal=True)
    state = base.steps[0].state
    step = PairedRepairStep(state, {**base.steps[0].outcomes, "ai": failed}, True)
    episode = RepairEpisode("cal", "cal", "validation", "test", True, [step])
    encoder = FeatureEncoder.fit([state])
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=1, batch_size=1, seed=5))
    policy.calibrate_constraints([episode], human_budget=0.0, unsafe_budget=0.0, grid_size=7)
    selected = policy.select(state, [AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP])
    assert selected == AuthorityAction.STOP
    assert policy.feasible_checkpoint_found


def test_ensemble_calibration_uses_member_values():
    episode = sample_episode()
    state = episode.steps[0].state
    encoder = FeatureEncoder.fit([step.state for step in episode.steps])
    members = [DQNPolicy(encoder, DQNConfig(training_epochs=0, seed=seed)) for seed in (1, 2)]
    ensemble = DQNEnsemblePolicy(members, encoder, DQNConfig(training_epochs=0, seed=3))
    for member in members:
        for parameter in member.online.parameters():
            parameter.data.zero_()
        member.online.layers[-1].bias.data[:] = torch.tensor([2.0, 1.0, 0.0])
    for parameter in ensemble.online.parameters():
        parameter.data.zero_()
    ensemble.online.layers[-1].bias.data[:] = torch.tensor([0.0, 0.0, 9.0])
    assert ensemble.select(state, list(AuthorityAction)) == AuthorityAction.AI
    assert np.allclose(ensemble.values_batch([state])[0], ensemble.values(state))


def test_evaluation_metrics_use_observed_branches():
    result = evaluate_policy(AutonomousPolicy(), [sample_episode()])
    assert result["repair_sr"] == 1.0
    assert result["hir"] == 0.0
    assert result["decision_states"] == 2
    assert result["collaboration_regret"] > 0


def test_uar_counts_incorrect_autonomous_completion():
    episode = sample_episode()
    failed = BranchOutcome(False, False, -0.2, 0.1, 0.1, terminal=True)
    step = PairedRepairStep(episode.steps[0].state, {**episode.steps[0].outcomes, "ai": failed}, True)
    one_step = RepairEpisode("bad", "bad", "test", "test", True, [step])
    result = evaluate_policy(AutonomousPolicy(), [one_step])
    assert result["uar"] == 1.0


def test_unsafe_cmdp_cost_counts_each_unsafe_action():
    episode = sample_episode()
    first = replace(episode.steps[0].outcomes["ai"], unsafe=True)
    second = replace(episode.steps[1].outcomes["ai"], success=False, unsafe=True)
    steps = [
        replace(episode.steps[0], outcomes={**episode.steps[0].outcomes, "ai": first}),
        replace(episode.steps[1], outcomes={**episode.steps[1].outcomes, "ai": second}),
    ]
    result = evaluate_policy(AutonomousPolicy(), [replace(episode, steps=steps)])
    assert result["records"][0]["unsafe_actions"] == 2
    assert result["unsafe_autonomy_cost_rate"] == 2.0


def test_one_sided_safety_certificate_rejects_nominal_validation_boundary():
    certificate = certify_unsafe_rate(unsafe_actions=6, observations=127, target=0.05)
    assert certificate.upper_bound == pytest.approx(0.09111, abs=1e-4)
    assert not certificate.certified
    assert minimum_observations_for_zero_unsafe(0.05) == 59
    assert maximum_certifiable_unsafe_actions(127, 0.05, 0.975) == 1
    human, unsafe = certify_policy_costs(1, 1, 127, 0.2, 0.05)
    assert human.certified and unsafe.certified


def test_group_level_certificate_exposes_insufficient_validation_power():
    certificate = certify_group_mean(0.0, groups=125, target=0.05, confidence=0.975, bound_range=3.0)
    assert certificate.upper_bound == pytest.approx(0.36442, abs=1e-4)
    assert not certificate.certified
    assert minimum_groups_for_zero_mean(0.05, 0.975, 3.0) == 6640


def test_support_gate_only_blocks_unsupported_ai_action():
    state = sample_episode().steps[0].state
    policy = AutonomousPolicy()
    wrapped = SupportCheckedPolicy(policy, {}, minimum_count=3)
    assert wrapped.select(state, [AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP]) == AuthorityAction.STOP
    assert wrapped.fallback_actions == 1
    support = {(support_key(state), int(AuthorityAction.AI)): 3}
    wrapped = SupportCheckedPolicy(policy, support, minimum_count=3)
    assert wrapped.select(state, [AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP]) == AuthorityAction.AI
    assert wrapped.fallback_actions == 0


def test_group_stable_selection_never_evaluates_unstable_candidate_on_test():
    episodes = [sample_episode(split="train", group=f"g-{index}", episode_id=f"e-{index}") for index in range(6)]
    configuration = StableSelectionConfig(human_operating_target=0.0, unsafe_operating_target=0.0, folds=3)
    result = select_stable_candidate({"always_human": AlwaysHumanPolicy()}, episodes, configuration)
    assert result.selected is None
    assert len(group_folds(episodes, configuration)) == 3


def test_group_fold_keeps_duplicate_group_episodes_together_and_enforces_unsafe_boundary():
    episodes = []
    for group in range(3):
        safe = sample_episode(split="train", group=f"boundary-{group}", episode_id=f"safe-{group}")
        unsafe_outcome = replace(safe.steps[0].outcomes["ai"], unsafe=True)
        unsafe = replace(safe, episode_id=f"unsafe-{group}", steps=[
            replace(safe.steps[0], state=replace(safe.steps[0].state, episode_id=f"unsafe-{group}"), outcomes={**safe.steps[0].outcomes, "ai": unsafe_outcome}),
            replace(safe.steps[1], state=replace(safe.steps[1].state, episode_id=f"unsafe-{group}")),
        ])
        episodes.extend([safe, unsafe])
    configuration = StableSelectionConfig(human_operating_target=0.0, unsafe_operating_target=0.5, folds=3)
    folds = group_folds(episodes, configuration)
    assert all(len({episode.group_id for episode in fold}) == 1 for fold in folds)
    assert select_stable_candidate({"autonomous": AutonomousPolicy()}, episodes, configuration).selected == "autonomous"
    extra = episodes[0]
    unsafe_outcome = replace(extra.steps[0].outcomes["ai"], unsafe=True)
    episodes[0] = replace(extra, steps=[replace(extra.steps[0], outcomes={**extra.steps[0].outcomes, "ai": unsafe_outcome}), extra.steps[1]])
    assert select_stable_candidate({"autonomous": AutonomousPolicy()}, episodes, configuration).selected is None


def test_vector_dqn_trains_three_cost_heads_with_masked_actions():
    episodes = [sample_episode(episode_id=f"vector-{index}", group=f"vector-{index}") for index in range(4)]
    encoder = FeatureEncoder.fit([step.state for episode in episodes for step in episode.steps])
    dqn = DQNConfig(training_epochs=1, batch_size=2, epsilon_decay_steps=3, seed=12)
    policy = VectorDQNPolicy(encoder, VectorDQNConfig(dqn, human_dual=0.5, unsafe_dual=1.0)).fit(episodes)
    values = policy._raw_values(episodes[0].steps[0].state)
    assert values.shape == (3, 3)
    exhausted = replace(episodes[0].steps[0].state, remaining_budget=0.0)
    assert policy.select(exhausted, [AuthorityAction.HUMAN, AuthorityAction.STOP]) in {AuthorityAction.HUMAN, AuthorityAction.STOP}


def test_exact_action_ties_use_stop_then_human_then_ai_for_learning_and_execution():
    values = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    assert masked_conservative_argmax(values, mask).tolist() == [int(AuthorityAction.STOP), int(AuthorityAction.HUMAN)]


def test_cluster_bootstrap_reports_single_policy_episode_metric_with_group_resampling():
    episodes = [sample_episode(episode_id=f"bootstrap-{index}", group=f"group-{index // 2}") for index in range(4)]
    records = evaluate_policy(AutonomousPolicy(), episodes)["records"]
    groups = {episode.episode_id: episode.group_id for episode in episodes}
    low, high = cluster_bootstrap_ci(records, groups, "repair_sr", samples=100, seed=9)
    assert (low, high) == (1.0, 1.0)


def test_locked_holdout_manifest_is_time_bound_and_group_disjoint(tmp_path):
    development_path = tmp_path / "development.jsonl"
    source_path = tmp_path / "source.jsonl"
    identifier_index_path = tmp_path / "historical_identifiers.json"
    raw_holdout_path = tmp_path / "raw_holdout.jsonl"
    raw_historical_holdout_path = tmp_path / "raw_historical_holdout.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    historical_holdout_path = tmp_path / "historical_holdout.jsonl"
    manifest_path = tmp_path / "holdout_manifest.json"
    historical_manifest_path = tmp_path / "historical_holdout_manifest.json"
    import_report_path = tmp_path / "import_report.json"
    historical_import_report_path = tmp_path / "historical_import_report.json"
    development = [
        sample_episode(split="train", group=f"development-group-{index}", episode_id=f"development-episode-{index}")
        for index in range(30)
    ]
    bound_holdouts, ledger_path, artifact_root = bind_paired_artifacts(
        tmp_path,
        [
            real_paired_episode(split="test", group="locked-group", episode_id="locked-episode"),
            real_paired_episode(split="test", group="historical-group", episode_id="fresh-historical-id"),
        ],
    )
    holdout = [bound_holdouts[0]]
    historical_holdout = [bound_holdouts[1]]
    historical = sample_episode(split="validation", group="historical-group", episode_id="historical-episode")
    write_episodes(source_path, [*development, historical])
    subprocess.run([
        sys.executable, "scripts/create_share_guard_development_data.py",
        "--episodes", str(source_path),
        "--out", str(development_path),
        "--metadata-out", str(identifier_index_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True)
    write_episodes(raw_holdout_path, holdout)
    write_episodes(raw_historical_holdout_path, historical_holdout)
    for input_path, output_path, report_path in (
        (raw_holdout_path, holdout_path, import_report_path),
        (raw_historical_holdout_path, historical_holdout_path, historical_import_report_path),
    ):
        subprocess.run([
            sys.executable, "scripts/import_share_real_paired_episodes.py",
            "--input", str(input_path),
            "--out", str(output_path),
            "--report", str(report_path),
            "--artifact-ledger", str(ledger_path),
            "--artifact-root", str(artifact_root),
        ], cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run([
        sys.executable, "scripts/create_share_locked_holdout_manifest.py",
        "--episodes", str(holdout_path),
        "--historical-identifier-index", str(identifier_index_path),
        "--frozen-at", "2026-08-12T00:00:00Z",
        "--attestation", "doi:10.0000/example-preregistration",
        "--import-report", str(import_report_path),
        "--out", str(manifest_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run([
        sys.executable, "scripts/create_share_locked_holdout_manifest.py",
        "--episodes", str(historical_holdout_path),
        "--historical-identifier-index", str(identifier_index_path),
        "--frozen-at", "2026-08-12T00:00:00Z",
        "--attestation", "doi:10.0000/example-preregistration",
        "--import-report", str(historical_import_report_path),
        "--out", str(historical_manifest_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True)
    identifier_provenance = load_historical_identifier_index(identifier_index_path, development_path, load_episodes(development_path))
    provenance = validate_locked_holdout(holdout_path, manifest_path, load_episodes(holdout_path), load_episodes(development_path), identifier_provenance)
    assert provenance["group_overlap"] == 0
    assert provenance["manifest"]["prior_unseen_attestation"] == "doi:10.0000/example-preregistration"
    assert provenance["manifest"]["ai_model_episode_counts"] == {"model-a": 1}
    with pytest.raises(ValueError, match="group leakage"):
        validate_locked_holdout(
            historical_holdout_path,
            historical_manifest_path,
            load_episodes(historical_holdout_path),
            load_episodes(development_path),
            identifier_provenance,
        )
    guard_results = tmp_path / "guard-locked-results"
    subprocess.run([
        sys.executable, "scripts/run_share_guard_experiments.py",
        "--episodes", str(development_path),
        "--locked-holdout", str(holdout_path),
        "--locked-holdout-manifest", str(manifest_path),
        "--historical-identifier-index", str(identifier_index_path),
        "--out", str(guard_results),
        "--sanity",
        "--bootstrap-samples", "10",
    ], cwd=Path(__file__).resolve().parents[1], check=True)
    guard_result = json.loads((guard_results / "results.json").read_text())
    assert guard_result["locked_holdout_evaluation"]["status"] == "not_evaluated"
    assert guard_result["locked_holdout_evaluation"]["reason"] == "all_predeclared_methods_must_select_a_stable_deployment_candidate"
    assert guard_result["development_diagnostics"] == {}
    manifest = json.loads(manifest_path.read_text())
    manifest["frozen_at"] = "2026-08-12T00:00:00"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="UTC offset"):
        validate_locked_holdout(holdout_path, manifest_path, load_episodes(holdout_path), load_episodes(development_path), identifier_provenance)


def test_real_paired_importer_binds_branch_provenance_and_rejects_synthetic_input(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "external_input.jsonl"
    output_path = tmp_path / "canonical_real.jsonl"
    report_path = tmp_path / "real_import.json"
    real, ledger_path, artifact_root = bind_paired_artifacts(tmp_path, [real_paired_episode()])
    real = real[0]
    write_episodes(input_path, [real])
    subprocess.run([
        sys.executable, "scripts/import_share_real_paired_episodes.py",
        "--input", str(input_path),
        "--out", str(output_path),
        "--report", str(report_path),
        "--artifact-ledger", str(ledger_path),
        "--artifact-root", str(artifact_root),
    ], cwd=repository, check=True)
    report = json.loads(report_path.read_text())
    assert report["evidence_kind"] == "externally_collected_real_paired_outcomes"
    assert report["ai_model_episode_counts"] == {"model-a": 1}
    assert report["human_benefit_label_rule"] == "share_label_human_benefit_v1"
    assert load_episodes(output_path)[0].synthetic is False
    with pytest.raises(ValueError, match="rejects synthetic"):
        validate_real_paired_episodes([sample_episode()], {})
    with pytest.raises(ValueError, match="human-benefit"):
        validate_real_paired_episodes([real_paired_episode()], {})
    ledger = load_artifact_ledger(ledger_path, artifact_root)
    altered_ai = replace(real.steps[0].outcomes["ai"], success=not real.steps[0].outcomes["ai"].success)
    altered = replace(real, steps=[replace(real.steps[0], outcomes={**real.steps[0].outcomes, "ai": altered_ai}), *real.steps[1:]])
    with pytest.raises(ValueError, match="outcome does not match"):
        validate_real_paired_episodes([altered], ledger)
    altered_label = replace(real.steps[0], human_beneficial=not real.steps[0].human_beneficial)
    with pytest.raises(ValueError, match="human_beneficial"):
        validate_real_paired_episodes([replace(real, steps=[altered_label, *real.steps[1:]])], ledger)
    synthetic_path = tmp_path / "synthetic.jsonl"
    write_episodes(synthetic_path, [sample_episode(split="test")])
    rejected = subprocess.run([
        sys.executable, "scripts/import_share_real_paired_episodes.py",
        "--input", str(synthetic_path),
        "--out", str(tmp_path / "rejected.jsonl"),
        "--report", str(tmp_path / "rejected.json"),
        "--artifact-ledger", str(ledger_path),
        "--artifact-root", str(artifact_root),
    ], cwd=repository, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "rejects synthetic" in rejected.stderr
    overlap = subprocess.run([
        sys.executable, "scripts/import_share_real_paired_episodes.py",
        "--input", str(input_path),
        "--out", str(artifact_root / "must-not-write.jsonl"),
        "--report", str(tmp_path / "overlap-report.json"),
        "--artifact-ledger", str(ledger_path),
        "--artifact-root", str(artifact_root),
    ], cwd=repository, text=True, capture_output=True)
    assert overlap.returncode != 0
    assert "artifact-root" in overlap.stderr
    artifact = next(artifact_root.glob("*.json"))
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_artifact_ledger(ledger_path, artifact_root)


def test_llm_profile_preflight_keeps_secrets_out_of_its_report(tmp_path):
    env_path = tmp_path / "profiles.env"
    env_path.write_text(
        "\n".join([
            "LLM_BASE_URL=https://primary.example/v1",
            "LLM_API_KEY=primary-secret",
            "LLM_MODEL=first-model",
            "LLM_BACKBONE_2_BASE_URL=https://second.example/v1",
            "LLM_BACKBONE_2_API_KEY=second-secret",
            "LLM_BACKBONE_2_MODEL=gpt-5.6-luna",
            "LLM_BACKBONE_2_REASONING_EFFORT=low",
            "LLM_BACKBONE_3_BASE_URL=https://third.example/v1",
            "LLM_BACKBONE_3_API_KEY=third-secret",
            "LLM_BACKBONE_3_MODEL=gpt-5.6-terra",
            "LLM_BACKBONE_3_REASONING_EFFORT=low",
        ]) + "\n"
    )
    profiles = load_llm_profiles(env_path)
    assert [(profile.model, profile.reasoning_effort) for profile in profiles] == [
        ("first-model", None), ("gpt-5.6-luna", "low"), ("gpt-5.6-terra", "low"),
    ]
    report = public_profile_report(env_path)
    assert report["credentials_configured"] is True
    serialized_report = json.dumps(report)
    assert all(value not in serialized_report for value in ("primary-secret", "second-secret", "third-secret"))
    cli_report_path = tmp_path / "profile_report.json"
    subprocess.run([
        sys.executable, "scripts/check_share_llm_profiles.py",
        "--env", str(env_path),
        "--out", str(cli_report_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True)
    assert json.loads(cli_report_path.read_text())["credentials_configured"] is True
    env_path.write_text(env_path.read_text().replace("LLM_BACKBONE_3_REASONING_EFFORT=low", "LLM_BACKBONE_3_REASONING_EFFORT=invalid"))
    with pytest.raises(ValueError, match="unsupported REASONING_EFFORT"):
        load_llm_profiles(env_path)
    env_path.write_text(env_path.read_text().replace("LLM_BACKBONE_3_REASONING_EFFORT=invalid", "LLM_BACKBONE_3_REASONING_EFFORT=low").replace("https://primary.example/v1", "http://primary.example/v1"))
    with pytest.raises(ValueError, match="requires HTTPS"):
        load_llm_profiles(env_path)
    with pytest.raises(ValueError, match="only permits insecure HTTP for a loopback proxy"):
        public_profile_report(env_path, allow_insecure_http=True)
    env_path.write_text(env_path.read_text().replace("http://primary.example/v1", "http://localhost:8080/v1"))
    report = public_profile_report(env_path, allow_insecure_http=True)
    assert report["insecure_http_acknowledged"] is True
    env_path.write_text(env_path.read_text().replace("http://localhost:8080/v1", "https://primary.example/v1/chat/completions"))
    with pytest.raises(ValueError, match="API root"):
        load_llm_profiles(env_path)


def test_historical_identifier_index_rejects_incomplete_metadata(tmp_path):
    source_path = tmp_path / "source.jsonl"
    development_path = tmp_path / "development.jsonl"
    index_path = tmp_path / "historical_identifiers.json"
    episodes = [sample_episode(split="train", group="train", episode_id="train"), sample_episode(split="validation", group="validation", episode_id="validation")]
    write_episodes(source_path, episodes)
    repository = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "scripts/create_share_guard_development_data.py",
        "--episodes", str(source_path),
        "--out", str(development_path),
        "--metadata-out", str(index_path),
    ], cwd=repository, check=True)
    index = json.loads(index_path.read_text())
    index["historical_group_ids"] = ["train"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="source metadata is inconsistent"):
        load_historical_identifier_index(index_path, development_path, load_episodes(development_path))


def test_paired_locked_holdout_comparison_uses_all_episode_pairs():
    episodes = [sample_episode(episode_id=f"paired-{index}", group=f"paired-group-{index // 2}") for index in range(4)]
    candidate = evaluate_policy(AutonomousPolicy(), episodes)["records"]
    reference = evaluate_policy(AlwaysHumanPolicy(), episodes)["records"]
    comparisons = paired_comparison(candidate, reference, episodes, bootstrap_samples=100, bootstrap_seed=4)
    assert set(comparisons) == {"repair_sr", "normalized_return", "hir", "unsafe_autonomy_cost_rate", "stopped_rate"}
    assert all("point_difference" in comparison and "ci95" in comparison for comparison in comparisons.values())


def observed_event_value(index, state, mode, policy_sha256, evidence_sha256, decision_at, action=AuthorityAction.AI):
    outcome = {
        "success": action == AuthorityAction.HUMAN,
        "unsafe": False,
        "quality_delta": 0.7 if action == AuthorityAction.HUMAN else -0.2,
        "latency": 0.2,
        "execution_cost": 0.2,
        "human_minutes": 5.0 if action == AuthorityAction.HUMAN else 0.0,
        "next_step": None,
        "terminal": True,
        "metadata": {"execution_source": "immutable-observed-record"},
    }
    event = {
        "schema_version": 2,
        "evidence_kind": "observed_deployment",
        "event_id": f"event-{index}",
        "episode_id": f"deployment-{index}",
        "group_id": f"deployment-group-{index}",
        "split": "test",
        "source": "externally-attested-deployment-ledger",
        "decision_at": decision_at,
        "executed_at": decision_at.replace("00:00Z", "00:01Z").replace("01:00Z", "01:01Z"),
        "outcome_observed_at": decision_at.replace("00:00Z", "00:02Z").replace("01:00Z", "01:02Z"),
        "execution_id": f"execution-{index}",
        "execution_artifact_sha256": f"{index + 1:064x}",
        "policy_mode": mode,
        "policy_state_sha256": policy_sha256,
        "evidence_snapshot_sha256": evidence_sha256,
        "state": dict(state.__dict__),
        "selected_action": int(action),
        "outcome": outcome,
        "next_state": None,
        "outcome_sha256": canonical_digest(outcome),
    }
    event["event_digest"] = canonical_digest(event)
    return event


def correction_value(episode_id, action, validated_at):
    observed_outcome = {
        "success": action == AuthorityAction.HUMAN,
        "unsafe": False,
        "quality_delta": 0.7 if action == AuthorityAction.HUMAN else -0.2,
        "latency": 0.2,
        "execution_cost": 0.2,
        "human_minutes": 5.0 if action == AuthorityAction.HUMAN else 0.0,
        "next_step": None,
        "terminal": True,
        "metadata": {"execution_source": "immutable-observed-record"},
    }
    validated_outcome = {
        **observed_outcome,
        "success": True,
        "quality_delta": 0.7,
    }
    value = {
        "correction_id": f"correction-{episode_id}",
        "event_id": f"event-{episode_id.rsplit('-', 1)[1]}",
        "episode_id": episode_id,
        "step": 0,
        "observed_action": int(action),
        "observed_outcome_sha256": canonical_digest(observed_outcome),
        "validated_outcome": validated_outcome,
        "validated_outcome_sha256": canonical_digest(validated_outcome),
        "validation_artifact_sha256": "c" * 64,
        "validated_success": True,
        "validated_unsafe": False,
        "validated_at": validated_at,
        "validator_id": "independent-reviewer-1",
        "provenance": "external-adjudication-record-1",
    }
    value["correction_digest"] = correction_digest(value)
    return value


def test_validated_evidence_store_binds_post_termination_observed_events(tmp_path):
    state = sample_episode(split="test", group="first", episode_id="deployment-0").steps[0].state
    event = observed_event_value(0, state, "adaptive", "a" * 64, ValidatedEvidenceStore([]).snapshot(())["content_sha256"], "2026-01-01T00:00:00Z")
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(json.dumps(event) + "\n")
    correction = ValidatedCorrection.from_dict(correction_value("deployment-0", AuthorityAction.AI, "2026-01-02T00:00:00Z"))
    store = ValidatedEvidenceStore([correction])
    store.bind_deployment_events(load_deployment_events(event_path))
    assert store.visible_before(parse_timestamp("2026-01-02T00:00:00Z")) == ()
    visible = store.visible_before(parse_timestamp("2026-01-03T00:00:00Z"))
    second = replace(state, episode_id="deployment-1", uncertainty=0.9)
    adjusted = store.feature_adjustment(second, visible)
    assert adjusted.uncertainty == pytest.approx(1.0 / np.sqrt(2.0))
    too_early = ValidatedCorrection.from_dict(correction_value("deployment-0", AuthorityAction.AI, "2026-01-01T00:00:02Z"))
    with pytest.raises(ValueError, match="after deployment episode termination"):
        ValidatedEvidenceStore([too_early]).bind_deployment_events(load_deployment_events(event_path))


def test_deployment_events_reject_within_episode_policy_change_and_corrected_transition_change(tmp_path):
    episode = sample_episode(split="test", group="first", episode_id="deployment-0")
    snapshot = ValidatedEvidenceStore([]).snapshot(())["content_sha256"]
    first = observed_event_value(0, episode.steps[0].state, "adaptive", "a" * 64, snapshot, "2026-01-01T00:00:00Z")
    first["outcome"] = {
        "success": False, "unsafe": False, "quality_delta": -0.2, "latency": 0.2,
        "execution_cost": 0.2, "human_minutes": 0.0, "next_step": 1, "terminal": False,
        "metadata": {"execution_source": "immutable-observed-record"},
    }
    first["next_state"] = dict(episode.steps[1].state.__dict__)
    first["outcome_sha256"] = canonical_digest(first["outcome"])
    first["event_digest"] = canonical_digest({key: value for key, value in first.items() if key != "event_digest"})
    second = observed_event_value(1, episode.steps[1].state, "adaptive", "b" * 64, snapshot, "2026-01-01T00:01:00Z")
    second["episode_id"] = "deployment-0"
    second["group_id"] = "deployment-group-0"
    second["state"] = dict(episode.steps[1].state.__dict__)
    second["event_digest"] = canonical_digest({key: value for key, value in second.items() if key != "event_digest"})
    event_path = tmp_path / "changed_policy_events.jsonl"
    event_path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="inconsistent metadata"):
        load_deployment_events(event_path)

    stable_second = {**second, "policy_state_sha256": "a" * 64}
    stable_second["event_digest"] = canonical_digest({key: value for key, value in stable_second.items() if key != "event_digest"})
    event_path.write_text(json.dumps(first) + "\n" + json.dumps(stable_second) + "\n")
    correction = correction_value("deployment-0", AuthorityAction.AI, "2026-01-02T00:00:00Z")
    correction["observed_outcome_sha256"] = canonical_digest(first["outcome"])
    correction["correction_digest"] = correction_digest(correction)
    with pytest.raises(ValueError, match="cannot revise the observed transition structure"):
        ValidatedEvidenceStore([ValidatedCorrection.from_dict(correction)]).bind_deployment_events(load_deployment_events(event_path))


def test_longitudinal_runner_summarizes_observed_arms_and_only_updates_adaptive_events(tmp_path):
    states = [sample_episode(split="test", group=f"group-{index}", episode_id=f"deployment-{index}").steps[0].state for index in range(4)]
    event_path = tmp_path / "deployment_events.jsonl"
    corrections_path = tmp_path / "corrections.jsonl"
    manifest_path = tmp_path / "deployment_manifest.json"
    encoder_path = tmp_path / "encoder.json"
    checkpoint_path = tmp_path / "frozen.pt"
    output_path = tmp_path / "longitudinal"
    encoder = FeatureEncoder.fit(states)
    encoder.save(encoder_path)
    policy = DQNPolicy(encoder, DQNConfig(training_epochs=0, batch_size=1, seed=31))
    policy.save(str(checkpoint_path))
    frozen_sha256 = sha256_file(checkpoint_path)
    adaptive_sha256 = "a" * 64
    correction_raw = correction_value("deployment-1", AuthorityAction.AI, "2026-01-02T12:00:00Z")
    correction = ValidatedCorrection.from_dict(correction_raw)
    empty_snapshot = ValidatedEvidenceStore([]).snapshot(())["content_sha256"]
    corrected_snapshot = ValidatedEvidenceStore([correction]).snapshot((correction,))["content_sha256"]
    events = [
        observed_event_value(0, states[0], "frozen", frozen_sha256, empty_snapshot, "2026-01-01T00:00:00Z", AuthorityAction.HUMAN),
        observed_event_value(1, states[1], "adaptive", adaptive_sha256, empty_snapshot, "2026-01-01T01:00:00Z"),
        observed_event_value(2, states[2], "frozen", frozen_sha256, corrected_snapshot, "2026-01-03T00:00:00Z", AuthorityAction.HUMAN),
        observed_event_value(3, states[3], "adaptive", adaptive_sha256, corrected_snapshot, "2026-01-03T01:00:00Z"),
    ]
    event_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    corrections_path.write_text(json.dumps(correction_raw) + "\n")
    repository = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "scripts/create_share_longitudinal_manifest.py",
        "--deployment-events", str(event_path),
        "--encoder", str(encoder_path),
        "--frozen-at", "2025-12-31T00:00:00Z",
        "--prior-unseen-attestation", "external-preregistration-1",
        "--external-provenance-attestation", "external-audit-ledger-1",
        "--assignment-protocol", "predeclared-balanced-interleaved-arms",
        "--frozen-policy-sha256", frozen_sha256,
        "--adaptive-policy-sha256", adaptive_sha256,
        "--code-revision", "test-revision",
        "--out", str(manifest_path),
    ], cwd=repository, check=True)
    subprocess.run([
        sys.executable, "scripts/run_share_longitudinal.py",
        "--deployment-events", str(event_path),
        "--deployment-manifest", str(manifest_path),
        "--validated-corrections", str(corrections_path),
        "--checkpoint", str(checkpoint_path),
        "--encoder", str(encoder_path),
        "--windows", "2",
        "--updates-per-correction", "1",
        "--out", str(output_path),
    ], cwd=repository, check=True)
    rows = json.loads((output_path / "longitudinal.json").read_text())
    audit = json.loads((output_path / "update_audit.json").read_text())
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert {(row["mode"], row["round"]) for row in rows} == {("frozen", 1), ("adaptive", 1), ("frozen", 2), ("adaptive", 2)}
    assert all(row["protocol_only"] and not row["causal_frozen_vs_adaptive_claim_supported"] for row in rows)
    assert rows[0]["evidence_snapshot"]["validated_corrections"] == 0
    assert rows[-1]["evidence_snapshot"]["validated_corrections"] == 1
    assert audit[0]["accepted_observed_adaptive_transitions_after_window"][0]["event_id"] == "event-1"
    assert audit[0]["adaptive_gradient_updates_after_window"] == 1
    assert audit[1]["accepted_observed_adaptive_transitions_after_window"] == []
    assert manifest["provenance_verified_by_runner"] is False
    assert len(json.loads((output_path / "episode_records.json").read_text())) == 4


def test_longitudinal_runner_rejects_paired_episode_counterfactual_input(tmp_path):
    paired_path = tmp_path / "paired_episodes.jsonl"
    write_episodes(paired_path, [sample_episode(split="test", group="paired", episode_id="paired")])
    completed = subprocess.run([
        sys.executable, "scripts/run_share_longitudinal.py",
        "--deployment-events", str(paired_path),
        "--deployment-manifest", str(tmp_path / "missing_manifest.json"),
        "--validated-corrections", str(tmp_path / "missing_corrections.jsonl"),
        "--checkpoint", str(tmp_path / "missing_checkpoint.pt"),
        "--encoder", str(tmp_path / "missing_encoder.json"),
        "--windows", "2",
    ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "deployment event missing fields" in completed.stderr


def test_guard_protocol_reserves_validation_for_independent_certification():
    episodes = [sample_episode(split="train", group=f"train-{index}", episode_id=f"train-{index}") for index in range(10)]
    episodes += [sample_episode(split="validation", group=f"val-{index}", episode_id=f"val-{index}") for index in range(2)]
    episodes += [sample_episode(split="test", group=f"test-{index}", episode_id=f"test-{index}") for index in range(2)]
    partitions = guard_partitions(episodes, seed=4, selection_fraction=0.3)
    assert len(partitions.certification) == 2
    assert {episode.split for episode in partitions.certification} == {"validation"}
    assert {episode.group_id for episode in partitions.fitting}.isdisjoint({episode.group_id for episode in partitions.selection})


def test_episode_validation_rejects_malformed_transition():
    episode = sample_episode()
    malformed = BranchOutcome(False, False, -0.2, 0.1, 0.1, next_step=None, terminal=False)
    step = PairedRepairStep(episode.steps[0].state, {**episode.steps[0].outcomes, "ai": malformed}, True)
    with pytest.raises(ValueError, match="lacks next_step"):
        validate_splits([RepairEpisode("bad", "bad", "test", "test", True, [step])])


def test_exact_oracle_respects_episode_level_constraints():
    episodes = [sample_episode(episode_id=f"oracle-{index}", group=f"oracle-{index}") for index in range(2)]
    options = enumerate_trajectory_options(episodes[0])
    assert any(option.human_used for option in options)
    assert any(not option.human_used for option in options)
    result = solve_constrained_oracle(episodes, human_budget=0.5, unsafe_budget=0.0)
    assert result.human_limit == 1
    assert sum(option.human_used for option in result.selections) <= 1
    assert sum(option.unsafe_actions for option in result.selections) == 0
    assert result.hir <= 0.5


def test_expected_oracle_uses_probabilities_and_respects_constraints():
    episode = sample_episode()
    result = solve_expected_root_oracle([episode], human_budget=0.0, unsafe_budget=0.0)
    assert result.hir == 0.0
    assert result.expected_unsafe_rate == 0.0
    assert result.stopped_rate == 1.0


def test_expected_oracle_accepts_a_stop_bound():
    episode = sample_episode()
    result = solve_expected_root_oracle(
        [episode],
        human_budget=1.0,
        unsafe_budget=1.0,
        max_stopped_rate=0.0,
    )
    assert result.stopped_rate == 0.0


def test_feature_ablation_and_feedback_corruption_are_explicit():
    episode = sample_episode()
    encoder = FeatureEncoder.fit([step.state for step in episode.steps])
    full = encoder.transform_one(episode.steps[0].state)
    without_risk = encoder.without("risk").transform_one(episode.steps[0].state)
    assert not np.array_equal(full, without_risk)
    corrupted = corrupt_human_feedback([episode], 1.0, 2)[0]
    assert corrupted.steps[0].outcomes["human"].success is False
    assert corrupted.steps[0].outcomes["human"].metadata["corrupted"] is True


def test_paired_hierarchical_statistics_preserve_episode_pairing():
    left = [{"episode_id": "a", "success": True}, {"episode_id": "b", "success": False}]
    right = [{"episode_id": "b", "success": True}, {"episode_id": "a", "success": False}]
    differences = paired_differences(left, right, "repair_sr")
    assert differences == {"a": 1.0, "b": -1.0}
    low, high = hierarchical_bootstrap_ci(differences, {"a": "g1", "b": "g2"}, samples=100, seed=2)
    assert low <= 0.0 <= high
    assert 0.0 < paired_permutation_test(differences, permutations=100, seed=2) <= 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_synthetic_builder_and_cli_sanity(tmp_path):
    row = {
        "id": "failure-1", "dataset": "demo", "task": "Get weather", "metadata": {},
        "events": [
            {"id": "q", "agent": "system", "type": "request", "output": "Get weather", "status": "success", "time": 1},
            {"id": "b", "agent": "executor", "type": "parameter_binding", "output": {}, "meta": {"depends_on": ["q"], "missing_required": ["city"]}, "status": "failed", "time": 2},
        ],
        "repair_label": {"failed": True, "failure_type": "parameter_binding_failure", "faulty_decision_node": "b", "expected_repair_action": "RebindParam", "affected_slice": []},
    }
    episodes = build_synthetic_episodes([row])
    assert episodes and episodes[0].synthetic and set(episodes[0].steps[0].outcomes) == {"ai", "human", "stop"}

    data_root = tmp_path / "flare"
    episode_path = tmp_path / "share" / "episodes.jsonl"
    result_root = tmp_path / "results"
    subprocess.run([sys.executable, "scripts/generate_flare_datasets.py", "--scale", "0.08", "--out", str(data_root), "--seed", "9"], cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run([sys.executable, "scripts/build_share_episodes.py", "--data", str(data_root), "--out", str(episode_path), "--seed", "9"], cwd=Path(__file__).resolve().parents[1], check=True)
    subprocess.run([sys.executable, "scripts/run_share_experiments.py", "--episodes", str(episode_path), "--out", str(result_root), "--sanity", "--batch-size", "4"], cwd=Path(__file__).resolve().parents[1], check=True)
    summary = json.loads((result_root / "summary.json").read_text())
    assert {row["policy"] for row in summary} == {"autonomous_flare", "always_human", "random_budget", "share_controller"}
    assert all(row["synthetic_evidence"] for row in summary)
    share_row = next(row for row in summary if row["policy"] == "share_controller")
    assert share_row["replay_protocol"] == "exploratory_logged_transitions"
    assert share_row["human_constraint_enabled"] and share_row["unsafe_constraint_enabled"]


def test_guard_sanity_run_is_resumable_and_keeps_development_outcomes_unread(tmp_path):
    result_root = tmp_path / "guard-results"
    development_path = tmp_path / "guard-development.jsonl"
    identifier_index_path = tmp_path / "historical-identifiers.json"
    repository = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        "scripts/create_share_guard_development_data.py",
        "--episodes", "datasets/share_synthetic/episodes.jsonl",
        "--out", str(development_path),
        "--metadata-out", str(identifier_index_path),
    ], cwd=repository, check=True)
    command = [
        sys.executable,
        "scripts/run_share_guard_experiments.py",
        "--episodes", str(development_path),
        "--out", str(result_root),
        "--sanity",
        "--bootstrap-samples", "10",
    ]
    subprocess.run(command, cwd=repository, check=True)
    subprocess.run([*command, "--resume"], cwd=repository, check=True)
    result = json.loads((result_root / "results.json").read_text())
    assert len(result["runs"]) == 3
    assert result["development_diagnostics"] == {}
    assert result["primary_development_protocol"] is False
    assert result["partitions"]["development_validation"]["episodes"] == 0
    assert result["partitions"]["development_test"]["episodes"] == 0


def test_guard_development_partitions_reject_historical_branches():
    train = sample_episode(split="train", group="train", episode_id="train")
    historical = sample_episode(split="validation", group="historical", episode_id="historical")
    with pytest.raises(ValueError, match="only train episodes"):
        guard_development_partitions([train, historical])


def test_guard_selection_summary_reports_foldwise_failure_modes():
    config = {"human_operating_target": 0.15, "unsafe_operating_target": 0.03, "folds": 3, "seed": 17}
    result = {
        "evidence_warning": "synthetic only",
        "runs": [
            {
                "method": "guard",
                "seed": 17,
                "status": "no_stable_candidate",
                "selected_candidate": None,
                "selection": {
                    "config": config,
                    "candidates": [
                        {
                            "candidate": "both",
                            "fold_metrics": [
                                {"hir": 0.2, "unsafe_autonomy_cost_rate": 0.04, "episodes": 2.0},
                                {"hir": 0.1, "unsafe_autonomy_cost_rate": 0.01, "episodes": 2.0},
                            ],
                            "mean_metrics": {"hir": 0.15, "unsafe_autonomy_cost_rate": 0.025, "episodes": 4.0},
                        },
                        {
                            "candidate": "human",
                            "fold_metrics": [
                                {"hir": 0.2, "unsafe_autonomy_cost_rate": 0.01, "episodes": 2.0},
                                {"hir": 0.1, "unsafe_autonomy_cost_rate": 0.01, "episodes": 2.0},
                            ],
                            "mean_metrics": {"hir": 0.15, "unsafe_autonomy_cost_rate": 0.01, "episodes": 4.0},
                        },
                    ],
                },
            }
        ],
    }
    summary = summarize_runs(result)
    assert summary["screen_status"] == "no_stable_candidate"
    assert summary["failure_mode_counts"] == {"human_and_unsafe": 1, "human_only": 1}
    assert summary["runs"][0]["least_violation_candidate"]["candidate"] == "human"
    assert constraint_ratio(0.0, 0.0) == 0.0
    assert constraint_ratio(0.01, 0.0) == MAX_VIOLATION_RATIO


def test_guard_selection_summary_counts_stop_ceiling_failure():
    config = {"human_operating_target": 0.2, "unsafe_operating_target": 0.05, "max_stopped_rate": 0.5, "folds": 3, "seed": 17}
    result = {
        "runs": [{
            "method": "guard", "seed": 17, "status": "no_stable_candidate", "selected_candidate": None,
            "selection": {"config": config, "candidates": [{
                "candidate": "stop_only", "stable": False,
                "fold_metrics": [{"hir": 0.1, "unsafe_autonomy_cost_rate": 0.02, "stopped_rate": 0.55, "episodes": 2.0}],
                "mean_metrics": {"hir": 0.1, "unsafe_autonomy_cost_rate": 0.02, "stopped_rate": 0.55, "episodes": 2.0},
            }]},
        }],
    }
    summary = summarize_runs(result)
    candidate = summary["runs"][0]["candidates"][0]
    assert summary["failure_mode_counts"] == {"stop_only": 1}
    assert candidate["stable"] is False
    assert candidate["max_fold_stopped_rate"] == pytest.approx(0.55)
