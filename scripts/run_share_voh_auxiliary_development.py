from __future__ import annotations

"""Development-only experiment for a decision-active pairwise VoH head.

The frozen SHARE-V2 DDQN ensemble supplies sequential AI/Stop values.  A
supervised Extra-Trees head estimates the paired downstream advantage of Human
over an autonomous AI continuation.  Its prediction replaces the Human--AI Q
gap, so the head affects actions rather than merely providing a post-hoc score.

Model complexity and the Human threshold are selected on validation only.  The
already inspected development-test split is diagnostic and cannot support a
confirmatory paper claim.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_v2_development_multistart import (
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
    atomic_json,
    sha256_file,
    summarize_without_records,
)
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy, deterministic_action
from share.environment import base_reward
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


PROTOCOL = "share-pairwise-voh-auxiliary-development-v1"
HEAD_SEEDS = (17, 29, 43, 71, 101)
LEAF_CANDIDATES = (3, 8, 16)
SAFETY_WEIGHT_CANDIDATES = (0.0, 1.0, 2.0)
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def autonomous_return(episode: RepairEpisode, step: int, reward: RewardConfig, gamma: float) -> float:
    outcome = episode.steps[step].outcome(AuthorityAction.AI)
    value = base_reward(outcome, AuthorityAction.AI, reward)
    if not outcome.terminal and outcome.next_step is not None:
        value += gamma * autonomous_return(episode, outcome.next_step, reward, gamma)
    return float(value)


def autonomous_bad(episode: RepairEpisode, step: int) -> bool:
    outcome = episode.steps[step].outcome(AuthorityAction.AI)
    bad = bool(outcome.unsafe or not outcome.success)
    if not outcome.terminal and outcome.next_step is not None:
        bad = bad or autonomous_bad(episode, outcome.next_step)
    return bad


def pairwise_rows(
    episodes: Sequence[RepairEpisode], encoder: FeatureEncoder, reward: RewardConfig, gamma: float,
    safety_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[RepairState]]:
    states: list[RepairState] = []
    targets: list[float] = []
    for episode in episodes:
        for step, paired in enumerate(episode.steps):
            human = base_reward(paired.outcome(AuthorityAction.HUMAN), AuthorityAction.HUMAN, reward)
            states.append(paired.state)
            human_outcome = paired.outcome(AuthorityAction.HUMAN)
            human_bad = bool(human_outcome.unsafe or not human_outcome.success)
            safety_benefit = float(autonomous_bad(episode, step)) - float(human_bad)
            targets.append(
                human - autonomous_return(episode, step, reward, gamma)
                + float(safety_weight) * safety_benefit
            )
    return encoder.transform(states), np.asarray(targets, dtype=np.float64), states


class PairwiseVoHPolicy(Policy):
    name = "share_pairwise_voh"

    def __init__(
        self,
        base: DQNEnsemblePolicy,
        head: ExtraTreesRegressor,
        encoder: FeatureEncoder,
        threshold: float = 0.0,
    ) -> None:
        self.base = base
        self.head = head
        self.encoder = encoder
        self.threshold = float(threshold)
        self._voh_cache: dict[tuple[str, int], float] = {}
        self._base_cache: dict[tuple[str, int], np.ndarray] = {}

    @staticmethod
    def _key(state: RepairState) -> tuple[str, int]:
        return state.episode_id, state.step

    def prime(self, states: Sequence[RepairState]) -> None:
        """Batch and cache threshold-invariant predictions for fast rollouts."""
        unseen = [state for state in states if self._key(state) not in self._voh_cache]
        if not unseen:
            return
        predictions = self.head.predict(self.encoder.transform(unseen))
        base_values = self.base.values_batch(unseen)
        for state, prediction, values in zip(unseen, predictions, base_values):
            key = self._key(state)
            self._voh_cache[key] = float(prediction)
            self._base_cache[key] = np.asarray(values, dtype=np.float64)

    def predicted_voh(self, state: RepairState) -> float:
        key = self._key(state)
        if key not in self._voh_cache:
            self.prime([state])
        return self._voh_cache[key]

    def values(self, state: RepairState) -> np.ndarray:
        key = self._key(state)
        if key not in self._base_cache:
            self.prime([state])
        values = self._base_cache[key].copy()
        # The auxiliary estimate is in reward units and directly determines
        # the Human--AI margin while the DDQN keeps the sequential AI/Stop gap.
        values[AuthorityAction.HUMAN] = (
            values[AuthorityAction.AI] + self.predicted_voh(state) - self.threshold
        )
        return values

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        # Separation of concerns: the auxiliary head only ranks Human against
        # autonomous continuation.  DDQN retains ownership of AI versus Stop.
        # If DDQN wanted Human but the auxiliary threshold rejects it (or the
        # global Human cap is exhausted), continue autonomously when feasible
        # instead of turning a Human-routing decision into an artificial Stop.
        if AuthorityAction.HUMAN in feasible and self.escalation_score(state) > 0.0:
            return AuthorityAction.HUMAN
        key = self._key(state)
        if key not in self._base_cache:
            self.prime([state])
        base_action = deterministic_action(self._base_cache[key], feasible)
        if base_action == AuthorityAction.HUMAN:
            return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP
        return base_action

    def escalation_score(self, state: RepairState) -> float:
        return self.predicted_voh(state) - self.threshold


def evaluate(policy: Policy, episodes: Sequence[RepairEpisode], reward: RewardConfig) -> dict[str, Any]:
    cap = math.floor(0.20 * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, reward, 0.20, None)
    result = summarize_without_records(raw)
    result.update(
        {
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "cap_respected": wrapper.human_interventions <= cap,
        }
    )
    if not result["cap_respected"]:
        raise RuntimeError("online Human hard cap was violated")
    return result


def aspiration_loss(metrics: Mapping[str, Any]) -> float:
    """Predeclared distance from the requested 20% operating point."""
    repair_gap = max(0.0, 0.88 - float(metrics["repair_sr"])) / 0.03
    uar_gap = max(0.0, float(metrics["uar"]) - 0.08) / 0.03
    stop_gap = max(0.0, float(metrics["stopped_rate"]) - 0.06) / 0.03
    hir_gap = max(0.0, float(metrics["hir"]) - 0.20) / 0.02
    proposal_gap = max(0.0, float(metrics.get("unconstrained_proposed_hir", 0.0)) - 0.19) / 0.02
    return repair_gap**2 + uar_gap**2 + stop_gap**2 + hir_gap**2 + 25.0 * proposal_gap**2


def select_threshold(
    policy: PairwiseVoHPolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
) -> tuple[float, dict[str, Any]]:
    policy.prime([step.state for episode in episodes for step in episode.steps])
    scores = np.asarray(
        [policy.predicted_voh(step.state) for episode in episodes for step in episode.steps]
    )
    candidates = np.unique(np.quantile(scores, np.linspace(0.45, 0.95, 41)))
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        policy.threshold = float(threshold)
        proposed = summarize_without_records(
            evaluate_policy(policy, episodes, reward, 0.20, None)
        )
        metrics = evaluate(policy, episodes, reward)
        metrics["unconstrained_proposed_hir"] = float(proposed["hir"])
        key = (
            float(proposed["hir"] <= 0.19 + 1e-12),
            -aspiration_loss(metrics),
            float(metrics["repair_sr"]),
            -float(metrics["uar"]),
            -float(metrics["stopped_rate"]),
            float(metrics["normalized_return"]),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    policy.threshold = best[1]
    return best[1], best[2]


def voh_statistics(
    policy: PairwiseVoHPolicy,
    episodes: Sequence[RepairEpisode],
    encoder: FeatureEncoder,
    reward: RewardConfig,
    gamma: float,
    seed: int,
) -> dict[str, Any]:
    x, truth, _ = pairwise_rows(episodes, encoder, reward, gamma, safety_weight=0.0)
    predicted = np.asarray(policy.head.predict(x), dtype=np.float64)
    rho = float(spearmanr(predicted, truth).statistic)
    order = np.argsort(predicted, kind="stable")
    bins = np.array_split(order, 5)
    quintiles = [float(np.mean(truth[index])) for index in bins]
    rng = np.random.default_rng(20260829 + seed)
    boot = []
    for _ in range(5000):
        index = rng.integers(0, len(truth), len(truth))
        value = spearmanr(predicted[index], truth[index]).statistic
        if np.isfinite(value):
            boot.append(float(value))
    return {
        "rho": rho,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "quintile_true_voh": quintiles,
        "strictly_monotone": all(left < right for left, right in zip(quintiles, quintiles[1:])),
        "states": len(truth),
    }


def aggregate(rows: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[section][metric]) for row in rows]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        result[f"{metric}_min"] = min(values)
        result[f"{metric}_max"] = max(values)
    rhos = [float(row[f"{section}_voh"]["rho"]) for row in rows]
    result["voh_rho_mean"] = mean(rhos)
    result["voh_rho_std"] = stdev(rhos) if len(rhos) > 1 else 0.0
    result["all_caps_respected"] = all(bool(row[section]["cap_respected"]) for row in rows)
    result["all_quintiles_monotone"] = all(bool(row[f"{section}_voh"]["strictly_monotone"]) for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_voh_auxiliary_development_20260829"))
    args = parser.parse_args()
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run on a non-allowlisted development dataset")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite a prior experiment ledger")
    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in EXPECTED_SPLIT_COUNTS}
    if {key: len(value) for key, value in split.items()} != EXPECTED_SPLIT_COUNTS:
        raise SystemExit("unexpected development split counts")
    args.out.mkdir(parents=True, exist_ok=False)

    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    members = [
        DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True)
        for seed in SEEDS
    ]
    base = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    base.calibrate_constraints(split["validation"], 0.20, 0.05)
    x_train, _, _ = pairwise_rows(split["train"], encoder, evaluation_reward, 0.95)

    manifest = {
        "protocol": PROTOCOL,
        "evidence_class": "development_candidate_only",
        "locked_holdout_accessed": False,
        "selection_split": "validation_only",
        "development_test_used_for_selection": False,
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "base_checkpoint_sha256": {str(seed): sha256_file(path) for seed, path in CHECKPOINTS.items()},
        "head_seeds": list(HEAD_SEEDS),
        "leaf_candidates": list(LEAF_CANDIDATES),
        "safety_weight_candidates": list(SAFETY_WEIGHT_CANDIDATES),
        "head": "ExtraTreesRegressor(n_estimators=400,max_features=0.8,bootstrap=True)",
        "target": "reward VoH plus a validation-selected autonomous-bad reduction term",
        "selection": "unconstrained proposed HIR<=.19; then minimum validation aspiration loss, Repair.SR, UAR, Stop, Return",
        "aspiration": {"repair_sr_min": 0.88, "uar_max": 0.08, "stopped_rate_max": 0.06, "hir_max": 0.20},
        "source_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.out / "manifest.json", manifest)

    # Complexity is chosen once with seed 17 on validation.  Other seeds are
    # repetitions of the fixed selected configuration, not additional tuning.
    pilot_rows = []
    for safety_weight in SAFETY_WEIGHT_CANDIDATES:
        _, y_train, _ = pairwise_rows(
            split["train"], encoder, evaluation_reward, 0.95, safety_weight=safety_weight
        )
        for leaf in LEAF_CANDIDATES:
            head = ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=leaf,
                max_features=0.8,
                bootstrap=True,
                n_jobs=-1,
                random_state=HEAD_SEEDS[0],
            ).fit(x_train, y_train)
            policy = PairwiseVoHPolicy(base, head, encoder)
            threshold, validation = select_threshold(policy, split["validation"], evaluation_reward)
            pilot_rows.append({
                "min_samples_leaf": leaf,
                "safety_weight": safety_weight,
                "threshold": threshold,
                "validation": validation,
            })
    selected = min(
        pilot_rows,
        key=lambda row: (
            float(row["validation"].get("unconstrained_proposed_hir", 1.0) > 0.19 + 1e-12),
            aspiration_loss(row["validation"]),
            -float(row["validation"]["repair_sr"]),
            float(row["validation"]["uar"]),
            float(row["validation"]["stopped_rate"]),
        ),
    )
    atomic_json(args.out / "pilot_validation.json", {"candidates": pilot_rows, "selected": selected})

    rows = []
    leaf = int(selected["min_samples_leaf"])
    safety_weight = float(selected["safety_weight"])
    _, y_train, _ = pairwise_rows(
        split["train"], encoder, evaluation_reward, 0.95, safety_weight=safety_weight
    )
    for seed in HEAD_SEEDS:
        head = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=leaf,
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        ).fit(x_train, y_train)
        policy = PairwiseVoHPolicy(base, head, encoder)
        threshold, validation = select_threshold(policy, split["validation"], evaluation_reward)
        development_test = evaluate(policy, split["test"], evaluation_reward)
        row = {
            "seed": seed,
            "min_samples_leaf": leaf,
            "safety_weight": safety_weight,
            "threshold": threshold,
            "validation": validation,
            "validation_voh": voh_statistics(policy, split["validation"], encoder, evaluation_reward, 0.95, seed),
            "development_test": development_test,
            "development_test_voh": voh_statistics(policy, split["test"], encoder, evaluation_reward, 0.95, seed),
        }
        model_path = args.out / f"voh_head_seed_{seed}.joblib"
        joblib.dump(
            {"model": head, "threshold": threshold, "min_samples_leaf": leaf, "safety_weight": safety_weight},
            model_path,
        )
        row["model_sha256"] = sha256_file(model_path)
        atomic_json(args.out / f"seed_{seed}.json", row)
        rows.append(row)

    summary = {
        "selected_min_samples_leaf": leaf,
        "selected_safety_weight": safety_weight,
        "validation": aggregate(rows, "validation"),
        "development_test_diagnostic": aggregate(rows, "development_test"),
        "interpretation_boundary": "Development-only evidence. Confirmation requires a new untouched workflow-group-disjoint holdout.",
    }
    atomic_json(args.out / "FINAL_RESULTS.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
