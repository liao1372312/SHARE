"""Post-confirmation leave-one-member-out and VoH diagnostics for SHARE V4."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_voh_auxiliary_development import pairwise_rows
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig

DATASET = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
OUT = ROOT / "results/share_v4_ensemble_stability_posthoc_20260829"
HEAD = ROOT / "results/share_voh_auxiliary_safety_aware_20260829/voh_head_seed_17.joblib"
HUMAN_PENALTY = 1.9403412342071533
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def run(policy: DQNEnsemblePolicy, episodes, reward: RewardConfig) -> dict:
    wrapper = OnlineHardHumanCapPolicy(policy, math.floor(0.20 * len(episodes)), False)
    raw = evaluate_policy(wrapper, episodes, reward, 0.20, None)
    result = {key: value for key, value in raw.items() if key != "records"}
    result["attempted_human_after_exhaustion"] = wrapper.attempted_human_after_exhaustion
    return result


def validity(predicted: np.ndarray, truth: np.ndarray, seed: int) -> dict:
    rho = float(spearmanr(predicted, truth).statistic)
    order = np.argsort(predicted, kind="stable")
    quintiles = [float(truth[index].mean()) for index in np.array_split(order, 5)]
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(10_000):
        index = rng.integers(0, len(truth), len(truth))
        value = spearmanr(predicted[index], truth[index]).statistic
        if np.isfinite(value):
            bootstrap.append(float(value))
    return {
        "rho": rho,
        "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "quintile_true_voh": quintiles,
        "strictly_monotone": all(left < right for left, right in zip(quintiles, quintiles[1:])),
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("post-hoc stability analysis already exists")
    OUT.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(DATASET)
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    members = {
        seed: DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True)
        for seed in SEEDS
    }
    rows = []
    for omitted in SEEDS:
        policy = DQNEnsemblePolicy(
            [member for seed, member in members.items() if seed != omitted],
            encoder,
            members[SEEDS[0]].config,
            training_reward,
        )
        policy.name = f"share_q_ensemble_leave_out_{omitted}"
        policy.human_calibration_penalty = HUMAN_PENALTY
        rows.append({"omitted_member_seed": omitted, **run(policy, episodes, evaluation_reward)})
    aggregate = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate[f"{metric}_mean"] = mean(values)
        aggregate[f"{metric}_std"] = stdev(values)
        aggregate[f"{metric}_min"] = min(values)
        aggregate[f"{metric}_max"] = max(values)

    full = DQNEnsemblePolicy(list(members.values()), encoder, members[SEEDS[0]].config, training_reward)
    states = [step.state for episode in episodes for step in episode.steps]
    values = full._raw_values_batch(states)
    q_score = values[:, AuthorityAction.HUMAN] - values[:, AuthorityAction.AI]
    _, truth, _ = pairwise_rows(episodes, encoder, evaluation_reward, 0.95)
    head = joblib.load(HEAD)["model"]
    auxiliary_score = np.asarray(head.predict(encoder.transform(states)), dtype=np.float64)
    result = {
        "evidence_class": "post-confirmation robustness and mechanism analysis; not model selection",
        "leave_one_member_out": rows,
        "leave_one_member_out_aggregate": aggregate,
        "q_ensemble_voh": validity(q_score, truth, 20260829),
        "development_frozen_pairwise_head_voh": validity(auxiliary_score, truth, 20260830),
    }
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "leave_one_member_out_aggregate": aggregate,
        "q_ensemble_voh": result["q_ensemble_voh"],
        "pairwise_head_voh": result["development_frozen_pairwise_head_voh"],
    }, indent=2))


if __name__ == "__main__":
    main()
