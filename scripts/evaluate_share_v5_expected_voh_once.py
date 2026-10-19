"""One-shot V5 evaluation of the frozen Q-ensemble plus expected-VoH filter."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS, sha256_file
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_voh_auxiliary_development import pairwise_rows
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RewardConfig
from share.voh import ExpectedVoHFilterPolicy, expected_voh

DATASET = ROOT / "datasets/share_v5_expected_voh_holdout_20260829/episodes.jsonl"
ARTIFACT = ROOT / "results/share_expected_voh_filter_development_20260829/expected_voh_filter.joblib"
OUT = ROOT / "results/share_v5_expected_voh_confirmation_20260829"


def validity(score, truth, seed):
    rho = float(spearmanr(score, truth).statistic)
    order = np.argsort(score, kind="stable")
    quintiles = [float(truth[index].mean()) for index in np.array_split(order, 5)]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(10_000):
        index = rng.integers(0, len(truth), len(truth))
        boot.append(float(spearmanr(score[index], truth[index]).statistic))
    return {"rho": rho, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "quintiles": quintiles, "strictly_monotone": all(a < b for a, b in zip(quintiles, quintiles[1:]))}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("V5 confirmation already started")
    OUT.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(DATASET)
    if len(episodes) != 150:
        raise SystemExit("unexpected V5 size")
    payload = joblib.load(ARTIFACT)
    manifest = {
        "protocol": "share-v5-frozen-expected-voh-filter-once-v1",
        "candidate_frozen_before_evaluation": True,
        "dataset_sha256": sha256_file(DATASET), "artifact_sha256": sha256_file(ARTIFACT),
        "evaluator_sha256": sha256_file(Path(__file__)), "episodes": 150, "hard_human_cap": 30,
    }
    (OUT / "pre_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    members = [DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True) for seed in SEEDS]
    base = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    base.human_calibration_penalty = float(payload["human_penalty"])
    policy = ExpectedVoHFilterPolicy(base, payload["head"], encoder, float(payload["threshold"]))
    states = [step.state for episode in episodes for step in episode.steps]
    policy.prime(states)
    wrapper = OnlineHardHumanCapPolicy(policy, 30, False)
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), 0.20, None)
    result = {key: value for key, value in raw.items() if key != "records"}
    score = np.asarray(payload["head"].predict(encoder.transform(states)), dtype=np.float64)
    _, realized, _ = pairwise_rows(episodes, encoder, RewardConfig(), 0.95)
    result.update({
        "human_interventions": wrapper.human_interventions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "hard_human_cap": 30, "cap_respected": wrapper.human_interventions <= 30,
        "expected_voh_validity": validity(score, expected_voh(episodes, RewardConfig()), 20260829),
        "realized_voh_sensitivity": validity(score, realized, 20260830),
        "evidence_class": "new workflow-group-disjoint replayable synthetic confirmation",
    })
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret", "attempted_human_after_exhaustion", "expected_voh_validity", "realized_voh_sensitivity")}, indent=2))


if __name__ == "__main__":
    main()
