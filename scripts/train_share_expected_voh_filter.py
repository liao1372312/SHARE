"""Train and freeze the public-API expected-VoH filter before V5 evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS, sha256_file
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RewardConfig
from share.voh import ExpectedVoHFilterPolicy, expected_voh

OUT = ROOT / "results/share_expected_voh_filter_development_20260829"
TRAIN = (
    ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl",
    ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl",
    ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl",
)
VALIDATION = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
HUMAN_PENALTY = 1.9403412342071533


def compact(raw):
    return {key: value for key, value in raw.items() if key != "records"}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("expected-VoH development ledger already exists")
    OUT.mkdir(parents=True, exist_ok=False)
    encoder = FeatureEncoder.load(ENCODER)
    reward = RewardConfig()
    training = [episode for path in TRAIN for episode in load_episodes(path)]
    validation = load_episodes(VALIDATION)
    train_states = [step.state for episode in training for step in episode.steps]
    validation_states = [step.state for episode in validation for step in episode.steps]
    head = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=16, max_features=0.8,
        bootstrap=True, n_jobs=-1, random_state=17,
    ).fit(encoder.transform(train_states), expected_voh(training, reward))
    predicted = np.asarray(head.predict(encoder.transform(validation_states)), dtype=np.float64)
    truth = expected_voh(validation, reward)

    training_reward = RewardConfig(rollback_penalty=0.60)
    members = [DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True) for seed in SEEDS]
    base = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    base.human_calibration_penalty = HUMAN_PENALTY
    policy = ExpectedVoHFilterPolicy(base, head, encoder, 0.0)
    policy.prime(validation_states)
    candidates = np.unique(np.quantile(predicted, np.linspace(0.0, 0.95, 40)))
    rows = []
    for threshold in candidates:
        policy.threshold = float(threshold)
        raw = compact(evaluate_policy(policy, validation, reward, 0.20, None))
        rows.append({"threshold": float(threshold), **raw})
    feasible = [row for row in rows if 0.18 <= float(row["hir"]) <= 0.20]
    selected = max(
        feasible,
        key=lambda row: (
            float(row["repair_sr"]), -float(row["uar"]), -float(row["stopped_rate"]),
            float(row["normalized_return"]),
        ),
    )
    artifact = OUT / "expected_voh_filter.joblib"
    joblib.dump({"head": head, "threshold": selected["threshold"], "human_penalty": HUMAN_PENALTY}, artifact)
    result = {
        "evidence_class": "development and V4-validation only; V5 remains untouched",
        "training_episodes": len(training),
        "validation_episodes": len(validation),
        "head_config": {"n_estimators": 500, "min_samples_leaf": 16, "max_features": 0.8, "bootstrap": True, "random_state": 17},
        "expected_voh_validation": {"rho": float(spearmanr(predicted, truth).statistic)},
        "selected_operating_point": selected,
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_sha256": sha256_file(artifact),
        "training_dataset_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in TRAIN},
        "validation_dataset_sha256": sha256_file(VALIDATION),
    }
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
