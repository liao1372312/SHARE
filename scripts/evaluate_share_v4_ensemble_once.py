"""Evaluate the frozen five-member SHARE-V2 Q ensemble exactly once."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS, sha256_file
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RewardConfig

DATASET = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
OUT = ROOT / "results/share_v4_ensemble_confirmation_20260829"
HUMAN_PENALTY = 1.9403412342071533


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("V4 ensemble confirmation already started")
    OUT.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(DATASET)
    if len(episodes) != 250 or len({episode.group_id for episode in episodes}) != 250:
        raise SystemExit("unexpected V4 dataset size")
    manifest = {
        "protocol": "share-v4-frozen-five-member-q-ensemble-once-v1",
        "candidate_frozen_before_evaluation": True,
        "selection_evidence": "development-only five-member arithmetic Q ensemble; fixed validation calibration",
        "dataset_sha256": sha256_file(DATASET),
        "episodes": 250,
        "hard_human_cap": 50,
        "members": list(SEEDS),
        "checkpoint_sha256": {str(seed): sha256_file(path) for seed, path in CHECKPOINTS.items()},
        "encoder_sha256": sha256_file(ENCODER),
        "human_calibration_penalty": HUMAN_PENALTY,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "ensemble_implementation_sha256": sha256_file(ROOT / "share/dqn.py"),
    }
    (OUT / "pre_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    members = [DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True) for seed in SEEDS]
    policy = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    policy.name = "share_v4_q_ensemble"
    policy.human_calibration_penalty = HUMAN_PENALTY
    wrapper = OnlineHardHumanCapPolicy(policy, math.floor(0.20 * len(episodes)), binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), 0.20, None)
    result = {key: value for key, value in raw.items() if key != "records"}
    result.update({
        "human_interventions": wrapper.human_interventions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "hard_human_cap": wrapper.cap,
        "cap_respected": wrapper.human_interventions <= wrapper.cap,
        "evidence_class": "new_workflow-group-disjoint replayable synthetic confirmation",
    })
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret", "attempted_human_after_exhaustion", "cap_respected")}, indent=2))


if __name__ == "__main__":
    main()
