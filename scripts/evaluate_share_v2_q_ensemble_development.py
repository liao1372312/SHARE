from __future__ import annotations

"""Evaluate a fixed five-member SHARE-V2 Q ensemble on development data only."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import (
    CHECKPOINTS,
    DEVELOPMENT_EPISODES_SHA256,
    ENCODER,
    EXPECTED_SPLIT_COUNTS,
    METRICS,
    SEEDS,
    atomic_json,
    sha256_file,
)
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RepairEpisode, RewardConfig


def evaluate(
    policy: DQNEnsemblePolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
) -> dict[str, Any]:
    cap = math.floor(0.20 * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, reward, 0.20, None)
    result = {key: value for key, value in raw.items() if key != "records"}
    result.update(
        {
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "value_policy_replans": wrapper.value_policy_replans,
            "cap_respected": wrapper.human_interventions <= cap,
        }
    )
    if not result["cap_respected"]:
        raise RuntimeError("online Human hard cap was violated")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/share_v2_q_ensemble_development_20260827"),
    )
    args = parser.parse_args()
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run on a non-allowlisted development dataset")
    episodes = load_episodes(args.episodes)
    split = {
        name: [episode for episode in episodes if episode.split == name]
        for name in ("train", "validation", "test")
    }
    counts = {name: len(values) for name, values in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected split counts: {counts}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "FINAL_RESULTS.json").exists():
        raise SystemExit("refusing to rerun a completed ensemble evaluation")

    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    members = [
        DQNPolicy.load(
            str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True
        )
        for seed in SEEDS
    ]
    ensemble = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    ensemble.name = "share_v2_q_ensemble"

    manifest = {
        "protocol": "share-v2-five-member-q-ensemble-development-v1",
        "evidence_class": "post_locked_development_candidate_only",
        "locked_holdout_accessed": False,
        "selection_rule": "fixed arithmetic mean of five Q networks; validation-only human-budget calibration",
        "development_test_used_for_selection": False,
        "deployment_cost_disclosure": "five Q-network forward passes per decision",
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "seeds": list(SEEDS),
        "training_reward": asdict(training_reward),
        "evaluation_reward": asdict(evaluation_reward),
        "encoder_sha256": sha256_file(ENCODER),
        "checkpoint_sha256": {str(seed): sha256_file(path) for seed, path in CHECKPOINTS.items()},
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.out / "pre_evaluation_manifest.json", manifest)

    # Calibration sees validation only. Unsafe autonomy remains reward-side,
    # matching SHARE-V2's human-only constraint formulation.
    ensemble.calibrate_constraints(split["validation"], 0.20, 0.05)
    validation = evaluate(ensemble, split["validation"], evaluation_reward)
    selected = (
        validation["hir"] <= 0.20 + 1e-12
        and validation["uar"] <= 0.12 + 1e-12
        and validation["stopped_rate"] <= 0.05 + 1e-12
    )
    development = evaluate(ensemble, split["test"], evaluation_reward)
    result = {
        "protocol": manifest["protocol"],
        "locked_holdout_accessed": False,
        "validation_gate_passed": selected,
        "calibration": {
            "human_penalty": ensemble.human_calibration_penalty,
            "ai_penalty": ensemble.ai_calibration_penalty,
            "member_human_penalty_mean": mean(
                member.human_calibration_penalty for member in members
            ),
        },
        "validation": validation,
        "development_test_diagnostic": development,
        "interpretation_boundary": "Development-only post-hoc redesign; confirmation requires a new untouched holdout.",
    }
    atomic_json(args.out / "FINAL_RESULTS.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
