from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNConfig, DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an independently trained SHARE Q ensemble.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_ensemble/result.json"))
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--constraint-confidence-z", type=float, default=0.0)
    args = parser.parse_args()
    encoder = FeatureEncoder.load(args.runs / "feature_encoder.json")
    checkpoints = sorted(args.runs.glob("share_controller_budget_*_seed_*.pt"))
    if len(checkpoints) < 2:
        raise SystemExit("at least two SHARE checkpoints are required")
    members = [DQNPolicy.load(str(path), encoder, trusted_checkpoint=True) for path in checkpoints]
    config = DQNConfig(
        human_budget=args.human_budget,
        unsafe_budget=args.unsafe_budget,
        constraint_confidence_z=args.constraint_confidence_z,
        training_epochs=0,
    )
    policy = DQNEnsemblePolicy(members, encoder, config)
    episodes = load_episodes(args.episodes)
    validation = [episode for episode in episodes if episode.split == "validation"]
    test = [episode for episode in episodes if episode.split == "test"]
    policy.calibrate_constraints(validation, args.human_budget, args.unsafe_budget)
    result = evaluate_policy(policy, test, human_budget=args.human_budget, unsafe_budget=args.unsafe_budget)
    result.update({
        "members": [path.name for path in checkpoints],
        "human_calibration_penalty": policy.human_calibration_penalty,
        "ai_calibration_penalty": policy.ai_calibration_penalty,
        "calibration_human_target": policy.calibration_human_target,
        "calibration_unsafe_target": policy.calibration_unsafe_target,
        "calibration_validation_hir": policy.calibration_validation_hir,
        "calibration_validation_unsafe": policy.calibration_validation_unsafe,
        "synthetic_evidence": all(episode.synthetic for episode in test),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
