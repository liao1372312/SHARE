from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy, stratified_evaluation
from share.features import FeatureEncoder
from share.robustness import corrupt_human_feedback
from share.robustness import chronological_blocks
from share.types import RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHARE mechanism ablations and corrupted-human-feedback robustness.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_rq3"))
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--constraint-confidence-z", type=float, default=0.0)
    parser.add_argument("--corruption-rates", default="0.05,0.1,0.2")
    parser.add_argument("--paired-replay", action="store_true", help="Use full paired counterfactual replay as a protocol ablation.")
    parser.add_argument("--updates-per-epoch", type=int, default=1_000)
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or cuda:<index>.")
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    train = [episode for episode in episodes if episode.split == "train"]
    test = [episode for episode in episodes if episode.split == "test"]
    base_encoder = FeatureEncoder.fit([step.state for episode in train for step in episode.steps])
    reward = RewardConfig()
    rows = []
    stratified = []
    longitudinal = []
    seeds = [int(value) for value in args.seeds.split(",")]
    variants = {
        "full_share": (base_encoder, True, True),
        "without_causal_root_features": (base_encoder.without("causal"), True, True),
        "without_repair_risk_features": (base_encoder.without("risk"), True, True),
        "without_repair_history": (base_encoder.without("history"), True, True),
        "without_human_budget_constraint": (base_encoder, False, True),
        "without_safety_constraint": (base_encoder, True, False),
    }
    validation = [episode for episode in episodes if episode.split == "validation"]
    for seed in seeds:
        full_policy = None
        for name, (encoder, human_constraint, safety_constraint) in variants.items():
            config = DQNConfig(
                training_epochs=args.epochs, human_budget=args.human_budget, unsafe_budget=args.unsafe_budget,
                constrained=human_constraint or safety_constraint, seed=seed,
                human_constraint=human_constraint, unsafe_constraint=safety_constraint,
                dual_learning_rate_human=5e-3 if human_constraint else 0.0,
                dual_learning_rate_unsafe=5e-3 if safety_constraint else 0.0,
                constraint_confidence_z=args.constraint_confidence_z,
                paired_replay=args.paired_replay, updates_per_epoch=args.updates_per_epoch,
                device=args.device,
            )
            policy = DQNPolicy(encoder, config, reward)
            policy.name = name
            policy.fit(train, validation)
            policy.calibrate_constraints(validation, args.human_budget, args.unsafe_budget)
            result = evaluate_policy(policy, test, reward, args.human_budget, args.unsafe_budget)
            rows.append({**{key: value for key, value in result.items() if key != "records"}, "experiment": "ablation", "corruption_rate": 0.0, "seed": seed, "synthetic_evidence": all(item.synthetic for item in test)})
            if name == "full_share":
                full_policy = policy
        for rate in [float(value) for value in args.corruption_rates.split(",")]:
            corrupted = corrupt_human_feedback(train, rate, seed)
            config = DQNConfig(training_epochs=args.epochs, human_budget=args.human_budget, unsafe_budget=args.unsafe_budget, seed=seed, constraint_confidence_z=args.constraint_confidence_z, paired_replay=args.paired_replay, updates_per_epoch=args.updates_per_epoch, device=args.device)
            policy = DQNPolicy(base_encoder, config, reward)
            policy.name = "share_corrupted_feedback"
            policy.fit(corrupted, validation)
            policy.calibrate_constraints(validation, args.human_budget, args.unsafe_budget)
            result = evaluate_policy(policy, test, reward, args.human_budget, args.unsafe_budget)
            rows.append({**{key: value for key, value in result.items() if key != "records"}, "experiment": "robustness", "corruption_rate": rate, "seed": seed, "synthetic_evidence": all(item.synthetic for item in test)})
        if full_policy is None:
            raise RuntimeError("full_share policy was not trained")
        stratified.extend(
            {**row, "seed": seed, "synthetic_evidence": all(item.synthetic for item in test)}
            for row in stratified_evaluation(full_policy, test, reward)
        )
        blocks = chronological_blocks(test, 5)
        adaptive_policy = copy.deepcopy(full_policy)
        for index, block in enumerate(blocks, 1):
            if not block:
                continue
            frozen_result = evaluate_policy(full_policy, block, reward, args.human_budget, args.unsafe_budget)
            longitudinal.append({**{key: value for key, value in frozen_result.items() if key != "records"}, "round": index, "seed": seed, "mode": "fixed_policy_block_evaluation", "evidence_store_updated": False, "protocol_only": True, "synthetic_evidence": all(item.synthetic for item in block)})
            adaptive_result = evaluate_policy(adaptive_policy, block, reward, args.human_budget, args.unsafe_budget)
            longitudinal.append({**{key: value for key, value in adaptive_result.items() if key != "records"}, "round": index, "seed": seed, "mode": "post_episode_replay_adaptation", "training_data_available_through_round": index - 1, "protocol_only": True, "synthetic_evidence": all(item.synthetic for item in block)})
            adaptive_policy.fit(block)
            adaptive_policy.calibrate_constraints(validation, args.human_budget, args.unsafe_budget)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "stratified.json").write_text(json.dumps(stratified, indent=2), encoding="utf-8")
    (args.out / "longitudinal.json").write_text(json.dumps(longitudinal, indent=2), encoding="utf-8")
    episode_sha256 = hashlib.sha256(args.episodes.read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps({"evidence_warning": "Synthetic results are engineering evidence only and are not practitioner outcomes.", "longitudinal_warning": "The fixed-policy condition does not implement evidence-store growth; longitudinal rows are protocol checks until real timestamped evidence-store features are supplied.", "replay_protocol": "paired_counterfactual_ablation" if args.paired_replay else "exploratory_logged_transitions", "constraint_confidence_z": args.constraint_confidence_z, "synthetic_evidence": all(item.synthetic for item in test), "episode_sha256": episode_sha256, "seeds": seeds, "epochs": args.epochs, "updates_per_epoch": args.updates_per_epoch, "device": args.device}, indent=2), encoding="utf-8")
    if rows:
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "runs": len(rows), "stratified_rows": len(stratified), "longitudinal_rows": len(longitudinal)}, indent=2))


if __name__ == "__main__":
    main()
