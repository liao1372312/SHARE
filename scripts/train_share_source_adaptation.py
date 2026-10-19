"""Train paired SHARE on the frozen mixed Flare/public-API development set."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.run_share_v2_development_multistart import config_for
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.features import FeatureEncoder
from share.types import RewardConfig


DATA = ROOT / "datasets/share_source_adaptation_20260930"
OUT = ROOT / "results/share_source_adaptation_20260930"
TRAIN_SHA = "307d294386304da8d0e6870168cff044424009fc4874458cc537332770cc72bc"
VALIDATION_SHA = "9498f326340fa6b526d20ef09111942f2a2b2f214f89bd61b3c9ccaab0d7eb5d"
SEEDS = (17, 29, 43, 71, 101)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if sha256_file(DATA / "train.jsonl") != TRAIN_SHA or sha256_file(DATA / "validation.jsonl") != VALIDATION_SHA:
        raise SystemExit("source-adaptation dataset changed")
    train, validation = load_episodes(DATA / "train.jsonl"), load_episodes(DATA / "validation.jsonl")
    if (len(train), len(validation)) != (942, 400):
        raise SystemExit("unexpected training or validation size")
    checkpoint = OUT / "checkpoints" / f"source_adapted_seed_{args.seed}.pt"
    audit_path = OUT / "training_audit" / f"seed_{args.seed}.json"
    if checkpoint.exists() or audit_path.exists():
        raise SystemExit("refusing to overwrite training artifact")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    config = config_for(args.seed, "paired_e30", "cuda:0")
    reward = RewardConfig(rollback_penalty=0.60)
    encoder_path = DATA / "feature_encoder.json"
    policy = DQNPolicy(FeatureEncoder.load(encoder_path), config, reward)
    policy.fit(train, validation)
    if policy.gradient_steps != 30_000:
        raise RuntimeError(f"expected 30,000 updates; got {policy.gradient_steps}")
    policy.save(str(checkpoint))
    audit = {
        "protocol": "share-public-api-source-adaptation-v1",
        "seed": args.seed,
        "training_episodes": len(train), "validation_episodes": len(validation),
        "train_sha256": TRAIN_SHA, "validation_sha256": VALIDATION_SHA,
        "encoder_sha256": sha256_file(encoder_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": asdict(config), "training_reward": asdict(reward),
        "gradient_steps": policy.gradient_steps,
        "selected_checkpoint_diagnostics": policy.selected_checkpoint_diagnostics,
        "gate_or_diagnostic_outcomes_read": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "gradient_steps": policy.gradient_steps, "checkpoint": str(checkpoint.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
