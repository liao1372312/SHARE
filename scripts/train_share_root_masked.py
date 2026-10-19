"""Train five paired SHARE members with development root_type masked as OOV."""
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


ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
VARIANTS = {
    "root": ("share_root_masked_development_20260930", "e50f7a1e24483bdb8123d3d4587d6612ae48d1ff0b28afd88b7bfc63e89d3b56", "share-training-root-oov-alignment-v1"),
    "root_domain": ("share_root_domain_masked_development_20260930", "80a78b51996eb0e9d54c37fa23a860245e56d299889ff3c53b5e35a93fe0ce11", "share-training-root-domain-oov-alignment-v1"),
}
SEEDS = (17, 29, 43, 71, 101)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="root")
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    name, expected_dataset, protocol = VARIANTS[args.variant]
    dataset = ROOT / "datasets" / name / "episodes.jsonl"
    out = ROOT / "results" / name
    if sha256_file(dataset) != expected_dataset:
        raise SystemExit("root-masked development dataset changed")
    episodes = load_episodes(dataset)
    train = [ep for ep in episodes if ep.split == "train"]
    validation = [ep for ep in episodes if ep.split == "validation"]
    if (len(train), len(validation)) != (542, 127):
        raise SystemExit("unexpected split sizes")
    checkpoint = out / "checkpoints" / f"root_masked_seed_{args.seed}.pt"
    audit_path = out / "training_audit" / f"seed_{args.seed}.json"
    if checkpoint.exists() or audit_path.exists():
        raise SystemExit("refusing to overwrite training artifact")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    config = config_for(args.seed, "paired_e30", "cuda:0")
    reward = RewardConfig(rollback_penalty=0.60)
    policy = DQNPolicy(FeatureEncoder.load(ENCODER), config, reward)
    policy.fit(train, validation)
    if policy.gradient_steps != 30_000:
        raise RuntimeError(f"expected 30,000 updates, got {policy.gradient_steps}")
    policy.save(str(checkpoint))
    audit = {
        "protocol": protocol,
        "seed": args.seed,
        "dataset_sha256": expected_dataset,
        "encoder_sha256": sha256_file(ENCODER),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": asdict(config),
        "training_reward": asdict(reward),
        "gradient_steps": policy.gradient_steps,
        "selected_checkpoint_diagnostics": policy.selected_checkpoint_diagnostics,
        "holdout_outcomes_read": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "gradient_steps": policy.gradient_steps, "checkpoint": str(checkpoint.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
