"""Train matched-update unpaired variants on development data only.

Full SHARE and Myopic already have frozen 30,000-update paired checkpoints.
This script gives w/o-Paired and DDQN exactly 1,000 updates per epoch for
30 epochs, matching their update budget while retaining behavior-selected
replay. The DDQN row also removes the training Human-cost regularizer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.run_share_v2_development_multistart import config_for
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.features import FeatureEncoder
from share.types import RepairEpisode, RewardConfig


DEVELOPMENT = ROOT / "datasets/share_synthetic/episodes.jsonl"
HOLDOUT = ROOT / "datasets/share_component_confirmation_232_20260930/episodes.jsonl"
ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
OUT = ROOT / "results/share_component_confirmation_e2_232_20260930"
EXPECTED_DEVELOPMENT_SHA = "174a2c387aa8fe4cece6374a54a9d23c5db062bde6349c71578384c386b5a694"
EXPECTED_HOLDOUT_SHA = "52e528dc385368af0329a0d9711c43fba0e9ee4827862341dd4a19f301631989"
SEEDS = (17, 29, 43, 71, 101)


class MatchedUpdateUnpairedPolicy(DQNPolicy):
    def _collect_exploratory_replay(self, episodes: list[RepairEpisode], rng: random.Random) -> None:
        start = self.gradient_steps
        super()._collect_exploratory_replay(episodes, rng)
        if self.gradient_steps - start > self.config.updates_per_epoch:
            raise RuntimeError("behavior-selected rollout exceeded the matched update budget")
        while self.gradient_steps - start < self.config.updates_per_epoch:
            self._optimize()
            self.decision_steps += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("wo_paired", "ddqn"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if sha256_file(DEVELOPMENT) != EXPECTED_DEVELOPMENT_SHA:
        raise SystemExit("development dataset changed")
    if sha256_file(HOLDOUT) != EXPECTED_HOLDOUT_SHA:
        raise SystemExit("sealed 232-group holdout changed")
    episodes = load_episodes(DEVELOPMENT)
    train = [ep for ep in episodes if ep.split == "train"]
    validation = [ep for ep in episodes if ep.split == "validation"]
    if (len(train), len(validation)) != (542, 127):
        raise SystemExit("unexpected development split sizes")
    encoder = FeatureEncoder.load(ENCODER)
    base = config_for(args.seed, "paired_e30", "cuda:0")
    common = {"paired_replay": False}
    if args.variant == "ddqn":
        common.update(constrained=False, human_constraint=False)
    config = replace(base, **common)
    reward = RewardConfig(rollback_penalty=0.60)
    path = OUT / "checkpoints" / f"{args.variant}_seed_{args.seed}.pt"
    audit_path = OUT / "training_audit" / f"{args.variant}_seed_{args.seed}.json"
    if path.exists() or audit_path.exists():
        raise SystemExit(f"refusing to overwrite {path} or {audit_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    policy = MatchedUpdateUnpairedPolicy(encoder, config, reward)
    policy.name = f"matched_e2_{args.variant}"
    policy.fit(train, validation)
    if policy.gradient_steps != 30_000:
        raise RuntimeError(f"expected 30,000 gradient steps, found {policy.gradient_steps}")
    policy.save(str(path))
    audit = {
        "protocol": "share-matched-e2-232-v1",
        "variant": args.variant,
        "seed": args.seed,
        "training_episodes": len(train),
        "validation_episodes": len(validation),
        "development_sha256": EXPECTED_DEVELOPMENT_SHA,
        "sealed_holdout_sha256": EXPECTED_HOLDOUT_SHA,
        "encoder_sha256": sha256_file(ENCODER),
        "checkpoint_sha256": sha256_file(path),
        "config": asdict(config),
        "training_reward": asdict(reward),
        "gradient_steps": policy.gradient_steps,
        "selected_checkpoint_diagnostics": policy.selected_checkpoint_diagnostics,
        "holdout_outcomes_read": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "seed": args.seed, "gradient_steps": policy.gradient_steps, "checkpoint": str(path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
