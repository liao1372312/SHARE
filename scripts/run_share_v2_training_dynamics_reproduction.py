from __future__ import annotations

"""Reproduce final SHARE-V2 training dynamics on development data only.

This is a post-hoc reproducibility diagnostic. It uses the frozen V2 training
configuration and the allowlisted development train/validation split, never
loads the development-test or independent workflow holdout for evaluation,
and does not modify the frozen checkpoints.
"""

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_v2_development_multistart import (
    DEFAULT_SEEDS,
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
    atomic_json,
    config_for,
    sha256_file,
)
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.features import FeatureEncoder
from share.types import RewardConfig


PROTOCOL_VERSION = "share-v2-training-dynamics-reproduction-v1"
ORIGINAL_CHECKPOINTS = {
    17: Path("results/share_v2_stop_penalty_tuning_20260827/p060/share_v2_seed_17.pt"),
    29: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_29.pt"),
    43: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_43.pt"),
    71: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_71.pt"),
    101: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_101.pt"),
}
HISTORY_FIELDS = (
    "epoch",
    "phase",
    "episodes",
    "decision_states",
    "repair_sr",
    "hir",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "p_ai",
    "p_human",
    "p_stop",
    "beta_h",
    "beta_u",
)


def state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key].cpu(), right[key].cpu()) for key in left)


def checkpoint_equivalence(reproduced: Path, original: Path) -> dict[str, Any]:
    new = torch.load(reproduced, map_location="cpu", weights_only=True)
    old = torch.load(original, map_location="cpu", weights_only=True)
    scalar_keys = (
        "beta_h",
        "beta_u",
        "human_calibration_penalty",
        "ai_calibration_penalty",
        "calibration_validation_hir",
        "calibration_validation_unsafe",
    )
    scalar_equal = {key: float(new[key]) == float(old[key]) for key in scalar_keys}
    return {
        "original": str(original),
        "original_sha256": sha256_file(original),
        "reproduced_sha256": sha256_file(reproduced),
        "model_tensors_exact": state_dict_equal(new["model"], old["model"]),
        "config_exact": new["config"] == old["config"],
        "reward_config_exact": new["reward_config"] == old["reward_config"],
        "calibration_scalars_exact": scalar_equal,
        "all_semantic_fields_exact": (
            state_dict_equal(new["model"], old["model"])
            and new["config"] == old["config"]
            and new["reward_config"] == old["reward_config"]
            and all(scalar_equal.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run: episodes are not the allowlisted development dataset")
    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    counts = {name: len(items) for name, items in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected development split counts: {counts}")

    # The test objects are counted only to verify the frozen split manifest; no
    # test episode is passed to fitting, diagnostics, selection, or calibration.
    encoder = FeatureEncoder.fit([step.state for episode in split["train"] for step in episode.steps])
    training_reward = RewardConfig(rollback_penalty=0.60)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    encoder.save(args.out / "feature_encoder.json")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "posthoc_development_training_reproducibility_diagnostic",
        "eligible_for_independent_or_locked_claim": False,
        "independent_holdout_accessed": False,
        "development_test_evaluated": False,
        "diagnostic_split": "validation",
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "seeds": list(DEFAULT_SEEDS),
        "training_variant": "paired_e30",
        "training_reward": asdict(training_reward),
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": {
            "scripts/run_share_v2_training_dynamics_reproduction.py": sha256_file(Path(__file__)),
            "scripts/run_share_v2_development_multistart.py": sha256_file(ROOT / "scripts/run_share_v2_development_multistart.py"),
            "share/dqn.py": sha256_file(ROOT / "share/dqn.py"),
        },
        "original_checkpoint_sha256": {
            str(seed): sha256_file(ROOT / path) for seed, path in ORIGINAL_CHECKPOINTS.items()
        },
    }
    atomic_json(args.out / "manifest.json", manifest)

    all_rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for seed in DEFAULT_SEEDS:
        print(json.dumps({"seed": seed, "status": "begin"}), flush=True)
        config = config_for(seed, "paired_e30", args.device)
        policy = DQNPolicy(encoder, config, training_reward)
        policy.name = "share_v2"
        policy.fit(split["train"], split["validation"])
        history = [dict(row) for row in policy.training_history]
        selected = dict(policy.selected_checkpoint_diagnostics or {})
        policy.calibrate_constraints(split["validation"], 0.20, 0.05)
        calibrated = dict(policy.policy_diagnostics(split["validation"], epoch=31, phase="post_calibration"))

        checkpoint = checkpoints / f"share_v2_reproduced_seed_{seed}.pt"
        policy.save(str(checkpoint))
        equivalence = checkpoint_equivalence(checkpoint, ROOT / ORIGINAL_CHECKPOINTS[seed])
        audits[str(seed)] = equivalence
        payload = {
            "seed": seed,
            "config": asdict(config),
            "training_reward": asdict(training_reward),
            "training_history": history,
            "selected_checkpoint_pre_calibration": selected,
            "post_calibration": calibrated,
            "checkpoint": str(checkpoint.relative_to(args.out)),
            "checkpoint_equivalence": equivalence,
        }
        atomic_json(args.out / f"seed_{seed}.json", payload)
        for row in history:
            all_rows.append({"seed": seed, **{field: row.get(field) for field in HISTORY_FIELDS}})
        print(json.dumps({"seed": seed, "status": "complete", "exact": equivalence["all_semantic_fields_exact"]}), flush=True)

    with (args.out / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed",) + HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    final_audit = {
        "all_five_semantically_exact": all(item["all_semantic_fields_exact"] for item in audits.values()),
        "per_seed": audits,
        "history_rows": len(all_rows),
        "expected_history_rows": len(DEFAULT_SEEDS) * 30,
        "no_test_or_holdout_evaluation": True,
    }
    atomic_json(args.out / "reproduction_audit.json", final_audit)
    print(json.dumps(final_audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
