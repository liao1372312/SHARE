from __future__ import annotations

"""Audit frozen SHARE-V2 checkpoints under the validation online hard cap.

This script is development-only.  It is allowlisted to the synthetic
train/validation/test corpus, never opens a locked holdout, and performs no
model or threshold selection.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_v2_development_multistart import (
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
)
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


SEEDS = (17, 29, 43, 71, 101)
CHECKPOINTS = {
    17: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/share_v2_seed_17.pt",
    **{
        seed: ROOT
        / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining"
        / f"share_v2_seed_{seed}.pt"
        for seed in (29, 43, 71, 101)
    },
}
ENCODER = (
    ROOT
    / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
)
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    for key in (
        "attempted_human_after_exhaustion",
        "post_exhaustion_ai",
        "post_exhaustion_stop",
        "stops_before_exhaustion",
    ):
        values = [int(row[key]) for row in rows]
        output[f"{key}_total"] = sum(values)
        output[f"{key}_mean"] = mean(values)
    output["all_caps_respected"] = all(bool(row["cap_respected"]) for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("datasets/share_synthetic/episodes.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/share_v2_validation_hard_cap_audit_20260827"),
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

    validation = split["validation"]
    cap = math.floor(0.20 * len(validation))
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "share-v2-validation-hard-cap-audit-v1",
        "purpose": "diagnosis_only_no_selection",
        "data_scope": "allowlisted development validation split only",
        "locked_holdout_accessed": False,
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "validation_order": [episode.episode_id for episode in validation],
        "human_cap": cap,
        "seeds": list(SEEDS),
        "training_reward": asdict(training_reward),
        "evaluation_reward": asdict(evaluation_reward),
        "encoder": {"path": str(ENCODER.relative_to(ROOT)), "sha256": sha256_file(ENCODER)},
        "checkpoints": {
            str(seed): {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for seed, path in CHECKPOINTS.items()
        },
    }
    atomic_json(args.out / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        policy = DQNPolicy.load(
            str(CHECKPOINTS[seed]),
            encoder,
            training_reward,
            require_reward_config=True,
        )
        policy.name = "share_v2"
        wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
        raw = evaluate_policy(wrapper, validation, evaluation_reward, 0.20, None)

        executed_after = Counter(
            int(event["executed_action"])
            for event in wrapper.events
            if event["exhausted_before_decision"]
        )
        stops_before = sum(
            int(
                event["executed_action"] == int(AuthorityAction.STOP)
                and not event["exhausted_before_decision"]
            )
            for event in wrapper.events
        )
        row = {key: value for key, value in raw.items() if key != "records"}
        row.update(
            {
                "seed": seed,
                "human_cap": cap,
                "human_interventions": wrapper.human_interventions,
                "cap_respected": wrapper.human_interventions <= cap,
                "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
                "post_exhaustion_ai": executed_after[int(AuthorityAction.AI)],
                "post_exhaustion_stop": executed_after[int(AuthorityAction.STOP)],
                "stops_before_exhaustion": stops_before,
            }
        )
        if not row["cap_respected"]:
            raise RuntimeError(f"seed {seed} violated the online Human cap")
        rows.append(row)

    output = {
        "evidence_class": "post_locked_development_validation_diagnostic",
        "locked_holdout_accessed": False,
        "selection_performed": False,
        "per_seed": rows,
        "aggregate": aggregate(rows),
    }
    atomic_json(args.out / "audit.json", output)
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
