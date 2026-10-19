from __future__ import annotations

"""Validation-only Stop-penalty tuning for paired-replay SHARE-V2.

Candidate penalties affect training only.  Every reported validation and
development-test metric is evaluated with the original frozen RewardConfig so
that Return remains comparable across candidates.  The script is hard-bound to
the allowlisted synthetic development dataset and never loads locked data.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_v2_development_multistart import (
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
    METRICS,
    atomic_json,
    config_for,
    sha256_file,
    summarize_without_records,
)
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


PROTOCOL_VERSION = "share-v2-stop-penalty-development-v1"


def evaluate_hard_cap(
    policy: DQNPolicy,
    episodes: Sequence[RepairEpisode],
    evaluation_reward: RewardConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = math.floor(0.20 * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, evaluation_reward, 0.20, None)
    human_episodes = sum(int(record["human_used"]) for record in raw["records"])
    human_actions = sum(
        int(action == int(AuthorityAction.HUMAN))
        for record in raw["records"]
        for action in record["actions"]
    )
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError("hard Human cap audit failed")
    result = summarize_without_records(raw)
    result.update(
        {
            "hard_human_cap": cap,
            "human_interventions": human_episodes,
            "human_actions": human_actions,
            "cap_respected": human_episodes <= cap,
        }
    )
    audit = {
        "hard_human_cap": cap,
        "human_interventions": human_episodes,
        "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "value_policy_replans": wrapper.value_policy_replans,
        "cap_respected": human_episodes <= cap,
    }
    return result, audit


def aggregate(rows: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    values = [row[section] for row in rows]
    output: dict[str, Any] = {"seeds": len(values)}
    for metric in METRICS:
        samples = [float(value[metric]) for value in values]
        output[f"{metric}_mean"] = mean(samples)
        output[f"{metric}_std"] = stdev(samples) if len(samples) > 1 else 0.0
        output[f"{metric}_min"] = min(samples)
        output[f"{metric}_max"] = max(samples)
    if section == "development_test":
        output["all_runs_respected_cap"] = all(bool(value["cap_respected"]) for value in values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollback-penalty", type=float, required=True)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must be a non-empty unique list")
    if not 0.25 < args.rollback_penalty < 2.0:
        raise SystemExit("candidate rollback penalty must be in (0.25, 2.0)")
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run: episodes are not the allowlisted development dataset")

    episodes = load_episodes(args.episodes)
    split = {
        name: [episode for episode in episodes if episode.split == name]
        for name in ("train", "validation", "test")
    }
    counts = {name: len(items) for name, items in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected development split counts: {counts}")

    encoder = FeatureEncoder.fit(
        [step.state for episode in split["train"] for step in episode.steps]
    )
    evaluation_reward = RewardConfig()
    training_reward = RewardConfig(rollback_penalty=args.rollback_penalty)
    args.out.mkdir(parents=True, exist_ok=True)
    encoder.save(args.out / "feature_encoder.json")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "post_locked_development_tuning_only",
        "locked_holdout_accessed": False,
        "eligible_for_locked_or_confirmatory_claim": False,
        "selection_split": "validation",
        "development_test_already_inspected": True,
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "seeds": list(seeds),
        "training_variant": "paired_e30",
        "training_reward": asdict(training_reward),
        "common_evaluation_reward": asdict(evaluation_reward),
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": {
            "scripts/run_share_v2_stop_penalty_tuning.py": sha256_file(Path(__file__)),
            "scripts/run_share_v2_development_multistart.py": sha256_file(ROOT / "scripts" / "run_share_v2_development_multistart.py"),
            "share/dqn.py": sha256_file(ROOT / "share" / "dqn.py"),
        },
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("existing output directory has a different manifest")
    else:
        atomic_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        result_path = args.out / f"seed_{seed}.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            config = config_for(seed, "paired_e30", args.device)
            policy = DQNPolicy(encoder, config, training_reward)
            policy.name = "share_v2"
            policy.fit(split["train"], split["validation"])
            policy.calibrate_constraints(split["validation"], 0.20, 0.05)
            validation = summarize_without_records(
                evaluate_policy(policy, split["validation"], evaluation_reward, 0.20, None)
            )
            development_test, cap_audit = evaluate_hard_cap(
                policy, split["test"], evaluation_reward
            )
            checkpoint = args.out / f"share_v2_seed_{seed}.pt"
            policy.save(str(checkpoint))
            payload = {
                "seed": seed,
                "rollback_penalty": args.rollback_penalty,
                "config": asdict(config),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation": validation,
                "development_test": development_test,
                "hard_cap_audit": cap_audit,
            }
            atomic_json(result_path, payload)
            print(json.dumps({"completed": str(result_path)}), flush=True)
        rows.append(payload)

    summary = {
        "rollback_penalty": args.rollback_penalty,
        "selection_uses_development_test": False,
        "validation_aggregate": aggregate(rows, "validation"),
        "development_test_aggregate": aggregate(rows, "development_test"),
        "interpretation_boundary": "Post-locked development evidence only; requires a new untouched holdout.",
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"out": str(args.out), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
