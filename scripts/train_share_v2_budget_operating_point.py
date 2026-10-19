from __future__ import annotations

"""Train one fixed SHARE-V2 budget operating point on development data only.

The V2 architecture, paired replay, rollback penalty, optimizer, epoch count,
and seeds are fixed.  Only the declared training-time Human budget changes.
The confirmation holdout is neither accepted as an argument nor loaded here.
"""

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_v2_development_multistart import (
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
    METRICS,
    atomic_json,
    config_for,
    sha256_file,
    summarize_without_records,
)
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig

PROTOCOL = "share-v2-fixed-budget-operating-point-v1"
SEEDS = (17, 29, 43, 71, 101)


def evaluate_hard_cap(
    policy: DQNPolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
    budget: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, reward, budget, None)
    human_episodes = sum(int(row["human_used"]) for row in raw["records"])
    human_actions = sum(
        int(action == int(AuthorityAction.HUMAN))
        for row in raw["records"]
        for action in row["actions"]
    )
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError("hard Human cap audit failed")
    result = summarize_without_records(raw)
    result.update(
        hard_human_cap=cap,
        human_interventions=human_episodes,
        human_actions=human_actions,
        cap_respected=True,
    )
    audit = {
        "hard_human_cap": cap,
        "human_interventions": wrapper.human_interventions,
        "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "value_policy_replans": wrapper.value_policy_replans,
        "cap_respected": True,
    }
    return result, audit


def aggregate(rows: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[section][metric]) for row in rows]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values)
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    output["all_caps_respected"] = all(bool(row[section]["cap_respected"]) for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, required=True, choices=(0.10, 0.30))
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing non-allowlisted development data")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite an operating-point ledger")
    args.out.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    counts = {name: len(rows) for name, rows in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected split counts: {counts}")
    encoder = FeatureEncoder.fit(step.state for episode in split["train"] for step in episode.steps)
    encoder.save(args.out / "feature_encoder.json")
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    manifest = {
        "protocol": PROTOCOL,
        "evidence_class": "development_training_only",
        "confirmation_holdout_accessed": False,
        "budget": args.budget,
        "seeds": list(SEEDS),
        "fixed_v2": {"paired_replay": True, "epochs": 30, "gamma": 0.95, "rollback_penalty": 0.60},
        "selection": "no hyperparameter selection; validation used only for checkpoint selection and Human-budget calibration",
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": {
            "script": sha256_file(Path(__file__)),
            "dqn": sha256_file(ROOT / "share/dqn.py"),
            "hard_cap": sha256_file(ROOT / "scripts/run_share_online_hard_cap.py"),
        },
    }
    atomic_json(args.out / "manifest.json", manifest)
    rows = []
    for seed in SEEDS:
        config = replace(config_for(seed, "paired_e30", args.device), human_budget=args.budget)
        policy = DQNPolicy(encoder, config, training_reward)
        policy.name = "share_v2"
        policy.fit(split["train"], split["validation"])
        policy.calibrate_constraints(split["validation"], args.budget, 0.05)
        validation, validation_audit = evaluate_hard_cap(policy, split["validation"], evaluation_reward, args.budget)
        diagnostic, diagnostic_audit = evaluate_hard_cap(policy, split["test"], evaluation_reward, args.budget)
        checkpoint = args.out / f"share_v2_budget_{args.budget:.1f}_seed_{seed}.pt"
        policy.save(str(checkpoint))
        row = {
            "seed": seed,
            "validation": validation,
            "validation_audit": validation_audit,
            "development_test_diagnostic": diagnostic,
            "development_test_audit": diagnostic_audit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config": asdict(config),
        }
        atomic_json(args.out / f"seed_{seed}.json", row)
        rows.append(row)
        print(json.dumps({"budget": args.budget, "seed": seed, "complete": True}), flush=True)
    summary = {
        "budget": args.budget,
        "validation": aggregate(rows, "validation"),
        "development_test_diagnostic": aggregate(rows, "development_test_diagnostic"),
        "interpretation_boundary": "Fixed V2 development operating point; confirmation results are produced separately after artifact sealing.",
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"out": str(args.out), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
