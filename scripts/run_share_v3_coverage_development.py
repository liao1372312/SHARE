from __future__ import annotations

"""Development-only coverage-oriented SHARE-V3 candidate selection.

This protocol addresses the Stop-heavy transfer observed for the original
controller without reading the opened locked holdout.  Candidates differ only
in development-time paired-replay training: a shorter discount reduces
long-horizon value extrapolation, longer training improves fitted replay
coverage, and the rollback cost prices abstention.  Selection uses *only* the
allowlisted validation split under the online 20% Human hard cap.  The already
inspected development-test split is retained solely as a diagnostic.

The script intentionally cannot load any holdout path.  Its output is a new
development candidate, not a replacement for the original locked result.
"""

import argparse
import csv
import hashlib
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


PROTOCOL_VERSION = "share-v3-coverage-development-v1"
PILOT_SEED = 17
FINAL_SEEDS = (17, 29, 43, 71, 101)

# These candidates were fixed before executing this script.  The first row is
# the SHARE-V2 training configuration and is retained as a development control.
CANDIDATES = (
    {"name": "v2_control", "gamma": 0.95, "epochs": 30, "rollback_penalty": 0.60},
    {"name": "short_discount", "gamma": 0.90, "epochs": 30, "rollback_penalty": 0.60},
    {"name": "short_discount_long", "gamma": 0.90, "epochs": 60, "rollback_penalty": 0.60},
    {"name": "coverage_cost", "gamma": 0.90, "epochs": 60, "rollback_penalty": 0.80},
)


def hard_cap_evaluate(
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
        raise RuntimeError("online hard-Human-cap audit failed")
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
        "human_interventions": wrapper.human_interventions,
        "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "value_policy_replans": wrapper.value_policy_replans,
        "cap_respected": wrapper.human_interventions <= cap,
    }
    return result, audit


def train_one(
    candidate: Mapping[str, Any],
    seed: int,
    split: Mapping[str, Sequence[RepairEpisode]],
    encoder: FeatureEncoder,
    device: str,
    checkpoint: Path | None,
    include_development_test: bool,
) -> dict[str, Any]:
    training_reward = RewardConfig(rollback_penalty=float(candidate["rollback_penalty"]))
    evaluation_reward = RewardConfig()
    config = replace(
        config_for(seed, "paired_e30", device),
        gamma=float(candidate["gamma"]),
        training_epochs=int(candidate["epochs"]),
    )
    policy = DQNPolicy(encoder, config, training_reward)
    policy.name = f"share_v3_{candidate['name']}"
    policy.fit(split["train"], split["validation"])
    policy.calibrate_constraints(split["validation"], 0.20, 0.05)
    validation, validation_audit = hard_cap_evaluate(policy, split["validation"], evaluation_reward)
    development_test = None
    test_audit = None
    if include_development_test:
        development_test, test_audit = hard_cap_evaluate(policy, split["test"], evaluation_reward)
    checkpoint_sha256 = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(checkpoint))
        checkpoint_sha256 = sha256_file(checkpoint)
    return {
        "candidate": dict(candidate),
        "seed": seed,
        "config": asdict(config),
        "training_reward": asdict(training_reward),
        "evaluation_reward": asdict(evaluation_reward),
        "validation": validation,
        "validation_hard_cap_audit": validation_audit,
        "development_test": development_test,
        "development_test_hard_cap_audit": test_audit,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_checkpoint_pre_calibration": policy.selected_checkpoint_diagnostics,
        "post_calibration": policy.policy_diagnostics(
            split["validation"], epoch=config.training_epochs, phase="post_calibration"
        ),
    }


def candidate_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["validation"]
    passes = (
        bool(metrics["cap_respected"])
        and float(metrics["hir"]) <= 0.20 + 1e-12
        and float(metrics["uar"]) <= 0.12 + 1e-12
        and float(metrics["stopped_rate"]) <= 0.05 + 1e-12
    )
    # Gates first, then repair, return, and coverage.  This key is fixed before
    # the pilot is executed and excludes every development-test quantity.
    return (
        float(passes),
        float(metrics["repair_sr"]),
        float(metrics["normalized_return"]),
        -float(metrics["stopped_rate"]),
    )


def summarize(rows: Sequence[Mapping[str, Any]], section: str) -> dict[str, Any]:
    values = [row[section] for row in rows]
    result: dict[str, Any] = {"seeds": len(values)}
    for metric in METRICS:
        samples = [float(value[metric]) for value in values]
        result[f"{metric}_mean"] = mean(samples)
        result[f"{metric}_std"] = stdev(samples) if len(samples) > 1 else 0.0
        result[f"{metric}_min"] = min(samples)
        result[f"{metric}_max"] = max(samples)
    result["all_runs_respected_cap"] = all(bool(value["cap_respected"]) for value in values)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_v3_coverage_development_20260827"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run on a non-allowlisted development dataset")
    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    counts = {name: len(items) for name, items in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected development split counts: {counts}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite a prior V3 development ledger")
    args.out.mkdir(parents=True, exist_ok=False)

    encoder = FeatureEncoder.fit(step.state for episode in split["train"] for step in episode.steps)
    encoder.save(args.out / "feature_encoder.json")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "post_locked_development_candidate_only",
        "locked_holdout_accessed": False,
        "eligible_to_replace_original_locked_result": False,
        "selection_split": "validation_only",
        "development_test_used_for_selection": False,
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "pilot_seed": PILOT_SEED,
        "final_seeds": list(FINAL_SEEDS),
        "candidates": list(CANDIDATES),
        "validation_gate": "hard-cap respected; HIR<=.20; UAR<=.12; Stop<=.05",
        "validation_ranking": "gate, Repair.SR, Return, lower Stop",
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_sha256": {
            "scripts/run_share_v3_coverage_development.py": sha256_file(Path(__file__)),
            "scripts/run_share_v2_development_multistart.py": sha256_file(ROOT / "scripts" / "run_share_v2_development_multistart.py"),
            "share/dqn.py": sha256_file(ROOT / "share" / "dqn.py"),
            "scripts/run_share_online_hard_cap.py": sha256_file(ROOT / "scripts" / "run_share_online_hard_cap.py"),
        },
    }
    atomic_json(args.out / "manifest.json", manifest)

    pilot_rows = []
    for candidate in CANDIDATES:
        row = train_one(candidate, PILOT_SEED, split, encoder, args.device, None, False)
        # The inspected development-test split is not read during selection.
        atomic_json(args.out / f"pilot_{candidate['name']}.json", row)
        pilot_rows.append(row)
    selected_pilot = max(pilot_rows, key=candidate_key)
    selected = selected_pilot["candidate"]

    final_rows = []
    for seed in FINAL_SEEDS:
        checkpoint = args.out / "checkpoints" / f"share_v3_{selected['name']}_seed_{seed}.pt"
        row = train_one(selected, seed, split, encoder, args.device, checkpoint, True)
        atomic_json(args.out / f"seed_{seed}.json", row)
        final_rows.append(row)

    pilot_table = []
    for row in pilot_rows:
        metrics = row["validation"]
        pilot_table.append(
            {
                "candidate": row["candidate"]["name"],
                "gamma": row["candidate"]["gamma"],
                "epochs": row["candidate"]["epochs"],
                "rollback_penalty": row["candidate"]["rollback_penalty"],
                **{metric: metrics[metric] for metric in METRICS},
                "cap_respected": metrics["cap_respected"],
                "passes_validation_gate": candidate_key(row)[0] == 1.0,
            }
        )
    write_csv(args.out / "pilot_validation.csv", pilot_table)
    summary = {
        "selected_candidate_by_validation_only": selected,
        "pilot_seed": PILOT_SEED,
        "pilot_validation": pilot_table,
        "selected_five_seed_validation": summarize(final_rows, "validation"),
        "selected_five_seed_development_test_diagnostic": summarize(final_rows, "development_test"),
        "all_validation_caps_respected": all(bool(row["validation"]["cap_respected"]) for row in final_rows),
        "all_development_test_caps_respected": all(bool(row["development_test"]["cap_respected"]) for row in final_rows),
        "interpretation_boundary": "Development-only post-locked candidate. No locked data were read. A new sealed workflow-group-disjoint holdout is required for confirmation.",
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"out": str(args.out), "selected": selected, "complete": True}, indent=2))


if __name__ == "__main__":
    main()
