from __future__ import annotations

"""Development-only SHARE-V2 multi-start and replay-intensity study.

The script is intentionally bound to the existing synthetic development
dataset.  It never loads the sealed 400-episode holdout and must not be used to
replace the published locked result.  Model/checkpoint selection uses only the
declared validation split; the already-inspected development-test split is
reported as diagnostic evidence.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


PROTOCOL_VERSION = "share-v2-development-multistart-v1"
DEVELOPMENT_EPISODES_SHA256 = "174a2c387aa8fe4cece6374a54a9d23c5db062bde6349c71578384c386b5a694"
EXPECTED_SPLIT_COUNTS = {"train": 542, "validation": 127, "test": 120}
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
VARIANTS = {
    "standard_e30": {"epochs": 30, "paired_replay": False},
    "standard_e60": {"epochs": 60, "paired_replay": False},
    "paired_e30": {"epochs": 30, "paired_replay": True},
    "paired_e60": {"epochs": 60, "paired_replay": True},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def config_for(seed: int, variant: str, device: str) -> DQNConfig:
    settings = VARIANTS[variant]
    return DQNConfig(
        hidden_sizes=(128, 64),
        learning_rate=1e-3,
        gamma=0.95,
        replay_capacity=50_000,
        batch_size=64,
        target_update=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        dual_learning_rate_human=5e-3,
        dual_learning_rate_unsafe=5e-3,
        human_budget=0.20,
        unsafe_budget=0.05,
        training_epochs=int(settings["epochs"]),
        constrained=True,
        human_constraint=True,
        unsafe_constraint=False,
        double_dqn=True,
        seed=seed,
        torch_threads=1,
        paired_replay=bool(settings["paired_replay"]),
        updates_per_epoch=1_000,
        constraint_confidence_z=0.0,
        checkpoint_selection=True,
        support_mode="posthoc_stop",
        minimum_ai_support=3,
        device=device,
    )


def summarize_without_records(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "records"}


def evaluate_with_hard_cap(
    policy: DQNPolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
    cap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    result = evaluate_policy(wrapper, episodes, reward, cap / len(episodes), None)
    human_actions = sum(
        int(action == int(AuthorityAction.HUMAN))
        for record in result["records"]
        for action in record["actions"]
    )
    human_episodes = sum(int(record["human_used"]) for record in result["records"])
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError("online hard-Human cap audit failed")
    summary = summarize_without_records(result)
    summary.update(
        {
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "human_actions": human_actions,
            "cap_respected": wrapper.human_interventions <= cap,
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
    return summary, audit


def run_one(
    variant: str,
    seed: int,
    split: Mapping[str, Sequence[RepairEpisode]],
    encoder: FeatureEncoder,
    reward: RewardConfig,
    device: str,
    out: Path,
) -> dict[str, Any]:
    config = config_for(seed, variant, device)
    policy = DQNPolicy(encoder, config, reward)
    policy.name = variant
    policy.fit(split["train"], split["validation"])
    policy.calibrate_constraints(split["validation"], 0.20, 0.05)

    validation = summarize_without_records(
        evaluate_policy(policy, split["validation"], reward, 0.20, None)
    )
    cap = math.floor(0.20 * len(split["test"]))
    development_test, cap_audit = evaluate_with_hard_cap(
        policy, split["test"], reward, cap
    )
    checkpoint = out / "checkpoints" / f"{variant}_seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(checkpoint))

    return {
        "variant": variant,
        "seed": seed,
        "config": asdict(config),
        "checkpoint": str(checkpoint.relative_to(out)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation": validation,
        "development_test": development_test,
        "hard_cap_audit": cap_audit,
        "post_calibration": policy.policy_diagnostics(
            split["validation"], epoch=config.training_epochs, phase="post_calibration"
        ),
        "selected_checkpoint_pre_calibration": policy.selected_checkpoint_diagnostics,
        "training_history": policy.training_history,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], split_name: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["variant"])].append(row[split_name])
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        values = buckets.get(variant, [])
        if not values:
            continue
        item: dict[str, Any] = {"variant": variant, "seeds": len(values)}
        for metric in METRICS:
            samples = [float(value[metric]) for value in values]
            item[f"{metric}_mean"] = mean(samples)
            item[f"{metric}_std"] = stdev(samples) if len(samples) > 1 else 0.0
            item[f"{metric}_min"] = min(samples)
            item[f"{metric}_max"] = max(samples)
        if split_name == "development_test":
            item["all_runs_respected_cap"] = all(bool(value["cap_respected"]) for value in values)
        output.append(item)
    return output


def validation_selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    # Predeclared before execution: feasibility first, then validation return,
    # repair success, and coverage.  Development-test outcomes are excluded.
    feasible = float(row["hir_mean"] <= 0.20 + 1e-12)
    return (
        feasible,
        float(row["normalized_return_mean"]),
        float(row["repair_sr_mean"]),
        -float(row["stopped_rate_mean"]),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(","))
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must be a non-empty unique list")
    unknown = sorted(set(variants) - set(VARIANTS))
    if not variants or unknown:
        raise SystemExit(f"unknown/empty variants: {unknown}")
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run: episodes are not the allowlisted development dataset")

    episodes = load_episodes(args.episodes)
    split = {
        name: [episode for episode in episodes if episode.split == name]
        for name in ("train", "validation", "test")
    }
    actual_counts = {name: len(items) for name, items in split.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected development split counts: {actual_counts}")

    encoder = FeatureEncoder.fit(
        [step.state for episode in split["train"] for step in episode.steps]
    )
    reward = RewardConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    encoder.save(args.out / "feature_encoder.json")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "post_locked_development_tuning_only",
        "locked_holdout_accessed": False,
        "eligible_for_locked_or_confirmatory_claim": False,
        "selection_split": "validation",
        "development_test_already_inspected": True,
        "episodes": str(args.episodes),
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": actual_counts,
        "seeds": list(seeds),
        "variants": {name: VARIANTS[name] for name in variants},
        "device": args.device,
        "reward_config": asdict(reward),
        "selection_rule": "feasible mean validation HIR <= .20; then maximum mean validation Return, Repair.SR, and coverage in that order",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_sha256": {
            "scripts/run_share_v2_development_multistart.py": sha256_file(Path(__file__)),
            "share/dqn.py": sha256_file(ROOT / "share" / "dqn.py"),
            "share/evaluation.py": sha256_file(ROOT / "share" / "evaluation.py"),
            "scripts/run_share_online_hard_cap.py": sha256_file(ROOT / "scripts" / "run_share_online_hard_cap.py"),
        },
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit("existing output directory has a different manifest")
    else:
        atomic_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            result_path = args.out / f"{variant}_seed_{seed}.json"
            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                payload = run_one(variant, seed, split, encoder, reward, args.device, args.out)
                atomic_json(result_path, payload)
                print(json.dumps({"completed": str(result_path)}), flush=True)
            rows.append(payload)

    validation_aggregate = aggregate(rows, "validation")
    development_aggregate = aggregate(rows, "development_test")
    selected = max(validation_aggregate, key=validation_selection_key)
    summary = {
        "selection_uses_development_test": False,
        "selected_variant_by_validation_only": selected["variant"],
        "validation_aggregate": validation_aggregate,
        "development_test_aggregate": development_aggregate,
        "interpretation_boundary": "Post-locked development evidence only; a new untouched holdout is required for any confirmatory manuscript replacement.",
    }
    atomic_json(args.out / "summary.json", summary)
    write_csv(args.out / "validation_aggregate.csv", validation_aggregate)
    write_csv(args.out / "development_test_aggregate.csv", development_aggregate)
    print(json.dumps({"out": str(args.out), "complete": True, "selected_by_validation": selected["variant"]}, indent=2))


if __name__ == "__main__":
    main()
