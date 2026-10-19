from __future__ import annotations

"""Run the frozen final SHARE/V2 post-hoc development ablation.

The script is hard-bound to the allowlisted synthetic development dataset.  It
does not contain, search for, or load either locked/confirmation dataset.  Full
V2 is loaded from frozen checkpoints; only ablations are trained, using train
and validation data.  The development-test split is already inspected and is
reported only as a post-hoc diagnostic.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from scripts.run_share_v2_development_multistart import (
    DEVELOPMENT_EPISODES_SHA256,
    EXPECTED_SPLIT_COUNTS,
    config_for,
)
from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


PROTOCOL_VERSION = "share-v2-final-development-ablation-20pct-v1"
SEEDS = (17, 29, 43, 71, 101)
VARIANTS = (
    "share_v2",
    "share_v2_myopic",
    "share_v2_wo_risk",
    "share_v2_wo_budget",
    "share_v2_wo_paired_replay",
    "share_v2_rollback_025",
)
ABLATIONS = VARIANTS[1:]
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
PRIMARY_METRICS = ("repair_sr", "uar", "stopped_rate", "collaboration_regret")
FULL_CHECKPOINTS = {
    17: Path("results/share_v2_stop_penalty_tuning_20260827/p060/share_v2_seed_17.pt"),
    29: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_29.pt"),
    43: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_43.pt"),
    71: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_71.pt"),
    101: Path("results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_101.pt"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def training_config(variant: str, seed: int, device: str) -> DQNConfig:
    if variant not in ABLATIONS:
        raise ValueError(f"{variant} is not trainable")
    base = config_for(seed, "paired_e30", device)
    if variant == "share_v2_myopic":
        return replace(base, gamma=0.0)
    if variant == "share_v2_wo_budget":
        return replace(base, constrained=False, human_constraint=False)
    if variant == "share_v2_wo_paired_replay":
        return replace(base, paired_replay=False)
    return base


def training_reward(variant: str) -> RewardConfig:
    return RewardConfig(rollback_penalty=0.25 if variant == "share_v2_rollback_025" else 0.60)


def validate_single_change(variant: str, config: DQNConfig, reward: RewardConfig, seed: int) -> None:
    base = config_for(seed, "paired_e30", config.device)
    config_diff = {
        key: (asdict(base)[key], asdict(config)[key])
        for key in asdict(base)
        if asdict(base)[key] != asdict(config)[key]
    }
    expected = {
        "share_v2_myopic": {"gamma"},
        "share_v2_wo_risk": set(),
        "share_v2_wo_budget": {"constrained", "human_constraint"},
        "share_v2_wo_paired_replay": {"paired_replay"},
        "share_v2_rollback_025": set(),
    }[variant]
    if set(config_diff) != expected:
        raise RuntimeError(f"{variant}: unexpected config differences {config_diff}")
    expected_rollback = 0.25 if variant == "share_v2_rollback_025" else 0.60
    if reward.rollback_penalty != expected_rollback:
        raise RuntimeError(f"{variant}: wrong training rollback penalty")


def calibrate(policy: DQNPolicy, validation: Sequence[RepairEpisode], variant: str) -> None:
    if variant != "share_v2_wo_budget":
        policy.calibrate_constraints(validation, 0.20, 0.05)
        return
    # Isolate training-time budget awareness: training and checkpoint selection
    # are unconstrained, while validation-only Human calibration is retained.
    policy.config.constrained = True
    policy.config.human_constraint = True
    policy.calibrate_constraints(validation, 0.20, 0.05)
    policy.config.constrained = False
    policy.config.human_constraint = False


def full_checkpoint(seed: int) -> Path:
    path = ROOT / FULL_CHECKPOINTS[seed]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_or_train(
    variant: str,
    seed: int,
    train: Sequence[RepairEpisode],
    validation: Sequence[RepairEpisode],
    base_encoder: FeatureEncoder,
    risk_encoder: FeatureEncoder,
    device: str,
    out: Path,
) -> tuple[DQNPolicy, dict[str, Any]]:
    encoder = risk_encoder if variant == "share_v2_wo_risk" else base_encoder
    if variant == "share_v2":
        path = full_checkpoint(seed)
        policy = DQNPolicy.load(
            str(path), encoder, RewardConfig(rollback_penalty=0.60),
            require_reward_config=True,
        )
        expected = config_for(seed, "paired_e30", "cuda:0")
        observed = asdict(policy.config)
        expected_dict = asdict(expected)
        # The stored checkpoint device is part of the frozen artifact.
        if observed != expected_dict:
            raise RuntimeError(f"seed={seed}: frozen Full V2 config mismatch")
        source = "frozen_full_v2_checkpoint"
        history: list[dict[str, Any]] = []
    else:
        config = training_config(variant, seed, device)
        reward = training_reward(variant)
        validate_single_change(variant, config, reward, seed)
        path = out / "checkpoints" / f"{variant}_seed_{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            policy = DQNPolicy.load(str(path), encoder, reward, require_reward_config=True)
            if asdict(policy.config) != asdict(config):
                raise RuntimeError(f"{variant}/seed={seed}: changed resumed config")
            source = "resumed_ablation_checkpoint"
            history = []
        else:
            policy = DQNPolicy(encoder, config, reward)
            policy.name = variant
            policy.fit(train, validation)
            calibrate(policy, validation, variant)
            history = list(policy.training_history)
            policy.save(str(path))
            source = "trained_on_train_validation_only"
    policy.name = variant
    action_support = Counter(action for (_, action), count in policy.logged_action_support.items() for _ in range(count))
    return policy, {
        "source": source,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "config": asdict(policy.config),
        "training_reward": asdict(policy.reward_config),
        "encoder_disabled_groups": list(encoder.disabled_groups),
        "gradient_steps": int(policy.gradient_steps),
        "replay_size": len(policy.replay),
        "logged_action_support": {
            AuthorityAction(action).name: int(count) for action, count in sorted(action_support.items())
        },
        "beta_h": float(policy.beta_h),
        "beta_u": float(policy.beta_u),
        "human_calibration_penalty": float(policy.human_calibration_penalty),
        "ai_calibration_penalty": float(policy.ai_calibration_penalty),
        "calibration_validation_hir": float(policy.calibration_validation_hir),
        "calibration_validation_unsafe": float(policy.calibration_validation_unsafe),
        "training_history": history,
    }


def evaluate_variant(
    policy: DQNPolicy,
    episodes: Sequence[RepairEpisode],
    cap: int,
    evaluation_reward: RewardConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    result = evaluate_policy(wrapper, episodes, evaluation_reward, 0.20, None)
    records = result["records"]
    human_episodes = sum(int(record["human_used"]) for record in records)
    eligible = [
        record for record in records
        if not record["human_used"] and int(record["actions"][-1]) != int(AuthorityAction.STOP)
    ]
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError(f"{policy.name}: online hard cap audit failed")
    stop_before = sum(
        int(event["executed_action"] == int(AuthorityAction.STOP) and not event["exhausted_before_decision"])
        for event in wrapper.events
    )
    stop_after = sum(
        int(event["executed_action"] == int(AuthorityAction.STOP) and event["exhausted_before_decision"])
        for event in wrapper.events
    )
    first_exhaustion = next(
        (index for index, event in enumerate(wrapper.events) if event["exhausted_before_decision"]),
        None,
    )
    result.update({
        "hard_human_cap": cap,
        "human_interventions": human_episodes,
        "uar_denominator": len(eligible),
        "cap_respected": human_episodes <= cap,
        "synthetic_development_evaluation": True,
    })
    audit = {
        "hard_human_cap": cap,
        "human_interventions": human_episodes,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "value_policy_replans": wrapper.value_policy_replans,
        "first_exhaustion_event_index": first_exhaustion,
        "stop_decisions_before_exhaustion": stop_before,
        "stop_decisions_after_exhaustion": stop_after,
        "cap_respected": human_episodes <= cap,
        "events": wrapper.events,
    }
    return result, audit


def aggregate(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload["results"]:
            buckets[str(row["policy"])].append(row)
    output = []
    for variant in VARIANTS:
        rows = buckets[variant]
        item: dict[str, Any] = {"policy": variant, "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values)
        for metric in ("human_interventions", "uar_denominator"):
            values = [int(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        item["all_runs_respected_cap"] = all(bool(row["cap_respected"]) for row in rows)
        output.append(item)
    return output


def exact_sign_flip(values: Sequence[float]) -> float:
    observed = abs(mean(values))
    permutations = (
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return sum(value >= observed - 1e-15 for value in permutations) / (2 ** len(values))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def record_metric(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    if metric == "repair_sr":
        return mean(float(record["success"]) for record in records)
    if metric == "hir":
        return mean(float(record["human_used"]) for record in records)
    if metric == "stopped_rate":
        return mean(float(int(record["actions"][-1]) == int(AuthorityAction.STOP)) for record in records)
    if metric == "normalized_return":
        return mean(sum(float(value) for value in record["rewards"]) for record in records)
    if metric == "uar":
        eligible = [
            record for record in records
            if not record["human_used"] and int(record["actions"][-1]) != int(AuthorityAction.STOP)
        ]
        return mean(float(record["unsafe"] or not record["success"]) for record in eligible) if eligible else math.nan
    if metric == "unsafe_autonomy_cost_rate":
        return mean(float(record["unsafe_actions"]) for record in records)
    if metric == "collaboration_regret":
        return mean(
            float(oracle) - float(selected)
            for record in records
            for selected, oracle in zip(record["selected_returns"], record["oracle_returns"])
        )
    raise ValueError(metric)


def paired_statistics(
    payloads: Sequence[Mapping[str, Any]],
    episode_by_id: Mapping[str, RepairEpisode],
) -> list[dict[str, Any]]:
    by_seed = {
        int(payload["seed"]): {row["policy"]: row for row in payload["results"]}
        for payload in payloads
    }
    groups = sorted({episode.group_id for episode in episode_by_id.values()})
    group_ids = {
        group: [episode_id for episode_id, episode in episode_by_id.items() if episode.group_id == group]
        for group in groups
    }
    rng = np.random.default_rng(20260827)
    output = []
    for ablation in ABLATIONS:
        for metric in METRICS:
            differences = [
                float(by_seed[seed][ablation][metric]) - float(by_seed[seed]["share_v2"][metric])
                for seed in SEEDS
            ]
            seed_indices = rng.integers(0, len(SEEDS), size=(20_000, len(SEEDS)))
            seed_boot = np.asarray(differences)[seed_indices].mean(axis=1)
            group_boot = []
            for _ in range(5_000):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                seed_differences = []
                for seed in SEEDS:
                    values = {}
                    for policy in ("share_v2", ablation):
                        records = {str(row["episode_id"]): row for row in by_seed[seed][policy]["records"]}
                        sample_records = [records[episode_id] for group in sampled for episode_id in group_ids[group]]
                        values[policy] = record_metric(sample_records, metric)
                    if not (math.isnan(values[ablation]) or math.isnan(values["share_v2"])):
                        seed_differences.append(values[ablation] - values["share_v2"])
                group_boot.append(mean(seed_differences))
            output.append({
                "comparison": f"{ablation}-minus-share_v2",
                "metric": metric,
                "per_seed_differences": differences,
                "mean_difference": mean(differences),
                "std_difference": stdev(differences),
                "seed_bootstrap_ci95": [float(np.percentile(seed_boot, 2.5)), float(np.percentile(seed_boot, 97.5))],
                "workflow_group_cluster_bootstrap_ci95": [float(np.percentile(group_boot, 2.5)), float(np.percentile(group_boot, 97.5))],
                "exact_sign_flip_p_two_sided": exact_sign_flip(differences),
            })
    primary = [index for index, row in enumerate(output) if row["metric"] in PRIMARY_METRICS]
    adjusted = holm_adjust([output[index]["exact_sign_flip_p_two_sided"] for index in primary])
    for index, value in zip(primary, adjusted):
        output[index]["holm_p_across_20_primary_tests"] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--run-variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    args.out = args.out.resolve()

    run_seeds = tuple(int(value) for value in args.run_seeds.split(","))
    run_variants = tuple(value for value in args.run_variants.split(",") if value)
    if not set(run_seeds).issubset(SEEDS) or not set(run_variants).issubset(VARIANTS):
        raise SystemExit("run seeds/variants must be subsets of the frozen matrix")
    if sha256_file(args.episodes) != DEVELOPMENT_EPISODES_SHA256:
        raise SystemExit("refusing to run: input is not the allowlisted development dataset")

    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    counts = {name: len(values) for name, values in split.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"unexpected development split counts: {counts}")
    evaluation = sorted(split["test"], key=lambda episode: episode.episode_id)
    episode_by_id = {episode.episode_id: episode for episode in evaluation}
    order = [episode.episode_id for episode in evaluation]
    cap = math.floor(0.20 * len(evaluation))
    encoder_path = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
    remaining_encoder_path = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining/feature_encoder.json"
    if sha256_file(encoder_path) != sha256_file(remaining_encoder_path):
        raise SystemExit("frozen Full V2 encoder artifacts disagree")
    base_encoder = FeatureEncoder.load(encoder_path)
    risk_encoder = base_encoder.without("risk")
    evaluation_reward = RewardConfig()

    plan_path = args.out / "ABLATION_PLAN.md"
    if not plan_path.is_file():
        raise SystemExit("frozen ABLATION_PLAN.md must exist before execution")
    checkpoint_hashes = {str(seed): sha256_file(full_checkpoint(seed)) for seed in SEEDS}
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "post_confirmation_already_inspected_synthetic_development_ablation",
        "confirmatory_or_preregistered": False,
        "locked_or_confirmation_data_accessed": False,
        "development_test_already_inspected": True,
        "selection_uses_development_test": False,
        "episodes": str(args.episodes),
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "human_budget": 0.20,
        "hard_human_episode_cap": cap,
        "episode_order": order,
        "episode_order_sha256": sha256_json(order),
        "training_reward_rollback_penalty": {variant: training_reward(variant).rollback_penalty for variant in ABLATIONS},
        "common_evaluation_reward": asdict(evaluation_reward),
        "full_v2_checkpoint_sha256": checkpoint_hashes,
        "ablation_plan_sha256": sha256_file(plan_path),
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": {
            "scripts/run_share_v2_final_ablation.py": sha256_file(Path(__file__)),
            "scripts/run_share_online_hard_cap.py": sha256_file(ROOT / "scripts" / "run_share_online_hard_cap.py"),
            "scripts/run_share_v2_development_multistart.py": sha256_file(ROOT / "scripts" / "run_share_v2_development_multistart.py"),
            "share/dqn.py": sha256_file(ROOT / "share" / "dqn.py"),
            "share/evaluation.py": sha256_file(ROOT / "share" / "evaluation.py"),
            "share/features.py": sha256_file(ROOT / "share" / "features.py"),
        },
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != protocol["protocol_sha256"]:
            raise SystemExit("refusing to resume a changed protocol")
    else:
        atomic_json(manifest_path, protocol)
        base_encoder.save(args.out / "feature_encoder.json")
        risk_encoder.save(args.out / "feature_encoder_wo_risk.json")

    for seed in run_seeds:
        seed_path = args.out / f"seed_{seed}.json"
        payload = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else {
            "seed": seed, "results": [], "training_audit": {}, "budget_audit": {}
        }
        completed = {row["policy"] for row in payload["results"]}
        for variant in run_variants:
            if variant in completed:
                continue
            policy, training_audit = load_or_train(
                variant, seed, split["train"], split["validation"], base_encoder,
                risk_encoder, args.device, args.out,
            )
            result, budget_audit = evaluate_variant(policy, evaluation, cap, evaluation_reward)
            payload["results"].append(result)
            payload["training_audit"][variant] = training_audit
            payload["budget_audit"][variant] = budget_audit
            atomic_json(seed_path, payload)
            print(json.dumps({"seed": seed, "variant": variant, "status": "complete"}), flush=True)

    payloads = []
    for seed in SEEDS:
        path = args.out / f"seed_{seed}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if {row["policy"] for row in payload["results"]} == set(VARIANTS):
                payloads.append(payload)
    if len(payloads) == len(SEEDS):
        aggregate_rows = aggregate(payloads)
        paired_rows = paired_statistics(payloads, episode_by_id)
        atomic_json(args.out / "aggregate.json", aggregate_rows)
        atomic_json(args.out / "paired_comparisons.json", paired_rows)
        write_csv(args.out / "aggregate.csv", aggregate_rows)
        write_csv(args.out / "paired_comparisons.csv", paired_rows)
        write_csv(args.out / "per_seed.csv", [
            {key: value for key, value in row.items() if key != "records"}
            for payload in payloads for row in payload["results"]
        ])
    print(json.dumps({"out": str(args.out), "complete_seeds": len(payloads), "matrix_complete": len(payloads) == len(SEEDS)}))


if __name__ == "__main__":
    main()
