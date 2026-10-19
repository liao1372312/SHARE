from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
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


PROTOCOL_VERSION = "share-final-development-ablation-20pct-v1"
SEEDS = (17, 29, 43, 71, 101)
POLICIES = ("share", "share_myopic", "share_wo_risk", "share_wo_budget", "share_dual")
ABLATIONS = POLICIES[1:]
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")
PRIMARY_METRICS = ("repair_sr", "uar", "stopped_rate", "collaboration_regret")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ablation_config(name: str, seed: int, epochs: int, device: str) -> DQNConfig:
    if name not in {"share_myopic", "share_wo_risk"}:
        raise ValueError(f"{name} is frozen/reused rather than trained here")
    return DQNConfig(
        training_epochs=epochs,
        human_budget=0.20,
        unsafe_budget=0.05,
        seed=seed,
        device=device,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.0 if name == "share_myopic" else 0.95,
        replay_capacity=50_000,
        target_update=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        dual_learning_rate_human=5e-3,
        dual_learning_rate_unsafe=5e-3,
        updates_per_epoch=1_000,
        paired_replay=False,
        constraint_confidence_z=0.0,
        double_dqn=True,
        constrained=True,
        human_constraint=True,
        unsafe_constraint=False,
    )


def assert_frozen_config(name: str, policy: DQNPolicy, seed: int) -> None:
    config = policy.config
    expected = {
        "share": (0.95, True, True, False),
        "share_wo_budget": (0.95, False, False, False),
        "share_dual": (0.95, True, True, True),
    }[name]
    observed = (config.gamma, config.constrained, config.human_constraint, config.unsafe_constraint)
    if observed != expected or config.seed != seed:
        raise RuntimeError(f"frozen {name}/seed={seed} config mismatch: {observed}, seed={config.seed}")


def train_ablation(
    name: str,
    seed: int,
    train: Sequence[RepairEpisode],
    validation: Sequence[RepairEpisode],
    encoder: FeatureEncoder,
    reward: RewardConfig,
    epochs: int,
    device: str,
    checkpoint: Path,
) -> tuple[DQNPolicy, list[dict[str, Any]]]:
    config = ablation_config(name, seed, epochs, device)
    policy = DQNPolicy(encoder, config, reward)
    policy.name = name
    policy.fit(train, validation)
    policy.calibrate_constraints(validation, 0.20, 0.05)
    diagnostics = list(policy.training_history)
    if policy.selected_checkpoint_diagnostics is not None:
        diagnostics.append(policy.selected_checkpoint_diagnostics)
    diagnostics.append(policy.policy_diagnostics(validation, epoch=epochs, phase="post_calibration"))
    policy.save(str(checkpoint))
    return policy, diagnostics


def load_policies(
    seed: int,
    train: Sequence[RepairEpisode],
    validation: Sequence[RepairEpisode],
    base_encoder: FeatureEncoder,
    risk_encoder: FeatureEncoder,
    reward: RewardConfig,
    epochs: int,
    device: str,
    frozen: Path,
    out: Path,
) -> tuple[list[DQNPolicy], dict[str, Any]]:
    policies: list[DQNPolicy] = []
    training_audit: dict[str, Any] = {}
    frozen_names = {
        "share": "share_h",
        "share_wo_budget": "share_ddqn",
        "share_dual": "full_share",
    }
    frozen_seed = json.loads((frozen / f"seed_{seed}.json").read_text(encoding="utf-8"))
    frozen_diagnostics = frozen_seed["diagnostics"]
    for name in POLICIES:
        encoder = risk_encoder if name == "share_wo_risk" else base_encoder
        if name in frozen_names:
            source = frozen_names[name]
            path = frozen / f"{source}_seed_{seed}.pt"
            policy = DQNPolicy.load(str(path), encoder, reward, require_reward_config=True)
            assert_frozen_config(name, policy, seed)
            training_audit[name] = {
                "source": "frozen_checkpoint",
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "diagnostics": frozen_diagnostics[source],
            }
        else:
            path = out / f"{name}_seed_{seed}.pt"
            if path.exists():
                policy = DQNPolicy.load(str(path), encoder, reward, require_reward_config=True)
                expected = ablation_config(name, seed, epochs, device)
                if asdict(policy.config) != asdict(expected):
                    raise RuntimeError(f"existing {name}/seed={seed} checkpoint has changed config")
                diagnostics = []
                source = "resumed_ablation_checkpoint"
            else:
                policy, diagnostics = train_ablation(
                    name, seed, train, validation, encoder, reward, epochs, device, path
                )
                source = "trained_on_train_validation_only"
            training_audit[name] = {
                "source": source,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "diagnostics": diagnostics,
            }
        policy.name = name
        training_audit[name].update({
            "config": asdict(policy.config),
            "encoder_disabled_groups": list(encoder.disabled_groups),
            "beta_h": float(policy.beta_h),
            "beta_u": float(policy.beta_u),
            "human_calibration_penalty": float(policy.human_calibration_penalty),
            "ai_calibration_penalty": float(policy.ai_calibration_penalty),
            "calibration_human_target": float(policy.calibration_human_target),
            "calibration_unsafe_target": float(policy.calibration_unsafe_target),
            "calibration_validation_hir": float(policy.calibration_validation_hir),
            "calibration_validation_unsafe": float(policy.calibration_validation_unsafe),
        })
        policies.append(policy)
    return policies, training_audit


def first_exhaustion_index(events: Sequence[Mapping[str, Any]]) -> int | None:
    return next((index for index, event in enumerate(events) if event["exhausted_before_decision"]), None)


def evaluate_seed(
    seed: int,
    policies: Sequence[DQNPolicy],
    evaluation: Sequence[RepairEpisode],
    reward: RewardConfig,
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    audits = {}
    for policy in policies:
        wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
        result = evaluate_policy(wrapper, evaluation, reward, 0.20, None)
        human_actions = sum(
            int(action == int(AuthorityAction.HUMAN))
            for record in result["records"] for action in record["actions"]
        )
        if wrapper.human_interventions > cap:
            raise RuntimeError(f"{policy.name}/seed={seed}: hard cap violated")
        result.update({
            "seed": seed,
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "human_actions": human_actions,
            "cap_respected": wrapper.human_interventions <= cap,
            "synthetic_development_evaluation": True,
        })
        audits[policy.name] = {
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "human_actions": human_actions,
            "cap_respected": wrapper.human_interventions <= cap,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "forced_autonomous_fallbacks": wrapper.forced_autonomous_fallbacks,
            "value_policy_replans": wrapper.value_policy_replans,
            "first_exhaustion_event_index": first_exhaustion_index(wrapper.events),
            "events": wrapper.events,
        }
        results.append(result)
    return results, audits


def aggregate(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload["results"]:
            buckets[row["policy"]].append(row)
    output = []
    for policy in POLICIES:
        rows = buckets[policy]
        item: dict[str, Any] = {"policy": policy, "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values)
        for metric in ("human_interventions", "human_actions"):
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
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
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
        eligible = [record for record in records if not record["human_used"] and int(record["actions"][-1]) != int(AuthorityAction.STOP)]
        return mean(float(record["unsafe"] or not record["success"]) for record in eligible) if eligible else float("nan")
    if metric == "collaboration_regret":
        values = [
            float(oracle) - float(selected)
            for record in records
            for selected, oracle in zip(record["selected_returns"], record["oracle_returns"])
        ]
        return mean(values)
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
    group_episode_ids = {
        group: {episode_id for episode_id, episode in episode_by_id.items() if episode.group_id == group}
        for group in groups
    }
    rng = np.random.default_rng(20260826)
    output = []
    for ablation in ABLATIONS:
        for metric in METRICS:
            differences = [
                float(by_seed[seed][ablation][metric]) - float(by_seed[seed]["share"][metric])
                for seed in SEEDS
            ]
            indices = rng.integers(0, len(SEEDS), size=(20_000, len(SEEDS)))
            seed_boot = np.asarray(differences)[indices].mean(axis=1)
            group_boot = []
            for _ in range(5_000):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                seed_diffs = []
                for seed in SEEDS:
                    per_policy = {}
                    for policy in ("share", ablation):
                        records_by_id = {
                            str(record["episode_id"]): record for record in by_seed[seed][policy]["records"]
                        }
                        sampled_records = [
                            records_by_id[episode_id]
                            for group in sampled for episode_id in group_episode_ids[group]
                        ]
                        per_policy[policy] = record_metric(sampled_records, metric)
                    if not (math.isnan(per_policy[ablation]) or math.isnan(per_policy["share"])):
                        seed_diffs.append(per_policy[ablation] - per_policy["share"])
                if seed_diffs:
                    group_boot.append(mean(seed_diffs))
            output.append({
                "comparison": f"{ablation}-minus-share",
                "metric": metric,
                "per_seed_differences": differences,
                "mean_difference": mean(differences),
                "std_difference": stdev(differences),
                "seed_bootstrap_ci95": [float(np.percentile(seed_boot, 2.5)), float(np.percentile(seed_boot, 97.5))],
                "workflow_group_cluster_bootstrap_ci95": [float(np.percentile(group_boot, 2.5)), float(np.percentile(group_boot, 97.5))],
                "exact_sign_flip_p_two_sided": exact_sign_flip(differences),
            })
    primary_indices = [index for index, row in enumerate(output) if row["metric"] in PRIMARY_METRICS]
    adjusted = holm_adjust([output[index]["exact_sign_flip_p_two_sided"] for index in primary_indices])
    for index, p_value in zip(primary_indices, adjusted):
        output[index]["holm_p_across_16_primary_tests"] = p_value
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen final SHARE development ablation.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--frozen", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--run-seeds", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    run_seeds = tuple(int(value) for value in args.run_seeds.split(",")) if args.run_seeds else seeds
    if seeds != SEEDS or not set(run_seeds).issubset(SEEDS):
        raise SystemExit(f"protocol requires exact seeds {SEEDS}; run-seeds may only select a subset")
    if args.epochs != 30:
        raise SystemExit("final protocol requires exactly 30 training epochs")
    frozen_manifest = json.loads((args.frozen / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != frozen_manifest["episode_sha256"]:
        raise SystemExit("dataset differs from frozen SHARE-H")
    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    if any(not values for values in split.values()):
        raise SystemExit("train/validation/development-evaluation splits must all be present")
    evaluation = sorted(split["test"], key=lambda episode: episode.episode_id)
    episode_by_id = {episode.episode_id: episode for episode in evaluation}
    order = [episode.episode_id for episode in evaluation]
    cap = math.floor(0.20 * len(evaluation))
    base_encoder = FeatureEncoder.load(args.frozen / "feature_encoder.json")
    risk_encoder = base_encoder.without("risk")
    reward = RewardConfig(**frozen_manifest["reward_config"])
    frozen_checkpoints = {
        name: {
            str(seed): sha256_file(args.frozen / f"{source}_seed_{seed}.pt")
            for seed in SEEDS
        }
        for name, source in {"share": "share_h", "share_wo_budget": "share_ddqn", "share_dual": "full_share"}.items()
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "synthetic_development_ablation",
        "locked_holdout_accessed": False,
        "development_split_name_in_dataset": "test",
        "development_evaluation_already_inspected": True,
        "training_splits_allowed": ["train", "validation"],
        "seeds": list(SEEDS),
        "epochs": 30,
        "device": args.device,
        "policies": list(POLICIES),
        "definitions": {
            "share": "frozen SHARE-H checkpoint; gamma=.95, full features, Human dual, reward-side safety",
            "share_myopic": "only training gamma changed from .95 to 0; otherwise SHARE-H",
            "share_wo_risk": "only drift_risk, uncertainty, and evidence_conflict encoder coordinates zeroed; otherwise SHARE-H",
            "share_wo_budget": "frozen unconstrained DDQN checkpoint with the same validation-only Human calibration; identical hard cap",
            "share_dual": "frozen original joint Human+unsafe dual checkpoint and calibration",
        },
        "human_budget": 0.20,
        "evaluation_episodes": len(evaluation),
        "hard_human_episode_cap": cap,
        "episode_order": order,
        "episode_order_sha256": sha256_json(order),
        "split_counts": {name: len(values) for name, values in split.items()},
        "episodes_sha256": sha256_file(args.episodes),
        "reward_config": asdict(reward),
        "base_encoder_sha256": sha256_file(args.frozen / "feature_encoder.json"),
        "risk_disabled_groups": list(risk_encoder.disabled_groups),
        "frozen_checkpoint_sha256": frozen_checkpoints,
        "fitting_or_recalibration_during_evaluation": False,
        "source_sha256": {
            "scripts/run_share_final_ablation.py": sha256_file(Path(__file__)),
            "scripts/run_share_online_hard_cap.py": sha256_file(ROOT / "scripts/run_share_online_hard_cap.py"),
            "share/dqn.py": sha256_file(ROOT / "share/dqn.py"),
            "share/features.py": sha256_file(ROOT / "share/features.py"),
            "share/evaluation.py": sha256_file(ROOT / "share/evaluation.py"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != protocol["protocol_sha256"]:
            raise SystemExit("refusing to resume changed ablation protocol")
    else:
        atomic_json(manifest_path, protocol)
        base_encoder.save(args.out / "feature_encoder.json")
        risk_encoder.save(args.out / "feature_encoder_wo_risk.json")
    for seed in run_seeds:
        seed_path = args.out / f"seed_{seed}.json"
        if seed_path.exists():
            print(json.dumps({"seed": seed, "status": "already complete"}), flush=True)
            continue
        policies, training_audit = load_policies(
            seed, split["train"], split["validation"], base_encoder, risk_encoder, reward,
            args.epochs, args.device, args.frozen, args.out,
        )
        results, budget_audit = evaluate_seed(seed, policies, evaluation, reward, cap)
        atomic_json(seed_path, {
            "seed": seed,
            "results": results,
            "training_audit": training_audit,
            "budget_audit": budget_audit,
        })
        print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
    payloads = [
        json.loads((args.out / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS if (args.out / f"seed_{seed}.json").exists()
    ]
    if len(payloads) == len(SEEDS):
        aggregate_rows = aggregate(payloads)
        paired = paired_statistics(payloads, episode_by_id)
        atomic_json(args.out / "aggregate.json", aggregate_rows)
        atomic_json(args.out / "paired_comparisons.json", paired)
        per_seed = [
            {key: value for key, value in row.items() if key != "records"}
            for payload in payloads for row in payload["results"]
        ]
        write_csv(args.out / "per_seed.csv", per_seed)
        write_csv(args.out / "aggregate.csv", aggregate_rows)
        write_csv(args.out / "paired_comparisons.csv", paired)
    print(json.dumps({"out": str(args.out), "completed_seeds": len(payloads), "complete": len(payloads) == len(SEEDS), "cap": cap}))


if __name__ == "__main__":
    main()
