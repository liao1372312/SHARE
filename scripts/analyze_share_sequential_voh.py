from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNPolicy
from share.environment import base_reward
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


SEEDS = (17, 29, 43, 71, 101)
POLICIES = ("share_static", "share_bandit", "share_ddqn", "share_h")
COMPARATORS = POLICIES[:-1]
METRICS = ("repair_sr", "uar", "stopped_rate", "collaboration_regret")
COMPLEXITY_BINS = ("C_low", "C_mid", "C_high")
HORIZONS = (1, 2, 3)
VOH_LABELS = ("Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complexity_bin(value: float) -> str:
    if value < 0.5:
        return "C_low"
    if value < 0.8:
        return "C_mid"
    return "C_high"


def safe_mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return mean(items) if items else None


def record_metrics(
    records: Sequence[Mapping[str, Any]],
    episode_by_id: Mapping[str, RepairEpisode],
    audit: Mapping[str, Any],
) -> dict[str, float | int | None]:
    autonomous = [
        record for record in records
        if not record["human_used"] and int(record["actions"][-1]) != int(AuthorityAction.STOP)
    ]
    regrets = [
        float(oracle) - float(selected)
        for record in records
        for selected, oracle in zip(record["selected_returns"], record["oracle_returns"])
    ]
    episode_ids = {str(record["episode_id"]) for record in records}
    events = [event for event in audit["events"] if str(event["episode_id"]) in episode_ids]
    return {
        "episodes": len(records),
        "workflow_groups": len({episode_by_id[str(record["episode_id"])].group_id for record in records}),
        "decision_states": sum(len(record["actions"]) for record in records),
        "repair_sr": safe_mean(float(record["success"]) for record in records),
        "uar": safe_mean(float(record["unsafe"] or not record["success"]) for record in autonomous),
        "uar_denominator": len(autonomous),
        "stopped_rate": safe_mean(float(int(record["actions"][-1]) == int(AuthorityAction.STOP)) for record in records),
        "collaboration_regret": safe_mean(regrets),
        "human_interventions": sum(
            int(event["executed_action"] == int(AuthorityAction.HUMAN)) for event in events
        ),
        "attempted_human_after_exhaustion": sum(
            int(event["exhausted_before_decision"] and event["proposed_action"] == int(AuthorityAction.HUMAN))
            for event in events
        ),
        "forced_autonomous_fallbacks": sum(int(event["fallback"] == "binary_autonomous") for event in events),
        "value_policy_replans": sum(int(event["fallback"] == "value_best_non_human") for event in events),
        "stop_count": sum(int(record["actions"][-1]) == int(AuthorityAction.STOP) for record in records),
        "full_run_human_interventions": int(audit["human_interventions"]),
        "hard_human_cap": int(audit["hard_human_cap"]),
        "cap_respected": int(audit["human_interventions"]) <= int(audit["hard_human_cap"]),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = dict(zip(keys, key))
        result["seeds"] = len(values)
        for metric in (*METRICS, "uar_denominator"):
            present = [float(row[metric]) for row in values if row.get(metric) is not None]
            result[f"{metric}_mean"] = mean(present) if present else None
            result[f"{metric}_std"] = stdev(present) if len(present) > 1 else 0.0 if present else None
        for count in ("episodes", "workflow_groups", "human_interventions", "attempted_human_after_exhaustion",
                      "forced_autonomous_fallbacks", "value_policy_replans", "stop_count"):
            result[f"{count}_mean"] = safe_mean(row.get(count, 0) for row in values)
        result["decision_states_mean"] = safe_mean(row.get("decision_states", 0) for row in values)
        output.append(result)
    return output


def exact_sign_flip(values: Sequence[float], alternative: str = "two-sided") -> float:
    observed = mean(values)
    permuted = [
        mean([sign * value for sign, value in zip(signs, values)])
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    if alternative == "greater":
        return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)
    if alternative == "less":
        return sum(value <= observed + 1e-15 for value in permuted) / len(permuted)
    return sum(abs(value) >= abs(observed) - 1e-15 for value in permuted) / len(permuted)


def paired_statistics(rows: Sequence[Mapping[str, Any]], stratum_key: str) -> list[dict[str, Any]]:
    by = {(row["seed"], row[stratum_key], row["policy"]): row for row in rows}
    strata = sorted({row[stratum_key] for row in rows}, key=str)
    rng = np.random.default_rng(20260826)
    output = []
    for stratum in strata:
        for comparator in COMPARATORS:
            for metric in METRICS:
                differences = []
                for seed in SEEDS:
                    left = by[(seed, stratum, comparator)].get(metric)
                    right = by[(seed, stratum, "share_h")].get(metric)
                    if left is not None and right is not None:
                        differences.append(float(left) - float(right))
                if not differences:
                    continue
                indices = rng.integers(0, len(differences), size=(20_000, len(differences)))
                boot = np.asarray(differences)[indices].mean(axis=1)
                output.append(
                    {
                        "stratum_type": stratum_key,
                        "stratum": stratum,
                        "comparison": f"{comparator}-minus-share_h",
                        "metric": metric,
                        "per_seed": differences,
                        "mean": mean(differences),
                        "std": stdev(differences) if len(differences) > 1 else 0.0,
                        "ci95_seed_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "p_exact_sign_flip_two_sided": exact_sign_flip(differences),
                        "seeds": len(differences),
                    }
                )
    return output


def visited_steps(episode: RepairEpisode, actions: Sequence[int]) -> list[int]:
    indices = []
    current = 0
    for raw_action in actions:
        indices.append(current)
        action = AuthorityAction(int(raw_action))
        outcome = episode.steps[current].outcome(action)
        if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None:
            current = outcome.next_step
    return indices


def forced_ai_return(
    episode: RepairEpisode,
    step: int,
    reward: RewardConfig,
    gamma: float = 0.95,
) -> tuple[float, bool, bool]:
    """Return from choosing AI now and continuing with AI at every reachable step."""
    action = AuthorityAction.AI
    outcome = episode.steps[step].outcome(action)
    value = base_reward(outcome, action, reward)
    success, unsafe = outcome.success, bool(outcome.unsafe or (action == AuthorityAction.AI and outcome.terminal and not outcome.success))
    if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None:
        continuation, success, later_unsafe = forced_ai_return(episode, outcome.next_step, reward, gamma)
        value += gamma * continuation
        unsafe = unsafe or later_unsafe
    return value, bool(success), bool(unsafe)


def forced_human_return(
    episode: RepairEpisode,
    step: int,
    reward: RewardConfig,
) -> tuple[float, bool, bool]:
    """Human is terminal in the paired simulator, so its downstream value is immediate."""
    action = AuthorityAction.HUMAN
    outcome = episode.steps[step].outcome(action)
    return (
        base_reward(outcome, action, reward),
        bool(outcome.success),
        bool(outcome.unsafe),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def voh_analysis(
    seeds: Sequence[int],
    episodes: Sequence[RepairEpisode],
    checkpoints: Path,
    encoder: FeatureEncoder,
    reward: RewardConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    observations = []
    bin_rows = []
    per_seed_rho = []
    per_seed_slopes = []
    for seed in seeds:
        policy = DQNPolicy.load(
            str(checkpoints / f"share_h_seed_{seed}.pt"), encoder, reward,
            require_reward_config=True,
        )
        states = [step.state for episode in episodes for step in episode.steps]
        q = policy._raw_values_batch(states)
        offset = 0
        seed_rows = []
        for episode in episodes:
            for step_index, paired in enumerate(episode.steps):
                predicted = float(q[offset, AuthorityAction.HUMAN] - q[offset, AuthorityAction.AI])
                human_value, human_success, human_unsafe = forced_human_return(episode, step_index, reward)
                ai_value, ai_success, ai_unsafe = forced_ai_return(episode, step_index, reward)
                row = {
                    "seed": seed,
                    "episode_id": episode.episode_id,
                    "group_id": episode.group_id,
                    "step": step_index,
                    "complexity": complexity_bin(paired.state.complexity),
                    "remaining_horizon": len(episode.steps) - step_index,
                    "predicted_voh": predicted,
                    "realized_downstream_reward_advantage": human_value - ai_value,
                    "realized_downstream_success_advantage": float(human_success) - float(ai_success),
                    "realized_downstream_safety_advantage": float(ai_unsafe) - float(human_unsafe),
                }
                seed_rows.append(row)
                offset += 1
        if len(seed_rows) % 5:
            raise ValueError(f"strict VoH quintiles require state count divisible by five; got {len(seed_rows)}")
        per_bin = len(seed_rows) // 5
        ranked = sorted(seed_rows, key=lambda row: (row["predicted_voh"], row["episode_id"], row["step"]))
        for rank, row in enumerate(ranked):
            bin_index = rank // per_bin
            row["voh_rank"] = rank + 1
            row["voh_bin"] = bin_index + 1
            row["voh_bin_label"] = VOH_LABELS[bin_index]
        observations.extend(seed_rows)
        rho = float(spearmanr(
            [row["predicted_voh"] for row in seed_rows],
            [row["realized_downstream_reward_advantage"] for row in seed_rows],
        ).statistic)
        per_seed_rho.append(rho)
        for bin_index in range(1, 6):
            selected = [row for row in seed_rows if row["voh_bin"] == bin_index]
            bin_rows.append(
                {
                    "seed": seed,
                    "voh_bin": bin_index,
                    "voh_bin_label": VOH_LABELS[bin_index - 1],
                    "states": len(selected),
                    "predicted_voh_mean": safe_mean(row["predicted_voh"] for row in selected),
                    "realized_downstream_reward_advantage": safe_mean(row["realized_downstream_reward_advantage"] for row in selected),
                    "realized_downstream_success_advantage": safe_mean(row["realized_downstream_success_advantage"] for row in selected),
                    "realized_downstream_safety_advantage": safe_mean(row["realized_downstream_safety_advantage"] for row in selected),
                }
            )
        advantages = [
            next(row for row in bin_rows if row["seed"] == seed and row["voh_bin"] == bin_index)["realized_downstream_reward_advantage"]
            for bin_index in range(1, 6)
        ]
        per_seed_slopes.append(float(np.polyfit(np.arange(1, 6), advantages, 1)[0]))
    aggregate_bins = []
    for bin_index in range(1, 6):
        selected = [row for row in bin_rows if row["voh_bin"] == bin_index]
        item = {"voh_bin": bin_index, "voh_bin_label": VOH_LABELS[bin_index - 1], "seeds": len(selected)}
        for metric in (
            "states", "predicted_voh_mean", "realized_downstream_reward_advantage",
            "realized_downstream_success_advantage", "realized_downstream_safety_advantage",
        ):
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate_bins.append(item)
    bin_advantages = [row["realized_downstream_reward_advantage_mean"] for row in aggregate_bins]
    rng = np.random.default_rng(20260826)
    rho_boot = []
    groups = sorted({row["group_id"] for row in observations})
    by_seed_group = defaultdict(list)
    for row in observations:
        by_seed_group[(row["seed"], row["group_id"])].append(row)
    for _ in range(5_000):
        sample_groups = rng.choice(groups, size=len(groups), replace=True)
        seed_rhos = []
        for seed in seeds:
            sample = [row for group in sample_groups for row in by_seed_group[(seed, group)]]
            seed_rhos.append(float(spearmanr(
                [row["predicted_voh"] for row in sample],
                [row["realized_downstream_reward_advantage"] for row in sample],
            ).statistic))
        rho_boot.append(mean(seed_rhos))
    statistics = {
        "estimand": "raw Q(Human)-Q(AI) versus G_H-G_AI, where Human terminates and AI is recursively continued with AI (gamma=0.95)",
        "bin_definition": "strict within-seed ranks of evaluation predictions only; ties broken by (episode_id, step); exactly 51 states per quintile",
        "observations": len(observations),
        "workflow_groups": len(groups),
        "per_seed_spearman_rho": dict(zip(map(str, seeds), per_seed_rho)),
        "spearman_rho_mean": mean(per_seed_rho),
        "spearman_rho_std": stdev(per_seed_rho),
        "spearman_rho_one_sided_exact_sign_flip_p": exact_sign_flip(per_seed_rho, alternative="greater"),
        "workflow_group_cluster_bootstrap_mean_rho_ci95": [float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5))],
        "per_seed_bin_mean_slope": dict(zip(map(str, seeds), per_seed_slopes)),
        "bin_mean_slope_mean": mean(per_seed_slopes),
        "bin_mean_slope_std": stdev(per_seed_slopes),
        "bin_mean_slope_one_sided_exact_sign_flip_p": exact_sign_flip(per_seed_slopes, alternative="greater"),
        "adjacent_bin_increases": sum(right > left for left, right in zip(bin_advantages, bin_advantages[1:])),
        "possible_adjacent_increases": 4,
        "highest_minus_lowest_bin_advantage": bin_advantages[-1] - bin_advantages[0],
    }
    return observations, bin_rows, aggregate_bins, statistics


def sequential_hypothesis(rows: Sequence[Mapping[str, Any]], stratum_key: str, ordered_strata: Sequence[Any]) -> dict[str, Any]:
    by = {(row["seed"], row[stratum_key], row["policy"]): row for row in rows}
    deltas = {}
    for stratum in ordered_strata:
        values = [
            float(by[(seed, stratum, "share_h")]["repair_sr"]) - float(by[(seed, stratum, "share_bandit")]["repair_sr"])
            for seed in SEEDS
        ]
        deltas[str(stratum)] = {"per_seed": values, "mean": mean(values), "std": stdev(values)}
    interaction = [
        deltas[str(ordered_strata[-1])]["per_seed"][index] - deltas[str(ordered_strata[0])]["per_seed"][index]
        for index in range(len(SEEDS))
    ]
    return {
        "claim": "SHARE-H minus SHARE-Bandit Repair.SR grows with sequential complexity",
        "repair_deltas": deltas,
        "highest_minus_lowest_per_seed": interaction,
        "highest_minus_lowest_mean": mean(interaction),
        "highest_minus_lowest_std": stdev(interaction),
        "one_sided_exact_sign_flip_p": exact_sign_flip(interaction, alternative="greater"),
        "reverse_direction_one_sided_exact_sign_flip_p": exact_sign_flip(interaction, alternative="less"),
        "direction_supported_descriptively": mean(interaction) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen sequential-VoH mechanism analysis.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--hard-cap-results", type=Path, default=Path("results/share_online_hard_cap_20pct_5seed_20260825"))
    parser.add_argument("--checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_manifest = json.loads((args.checkpoints / "manifest.json").read_text(encoding="utf-8"))
    hard_cap_manifest = json.loads((args.hard_cap_results / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != checkpoint_manifest["episode_sha256"]:
        raise SystemExit("dataset differs from frozen checkpoints")
    episodes_all = load_episodes(args.episodes)
    evaluation = [episode for episode in episodes_all if episode.split == "test"]
    episode_by_id = {episode.episode_id: episode for episode in evaluation}
    complexity_rows, horizon_rows = [], []
    for seed in SEEDS:
        payload = json.loads((args.hard_cap_results / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for result in payload["results"]:
            policy = result["policy"]
            if policy not in POLICIES:
                continue
            records = result["records"]
            audit = payload["budget_audit"][policy]
            for label in COMPLEXITY_BINS:
                selected = [
                    record for record in records
                    if complexity_bin(episode_by_id[record["episode_id"]].steps[0].state.complexity) == label
                ]
                complexity_rows.append({
                    "seed": seed, "policy": policy, "complexity": label,
                    **record_metrics(selected, episode_by_id, audit),
                })
            for horizon in HORIZONS:
                selected = [
                    record for record in records
                    if len(episode_by_id[record["episode_id"]].steps) == horizon
                ]
                horizon_rows.append({
                    "seed": seed, "policy": policy, "horizon": f"H{horizon}",
                    **record_metrics(selected, episode_by_id, audit),
                })
    complexity_aggregate = aggregate_rows(complexity_rows, ("complexity", "policy"))
    horizon_aggregate = aggregate_rows(horizon_rows, ("horizon", "policy"))
    complexity_paired = paired_statistics(complexity_rows, "complexity")
    horizon_paired = paired_statistics(horizon_rows, "horizon")
    encoder = FeatureEncoder.load(args.checkpoints / "feature_encoder.json")
    reward = RewardConfig(**checkpoint_manifest["reward_config"])
    voh_states, voh_bins_per_seed, voh_bins, voh_statistics = voh_analysis(
        SEEDS, evaluation, args.checkpoints, encoder, reward
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "complexity_per_seed.csv", complexity_rows)
    write_csv(args.out / "complexity_aggregate.csv", complexity_aggregate)
    write_csv(args.out / "horizon_per_seed.csv", horizon_rows)
    write_csv(args.out / "horizon_aggregate.csv", horizon_aggregate)
    write_csv(args.out / "voh_states.csv", voh_states)
    write_csv(args.out / "voh_bins_per_seed.csv", voh_bins_per_seed)
    write_csv(args.out / "voh_bins.csv", voh_bins)
    (args.out / "paired_differences.json").write_text(json.dumps({"complexity": complexity_paired, "horizon": horizon_paired}, indent=2), encoding="utf-8")
    hypothesis = {
        "complexity": sequential_hypothesis(complexity_rows, "complexity", COMPLEXITY_BINS),
        "initial_horizon": sequential_hypothesis(horizon_rows, "horizon", ("H2", "H3")),
    }
    (args.out / "sequential_hypothesis.json").write_text(json.dumps(hypothesis, indent=2), encoding="utf-8")
    (args.out / "voh_statistics.json").write_text(json.dumps(voh_statistics, indent=2), encoding="utf-8")
    manifest = {
        "protocol": "share-sequential-voh-mechanism-v1",
        "evidence": "synthetic development mechanism analysis",
        "locked_holdout_accessed": False,
        "training_or_recalibration": False,
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "human_budget": 0.20,
        "hard_cap": 24,
        "hard_cap_protocol_sha256": hard_cap_manifest["protocol_sha256"],
        "episodes_sha256": sha256_file(args.episodes),
        "complexity_bins": {"C_low": "complexity < 0.5", "C_mid": "0.5 <= complexity < 0.8", "C_high": "complexity >= 0.8"},
        "complexity_episode_counts": {label: sum(complexity_bin(episode.steps[0].state.complexity) == label for episode in evaluation) for label in COMPLEXITY_BINS},
        "initial_horizon_bins": ["H1", "H2", "H3"],
        "initial_horizon_episode_counts": {f"H{horizon}": sum(len(episode.steps) == horizon for episode in evaluation) for horizon in HORIZONS},
        "horizon_inference": "H2 versus H3 only; H1 (n=1) is descriptive and unsupported",
        "voh_bins": "strict within-seed rank quintiles from evaluation predictions only; 51 states per bin; no outcomes used",
        "voh_ai_counterfactual": "AI at the indexed state followed recursively by AI at every reachable downstream state; gamma=0.95",
        "stratum_estimand": "subsets of the original full 120-episode hard-cap trajectory; the cap is never reset within strata",
        "source_sha256": {
            "scripts/analyze_share_sequential_voh.py": sha256_file(Path(__file__)),
            "share/dqn.py": sha256_file(ROOT / "share/dqn.py"),
            "share/environment.py": sha256_file(ROOT / "share/environment.py"),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complexity_rows": len(complexity_rows), "horizon_rows": len(horizon_rows), "voh_states": len(voh_states)}))


if __name__ == "__main__":
    main()
