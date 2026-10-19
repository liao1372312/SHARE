"""Post-hoc VoH analysis for the frozen final SHARE confirmation checkpoints.

This script does not execute an authority policy, retrain a model, recalibrate a
threshold, or alter the opened confirmation results.  It loads the already
frozen Q-functions and computes Q(Human)-Q(AI) on the already opened paired
confirmation states.  The resulting evidence is explicitly post hoc and must
not be described as part of the original preregistered comparison family.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNPolicy
from share.environment import base_reward
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


DATASET = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
PRE_EVALUATION = ROOT / "results/share_v2_confirmation_locked_20260827/pre_evaluation_manifest.json"
ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
OUT = ROOT / "results/share_v2_confirmation_voh_posthoc_20260827"
SEEDS = (17, 29, 43, 71, 101)
CHECKPOINTS = {
    17: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/share_v2_seed_17.pt",
    29: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_29.pt",
    43: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_43.pt",
    71: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_71.pt",
    101: ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_101.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_flip(values: list[float], alternative: str = "greater") -> float:
    observed = mean(values)
    permuted = [
        mean(sign * value for sign, value in zip(signs, values))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    if alternative == "greater":
        return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)
    return sum(abs(value) >= abs(observed) - 1e-15 for value in permuted) / len(permuted)


def forced_ai_return(episode, step: int, reward: RewardConfig, gamma: float = 0.95) -> float:
    outcome = episode.steps[step].outcome(AuthorityAction.AI)
    value = base_reward(outcome, AuthorityAction.AI, reward)
    if not outcome.terminal and outcome.next_step is not None:
        value += gamma * forced_ai_return(episode, outcome.next_step, reward, gamma)
    return float(value)


def forced_human_return(episode, step: int, reward: RewardConfig) -> float:
    outcome = episode.steps[step].outcome(AuthorityAction.HUMAN)
    return float(base_reward(outcome, AuthorityAction.HUMAN, reward))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seal = json.loads(PRE_EVALUATION.read_text(encoding="utf-8"))
    if sha256(DATASET) != seal["dataset_sha256"]:
        raise SystemExit("confirmation dataset hash differs from the sealed manifest")
    if sha256(ENCODER) != seal["encoder_sha256"]:
        raise SystemExit("encoder hash differs from the sealed manifest")
    for seed, checkpoint in CHECKPOINTS.items():
        if sha256(checkpoint) != seal["checkpoints"][str(seed)]:
            raise SystemExit(f"checkpoint hash differs for seed {seed}")

    episodes = load_episodes(DATASET)
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    observations: list[dict] = []
    per_seed_rho: dict[str, float] = {}
    per_seed_slopes: dict[str, float] = {}
    bin_rows: list[dict] = []

    states = [step.state for episode in episodes for step in episode.steps]
    for seed in SEEDS:
        policy = DQNPolicy.load(
            str(CHECKPOINTS[seed]), encoder, training_reward,
            require_reward_config=True,
        )
        q_values = policy._raw_values_batch(states)
        seed_rows = []
        offset = 0
        for episode in episodes:
            for step_index, _ in enumerate(episode.steps):
                predicted = float(
                    q_values[offset, AuthorityAction.HUMAN]
                    - q_values[offset, AuthorityAction.AI]
                )
                realized = (
                    forced_human_return(episode, step_index, evaluation_reward)
                    - forced_ai_return(episode, step_index, evaluation_reward)
                )
                seed_rows.append({
                    "seed": seed,
                    "episode_id": episode.episode_id,
                    "group_id": episode.group_id,
                    "step": step_index,
                    "predicted_voh": predicted,
                    "realized_human_minus_ai_downstream_reward": realized,
                })
                offset += 1

        ranked = sorted(seed_rows, key=lambda row: (row["predicted_voh"], row["episode_id"], row["step"]))
        split_indices = np.array_split(np.arange(len(ranked)), 5)
        bin_advantages = []
        for bin_index, indices in enumerate(split_indices, start=1):
            selected = [ranked[int(index)] for index in indices]
            for row in selected:
                row["voh_bin"] = bin_index
            advantage = mean(row["realized_human_minus_ai_downstream_reward"] for row in selected)
            bin_advantages.append(advantage)
            bin_rows.append({
                "seed": seed,
                "voh_bin": bin_index,
                "states": len(selected),
                "predicted_voh_mean": mean(row["predicted_voh"] for row in selected),
                "realized_advantage_mean": advantage,
            })
        observations.extend(seed_rows)
        rho = float(spearmanr(
            [row["predicted_voh"] for row in seed_rows],
            [row["realized_human_minus_ai_downstream_reward"] for row in seed_rows],
        ).statistic)
        per_seed_rho[str(seed)] = rho
        per_seed_slopes[str(seed)] = float(np.polyfit(np.arange(1, 6), bin_advantages, 1)[0])

    aggregate_bins = []
    for bin_index in range(1, 6):
        selected = [row for row in bin_rows if row["voh_bin"] == bin_index]
        aggregate_bins.append({
            "voh_bin": bin_index,
            "states_total": sum(row["states"] for row in selected),
            "predicted_voh_mean": mean(row["predicted_voh_mean"] for row in selected),
            "predicted_voh_std": stdev(row["predicted_voh_mean"] for row in selected),
            "realized_advantage_mean": mean(row["realized_advantage_mean"] for row in selected),
            "realized_advantage_std": stdev(row["realized_advantage_mean"] for row in selected),
        })

    by_seed_group = defaultdict(list)
    for row in observations:
        by_seed_group[(int(row["seed"]), row["group_id"])].append(row)
    groups = sorted({row["group_id"] for row in observations})
    rng = np.random.default_rng(20260827)
    bootstrap_rhos = []
    for _ in range(5000):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        seed_rhos = []
        for seed in SEEDS:
            sample = [row for group in sampled_groups for row in by_seed_group[(seed, group)]]
            seed_rhos.append(float(spearmanr(
                [row["predicted_voh"] for row in sample],
                [row["realized_human_minus_ai_downstream_reward"] for row in sample],
            ).statistic))
        bootstrap_rhos.append(mean(seed_rhos))

    rho_values = list(per_seed_rho.values())
    slope_values = list(per_seed_slopes.values())
    statistics = {
        "evidence_status": "post_hoc_after_confirmation_opened_not_preregistered",
        "estimand": "frozen Q(Human)-Q(AI) versus replay Human return minus forced-AI downstream return under the common evaluation reward",
        "workflow_groups": len(groups),
        "states_per_seed": len(states),
        "observations_across_seeds": len(observations),
        "per_seed_spearman_rho": per_seed_rho,
        "spearman_rho_mean": mean(rho_values),
        "spearman_rho_std": stdev(rho_values),
        "spearman_rho_one_sided_exact_sign_flip_p": exact_sign_flip(rho_values),
        "workflow_group_cluster_bootstrap_mean_rho_ci95": [
            float(np.percentile(bootstrap_rhos, 2.5)),
            float(np.percentile(bootstrap_rhos, 97.5)),
        ],
        "per_seed_quintile_slope": per_seed_slopes,
        "quintile_slope_mean": mean(slope_values),
        "quintile_slope_std": stdev(slope_values),
        "quintile_slope_one_sided_exact_sign_flip_p": exact_sign_flip(slope_values),
        "highest_minus_lowest_bin_advantage": (
            aggregate_bins[-1]["realized_advantage_mean"]
            - aggregate_bins[0]["realized_advantage_mean"]
        ),
        "adjacent_bin_increases": sum(
            right["realized_advantage_mean"] > left["realized_advantage_mean"]
            for left, right in zip(aggregate_bins, aggregate_bins[1:])
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "voh_states.csv", observations)
    write_csv(OUT / "voh_bins_per_seed.csv", bin_rows)
    write_csv(OUT / "voh_bins.csv", aggregate_bins)
    (OUT / "voh_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    manifest = {
        "protocol": "share-v2-confirmation-voh-posthoc-v1",
        "evidence_status": statistics["evidence_status"],
        "training_or_recalibration": False,
        "policy_evaluation_rerun": False,
        "confirmation_outcomes_used_for_method_selection": False,
        "dataset_sha256": sha256(DATASET),
        "encoder_sha256": sha256(ENCODER),
        "checkpoint_sha256": {str(seed): sha256(path) for seed, path in CHECKPOINTS.items()},
        "pre_evaluation_manifest_sha256": sha256(PRE_EVALUATION),
        "analysis_script_sha256": sha256(Path(__file__)),
        "seeds": list(SEEDS),
        "training_reward_rollback_penalty": 0.60,
        "realized_advantage_reward_rollback_penalty": 0.25,
        "bootstrap_seed": 20260827,
        "bootstrap_samples": 5000,
        "interpretation_boundary": "exploratory mechanism evidence only; not a confirmatory test and not eligible for model selection",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "# Final SHARE confirmation VoH analysis (post hoc)",
        "",
        "This analysis was created after the independent confirmation set had been opened. It does not rerun policy evaluation, retrain, recalibrate, or select a model. It is exploratory mechanism evidence, not part of the original preregistered family.",
        "",
        f"- Mean Spearman rho: {statistics['spearman_rho_mean']:.4f} ± {statistics['spearman_rho_std']:.4f}",
        f"- Workflow-group bootstrap 95% CI: [{statistics['workflow_group_cluster_bootstrap_mean_rho_ci95'][0]:.4f}, {statistics['workflow_group_cluster_bootstrap_mean_rho_ci95'][1]:.4f}]",
        f"- One-sided exact five-seed sign-flip p: {statistics['spearman_rho_one_sided_exact_sign_flip_p']:.4f}",
        f"- Mean quintile slope: {statistics['quintile_slope_mean']:.4f} ± {statistics['quintile_slope_std']:.4f}",
        f"- Quintile-slope one-sided exact p: {statistics['quintile_slope_one_sided_exact_sign_flip_p']:.4f}",
        f"- Highest-minus-lowest realized advantage: {statistics['highest_minus_lowest_bin_advantage']:.4f}",
        f"- Adjacent increases: {statistics['adjacent_bin_increases']}/4",
        "",
        "## Quintiles",
        "",
        "| Bin | States | Predicted VoH | Realized Human-minus-AI advantage |",
        "|---:|---:|---:|---:|",
    ]
    for row in aggregate_bins:
        report.append(
            f"| {row['voh_bin']} | {row['states_total']} | "
            f"{row['predicted_voh_mean']:.4f} ± {row['predicted_voh_std']:.4f} | "
            f"{row['realized_advantage_mean']:.4f} ± {row['realized_advantage_std']:.4f} |"
        )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    sums = []
    for path in sorted(OUT.iterdir()):
        if path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
