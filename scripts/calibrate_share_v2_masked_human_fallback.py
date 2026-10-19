from __future__ import annotations

"""Validation-only robust fallback calibration for frozen SHARE-V2.

The trained Q networks and their ordinary three-action decisions remain
unchanged.  The only candidate parameter is a Stop penalty applied when the
online hard-cap evaluator masks Human and requests a second AI/Stop decision.
Selection uses 0%, 10%, and 20% validation hard-cap stress points across all
five seeds.  The already inspected development-test split is reported only
after selection, and no locked holdout is loaded.
"""

import argparse
import csv
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

from scripts.audit_share_v2_validation_hard_cap import (
    CHECKPOINTS,
    DEVELOPMENT_EPISODES_SHA256,
    ENCODER,
    EXPECTED_SPLIT_COUNTS,
    METRICS,
    SEEDS,
    atomic_json,
    sha256_file,
)
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy, deterministic_action
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


PROTOCOL = "share-v2-masked-human-fallback-calibration-v1"
CANDIDATE_STOP_PENALTIES = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
VALIDATION_BUDGETS = (0.0, 0.10, 0.20)
MIN_REPAIR_GAIN = 0.005
MAX_UAR_INCREASE = 0.02


class MaskedHumanFallbackPolicy(Policy):
    """Adjust AI/Stop only when the caller has made Human infeasible."""

    name = "share_v2_masked_human_fallback"

    def __init__(self, base: DQNPolicy, stop_penalty: float):
        self.base = base
        self.stop_penalty = float(stop_penalty)

    def escalation_score(self, state: RepairState) -> float:
        return self.base.escalation_score(state)

    def select(
        self, state: RepairState, feasible: Sequence[AuthorityAction]
    ) -> AuthorityAction:
        if AuthorityAction.HUMAN in feasible or self.stop_penalty <= 0.0:
            return self.base.select(state, feasible)
        values = self.base.values(state)
        values[AuthorityAction.STOP] -= self.stop_penalty
        return deterministic_action(values, feasible)


def evaluate(
    policy: DQNPolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
    budget: float,
    stop_penalty: float,
) -> dict[str, Any]:
    cap = math.floor(budget * len(episodes))
    calibrated = MaskedHumanFallbackPolicy(policy, stop_penalty)
    wrapper = OnlineHardHumanCapPolicy(calibrated, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, reward, budget, None)
    row = {key: value for key, value in raw.items() if key != "records"}
    row.update(
        {
            "nominal_budget": budget,
            "stop_penalty": stop_penalty,
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "value_policy_replans": wrapper.value_policy_replans,
            "cap_respected": wrapper.human_interventions <= cap,
        }
    )
    if not row["cap_respected"]:
        raise RuntimeError("online Human hard cap was violated")
    return row


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for penalty in CANDIDATE_STOP_PENALTIES:
        for budget in VALIDATION_BUDGETS:
            items = [
                row
                for row in rows
                if float(row["stop_penalty"]) == penalty
                and float(row["nominal_budget"]) == budget
            ]
            item: dict[str, Any] = {
                "stop_penalty": penalty,
                "nominal_budget": budget,
                "seeds": len(items),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in items]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            item["attempted_human_after_exhaustion_mean"] = mean(
                int(row["attempted_human_after_exhaustion"]) for row in items
            )
            item["all_caps_respected"] = all(bool(row["cap_respected"]) for row in items)
            output.append(item)
    return output


def select_candidate(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_candidate = {
        penalty: {
            float(row["nominal_budget"]): row
            for row in aggregates
            if float(row["stop_penalty"]) == penalty
        }
        for penalty in CANDIDATE_STOP_PENALTIES
    }
    baseline = by_candidate[0.0]
    baseline_repair = mean(
        float(baseline[budget]["repair_sr_mean"]) for budget in VALIDATION_BUDGETS
    )

    ranked: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for penalty, rows in by_candidate.items():
        primary = rows[0.20]
        mean_repair = mean(float(rows[b]["repair_sr_mean"]) for b in VALIDATION_BUDGETS)
        mean_return = mean(float(rows[b]["normalized_return_mean"]) for b in VALIDATION_BUDGETS)
        mean_stop = mean(float(rows[b]["stopped_rate_mean"]) for b in VALIDATION_BUDGETS)
        stress_uar_safe = all(
            float(rows[b]["uar_mean"])
            <= float(baseline[b]["uar_mean"]) + MAX_UAR_INCREASE + 1e-12
            for b in VALIDATION_BUDGETS
        )
        gate = (
            bool(primary["all_caps_respected"])
            and float(primary["hir_mean"]) <= 0.20 + 1e-12
            and float(primary["uar_mean"]) <= 0.12 + 1e-12
            and float(primary["stopped_rate_mean"]) <= 0.05 + 1e-12
            and stress_uar_safe
        )
        summary = {
            "stop_penalty": penalty,
            "passes_gate": gate,
            "mean_repair_across_validation_budgets": mean_repair,
            "repair_gain_vs_zero": mean_repair - baseline_repair,
            "mean_return_across_validation_budgets": mean_return,
            "mean_stop_across_validation_budgets": mean_stop,
        }
        key = (mean_repair, mean_return, -mean_stop, -penalty)
        if gate:
            ranked.append((key, penalty, summary))

    eligible = [item for item in ranked if item[2]["repair_gain_vs_zero"] >= MIN_REPAIR_GAIN]
    if not eligible:
        selected_penalty = 0.0
        reason = "no candidate cleared the gate with >=0.5 percentage-point mean Repair.SR gain"
    else:
        selected_penalty = max(eligible, key=lambda item: item[0])[1]
        reason = "highest validation stress-test mean Repair.SR among gated candidates"
    summaries = [item[2] for item in sorted(ranked, key=lambda item: item[1])]
    return {
        "selected_stop_penalty": selected_penalty,
        "selection_reason": reason,
        "baseline_mean_repair": baseline_repair,
        "candidate_summaries": summaries,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/share_v2_masked_fallback_calibration_20260827"),
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
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "FINAL_RESULTS.json").exists():
        raise SystemExit("refusing to rerun a completed calibration")

    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    encoder = FeatureEncoder.load(ENCODER)
    manifest = {
        "protocol": PROTOCOL,
        "evidence_class": "post_locked_development_candidate_only",
        "locked_holdout_accessed": False,
        "selection_split": "validation_only",
        "development_test_used_for_selection": False,
        "candidate_stop_penalties": list(CANDIDATE_STOP_PENALTIES),
        "validation_budgets": list(VALIDATION_BUDGETS),
        "gate": "20% HIR<=.20, UAR<=.12, Stop<=.05, zero cap violations; each stress-budget UAR increase<=.02",
        "minimum_repair_gain": MIN_REPAIR_GAIN,
        "ranking": "mean Repair.SR over validation 0/10/20%; then Return, Stop, and smaller penalty",
        "episodes_sha256": DEVELOPMENT_EPISODES_SHA256,
        "split_counts": counts,
        "seeds": list(SEEDS),
        "training_reward": asdict(training_reward),
        "evaluation_reward": asdict(evaluation_reward),
        "encoder_sha256": sha256_file(ENCODER),
        "checkpoint_sha256": {str(seed): sha256_file(path) for seed, path in CHECKPOINTS.items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.out / "pre_evaluation_manifest.json", manifest)

    policies = {
        seed: DQNPolicy.load(
            str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True
        )
        for seed in SEEDS
    }
    validation_rows: list[dict[str, Any]] = []
    for penalty in CANDIDATE_STOP_PENALTIES:
        for budget in VALIDATION_BUDGETS:
            for seed in SEEDS:
                row = evaluate(
                    policies[seed], split["validation"], evaluation_reward, budget, penalty
                )
                row["seed"] = seed
                validation_rows.append(row)
    validation_aggregate = aggregate(validation_rows)
    selection = select_candidate(validation_aggregate)

    selected = float(selection["selected_stop_penalty"])
    development_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        row = evaluate(
            policies[seed], split["test"], evaluation_reward, 0.20, selected
        )
        row["seed"] = seed
        development_rows.append(row)

    result = {
        "protocol": PROTOCOL,
        "locked_holdout_accessed": False,
        "selection": selection,
        "validation_aggregate": validation_aggregate,
        "development_test_diagnostic": development_rows,
        "interpretation_boundary": "Development-only post-hoc redesign; confirmation requires a new untouched holdout.",
    }
    atomic_json(args.out / "FINAL_RESULTS.json", result)
    write_csv(args.out / "validation_aggregate.csv", validation_aggregate)
    write_csv(args.out / "development_test_diagnostic.csv", development_rows)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
