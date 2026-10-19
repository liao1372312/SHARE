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
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNPolicy
from share.environment import RepairAuthorityEnv
from share.evaluation import evaluate_policy, immediate_reward_tensor
from share.features import FeatureEncoder
from share.policies import ContextualBanditPolicy, SupervisedBenefitPolicy, ThresholdPolicy
from share.types import AuthorityAction, RepairEpisode, RewardConfig


PROTOCOL_VERSION = "share-budget-matched-20pct-v1"
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
COMPARATORS = POLICIES[:-1]
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
PRIMARY_METRICS = ("repair_sr", "uar", "normalized_return", "collaboration_regret")


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


def episode_hir(policy: Any, episodes: Sequence[RepairEpisode]) -> float:
    """Resource-use-only rollout used for validation calibration."""
    human_episodes = 0
    for episode in episodes:
        env = RepairAuthorityEnv(episode)
        state = env.reset()
        human_used = False
        while not env.done:
            action = policy.select(state, env.feasible_actions())
            human_used = human_used or action == AuthorityAction.HUMAN
            result = env.step(action)
            if result.next_state is not None:
                state = result.next_state
        human_episodes += int(human_used)
    return human_episodes / len(episodes)


def breakpoint_candidates(values: Sequence[float]) -> list[float]:
    unique = sorted({float(value) for value in values if math.isfinite(float(value))})
    if not unique:
        raise ValueError("calibration requires at least one finite breakpoint")
    candidates = {unique[0] - 1.0, unique[-1] + 1.0}
    for value in unique:
        candidates.add(value)
        candidates.add(float(np.nextafter(value, -np.inf)))
        candidates.add(float(np.nextafter(value, np.inf)))
    return sorted(candidates)


def calibrate_scalar_validation_only(
    policy: Any,
    episodes: Sequence[RepairEpisode],
    candidates: Sequence[float],
    setter: Callable[[float], None],
    *,
    target: float,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Choose a scalar using validation HIR only; no quality or safety outcome is read."""
    cache: dict[int, dict[str, Any]] = {}

    def evaluate(index: int) -> dict[str, Any]:
        if index in cache:
            return cache[index]
        candidate = float(candidates[index])
        setter(float(candidate))
        hir = episode_hir(policy, episodes)
        in_band = lower - 1e-12 <= hir <= upper + 1e-12
        cache[index] = {
            "index": index,
            "candidate": candidate,
            "validation_hir": hir,
            "in_band": in_band,
        }
        return cache[index]

    # Increasing a Human threshold/penalty can only postpone or remove a Human
    # action, so episode-level HIR is monotone non-increasing. Locate the target
    # crossing, then inspect its neighboring exact breakpoints.
    low, high = 0, len(candidates) - 1
    while low <= high:
        middle = (low + high) // 2
        row = evaluate(middle)
        if row["validation_hir"] > target:
            low = middle + 1
        else:
            high = middle - 1
    inspect = {0, len(candidates) - 1}
    inspect.update(range(max(0, low - 4), min(len(candidates), low + 5)))
    for index in sorted(inspect):
        evaluate(index)

    best_key = None
    selected = None
    for index, row in sorted(cache.items()):
        hir = row["validation_hir"]
        key = (not row["in_band"], abs(hir - target), hir, index)
        if best_key is None or key < best_key:
            best_key = key
            selected = row["candidate"]
    assert selected is not None
    setter(selected)
    chosen = next(row for row in cache.values() if row["candidate"] == selected)
    return {
        "selection_split": "validation",
        "selection_metric": "episode_level_hir_only",
        "target_hir": target,
        "allowed_hir": [lower, upper],
        "candidate_count": len(candidates),
        "evaluated_candidate_count": len(cache),
        "selected_scalar": selected,
        "validation_hir": chosen["validation_hir"],
        "validation_hir_in_band": chosen["in_band"],
        "tie_break": "in-band, absolute distance to 0.20, lower HIR, candidate order",
    }


def action_proportions(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    counts = {action: 0 for action in AuthorityAction}
    for record in records:
        for raw in record["actions"]:
            counts[AuthorityAction(int(raw))] += 1
    total = max(1, sum(counts.values()))
    return {
        "p_ai": counts[AuthorityAction.AI] / total,
        "p_human": counts[AuthorityAction.HUMAN] / total,
        "p_stop": counts[AuthorityAction.STOP] / total,
    }


def fit_and_calibrate_comparators(
    encoder: FeatureEncoder,
    split: Mapping[str, Sequence[RepairEpisode]],
    reward: RewardConfig,
    seed: int,
    checkpoint_root: Path,
    target: float,
    lower: float,
    upper: float,
) -> tuple[list[Any], dict[str, Any]]:
    validation_states = [step.state for episode in split["validation"] for step in episode.steps]
    calibrations: dict[str, Any] = {}

    conf = ThresholdPolicy("uncertainty", 0.0, True, "share_conf")
    conf_values = [float(state.uncertainty) for state in validation_states]
    calibrations[conf.name] = calibrate_scalar_validation_only(
        conf, split["validation"], breakpoint_candidates(conf_values),
        lambda value: setattr(conf, "threshold", value),
        target=target, lower=lower, upper=upper,
    )

    static = SupervisedBenefitPolicy(encoder, target, seed).fit(split["train"])
    static.name = "share_static"
    static_values = [static._score(state) for state in validation_states]
    calibrations[static.name] = calibrate_scalar_validation_only(
        static, split["validation"], breakpoint_candidates(static_values),
        lambda value: setattr(static, "threshold", value),
        target=target, lower=lower, upper=upper,
    )

    rewards = immediate_reward_tensor(split["train"], reward)
    bandit = ContextualBanditPolicy(encoder, target).fit(split["train"], rewards)
    bandit.name = "share_bandit"
    bandit_values = [bandit._advantage(state) for state in validation_states]
    calibrations[bandit.name] = calibrate_scalar_validation_only(
        bandit, split["validation"], breakpoint_candidates(bandit_values),
        lambda value: setattr(bandit, "human_penalty", value),
        target=target, lower=lower, upper=upper,
    )

    ddqn = DQNPolicy.load(
        str(checkpoint_root / f"share_ddqn_seed_{seed}.pt"), encoder, reward,
        require_reward_config=True,
    )
    ddqn.name = "share_ddqn"
    ddqn.human_calibration_penalty = 0.0
    ddqn.ai_calibration_penalty = 0.0
    raw = ddqn._raw_values_batch(validation_states)
    ddqn_values = raw[:, AuthorityAction.HUMAN] - np.maximum(
        raw[:, AuthorityAction.AI], raw[:, AuthorityAction.STOP]
    )
    calibrations[ddqn.name] = calibrate_scalar_validation_only(
        ddqn, split["validation"], breakpoint_candidates(ddqn_values.tolist()),
        lambda value: setattr(ddqn, "human_calibration_penalty", value),
        target=target, lower=lower, upper=upper,
    )
    return [conf, static, bandit, ddqn], calibrations


def run_seed(
    seed: int,
    encoder: FeatureEncoder,
    split: Mapping[str, Sequence[RepairEpisode]],
    reward: RewardConfig,
    checkpoint_root: Path,
    target: float,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    comparators, calibrations = fit_and_calibrate_comparators(
        encoder, split, reward, seed, checkpoint_root, target, lower, upper
    )
    share_h = DQNPolicy.load(
        str(checkpoint_root / f"share_h_seed_{seed}.pt"), encoder, reward,
        require_reward_config=True,
    )
    share_h.name = "share_h"
    calibrations[share_h.name] = {
        "selection_split": "frozen_rescue_checkpoint",
        "selected_scalar": share_h.human_calibration_penalty,
        "validation_hir": share_h.calibration_validation_hir,
        "not_retrained_or_recalibrated": True,
    }
    rows = []
    for policy in [*comparators, share_h]:
        result = evaluate_policy(policy, split["test"], reward, target, None)
        result.update(action_proportions(result["records"]))
        result.update(
            {
                "seed": seed,
                "target_budget": target,
                "evaluation_hir_in_band": lower - 1e-12 <= result["hir"] <= upper + 1e-12,
                "synthetic_development_evaluation": True,
            }
        )
        rows.append(result)
    return {"seed": seed, "calibrations": calibrations, "results": rows}


def aggregate_rows(seed_payloads: Sequence[Mapping[str, Any]], lower: float, upper: float) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for payload in seed_payloads:
        for row in payload["results"]:
            buckets[row["policy"]].append(row)
    output = []
    for policy in POLICIES:
        rows = buckets[policy]
        item: dict[str, Any] = {"policy": policy, "seeds": len(rows)}
        for metric in (*METRICS, "p_ai", "p_human", "p_stop"):
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        item["mean_hir_in_band"] = lower - 1e-12 <= item["hir_mean"] <= upper + 1e-12
        item["all_seed_hir_in_band"] = all(row["evaluation_hir_in_band"] for row in rows)
        output.append(item)
    return output


def exact_sign_flip_p(values: Sequence[float]) -> float:
    observed = abs(mean(values))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(abs(mean([sign * value for sign, value in zip(signs, values)])))
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def paired_differences(seed_payloads: Sequence[Mapping[str, Any]], draws: int = 20_000) -> list[dict[str, Any]]:
    by_seed = {
        int(payload["seed"]): {row["policy"]: row for row in payload["results"]}
        for payload in seed_payloads
    }
    rng = np.random.default_rng(20260825)
    rows = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            values = [
                float(policies[comparator][metric]) - float(policies["share_h"][metric])
                for _, policies in sorted(by_seed.items())
            ]
            sampled = rng.integers(0, len(values), size=(draws, len(values)))
            boot = np.asarray(values)[sampled].mean(axis=1)
            rows.append(
                {
                    "comparison": f"{comparator}-minus-share_h",
                    "metric": metric,
                    "direction": "comparator minus SHARE-H",
                    "per_seed": {
                        str(seed): float(policies[comparator][metric]) - float(policies["share_h"][metric])
                        for seed, policies in sorted(by_seed.items())
                    },
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "ci95_seed_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "p_exact_sign_flip": exact_sign_flip_p(values),
                    "seeds": len(values),
                }
            )
    for metric in PRIMARY_METRICS:
        family = sorted((row for row in rows if row["metric"] == metric), key=lambda row: row["p_exact_sign_flip"])
        running = 0.0
        for index, row in enumerate(family):
            running = max(running, min(1.0, (len(family) - index) * row["p_exact_sign_flip"]))
            row["p_holm"] = running
    return rows


def pareto_assessment(aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy = {row["policy"]: row for row in aggregate}
    reference = by_policy["share_h"]
    comparisons = []
    for name in COMPARATORS:
        row = by_policy[name]
        eligible = bool(row["mean_hir_in_band"])
        weak = (
            row["repair_sr_mean"] >= reference["repair_sr_mean"] - 1e-12
            and row["uar_mean"] <= reference["uar_mean"] + 1e-12
            and row["unsafe_autonomy_cost_rate_mean"] <= reference["unsafe_autonomy_cost_rate_mean"] + 1e-12
            and row["stopped_rate_mean"] <= reference["stopped_rate_mean"] + 1e-12
            and row["normalized_return_mean"] >= reference["normalized_return_mean"] - 1e-12
            and row["collaboration_regret_mean"] <= reference["collaboration_regret_mean"] + 1e-12
        )
        strict = any(
            (
                row[metric] > reference[metric] + 1e-12
                if direction == "max"
                else row[metric] < reference[metric] - 1e-12
            )
            for metric, direction in (
                ("repair_sr_mean", "max"), ("uar_mean", "min"),
                ("unsafe_autonomy_cost_rate_mean", "min"), ("stopped_rate_mean", "min"),
                ("normalized_return_mean", "max"), ("collaboration_regret_mean", "min"),
            )
        )
        comparisons.append({"comparator": name, "eligible_mean_hir": eligible, "dominates_share_h": eligible and weak and strict})
    eligible = [row for row in comparisons if row["eligible_mean_hir"]]
    if not eligible:
        conclusion = "inconclusive_no_budget_matched_comparator"
        on_frontier = None
    else:
        on_frontier = not any(row["dominates_share_h"] for row in eligible)
        conclusion = "share_h_on_frontier" if on_frontier else "share_h_dominated"
    return {
        "budget_eligibility_rule": "five-seed mean evaluation HIR in [0.18,0.22]; per-seed compliance also reported",
        "dominance_dimensions": {
            "maximize": ["repair_sr", "normalized_return"],
            "minimize": ["uar", "unsafe_autonomy_cost_rate", "stopped_rate", "collaboration_regret"],
        },
        "comparisons": comparisons,
        "eligible_comparator_count": len(eligible),
        "conclusion": conclusion,
        "share_h_on_frontier_among_mean_hir_eligible_methods": on_frontier,
    }


def write_csvs(out: Path, seed_payloads: Sequence[Mapping[str, Any]], aggregate: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> None:
    per_seed = []
    for payload in seed_payloads:
        for row in payload["results"]:
            per_seed.append({key: value for key, value in row.items() if key != "records"})
    for name, rows in (("per_seed.csv", per_seed), ("aggregate.csv", list(aggregate))):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    flat_paired = []
    for row in paired:
        base = {key: value for key, value in row.items() if key != "per_seed"}
        for seed, value in row["per_seed"].items():
            flat_paired.append({**base, "seed": seed, "difference": value})
    fields = list(dict.fromkeys(key for row in flat_paired for key in row))
    with (out / "paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_paired)


def protocol(args: argparse.Namespace, checkpoint_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "synthetic_development_budget_matched_diagnostic",
        "evaluation_split_already_inspected": True,
        "locked_test_touched": False,
        "episodes_sha256": sha256_file(args.episodes),
        "source_rescue_protocol_sha256": checkpoint_manifest["protocol_sha256"],
        "policies": list(POLICIES),
        "seeds": [int(value) for value in args.seeds.split(",")],
        "target_hir": args.target_hir,
        "allowed_evaluation_hir": [args.hir_lower, args.hir_upper],
        "calibration": "all comparator scalar choices use validation episode-level HIR only; evaluation is run once and never feeds selection",
        "share_h": "load frozen rescue checkpoint without retraining or recalibration",
        "ddqn": "load frozen rescue DDQN weights; recalibrate human penalty only on validation",
        "selection_prohibitions": ["evaluation Repair.SR", "evaluation HIR", "evaluation UAR", "evaluation Stop", "evaluation Return", "evaluation collaboration regret"],
        "reward_config": checkpoint_manifest["reward_config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict validation-calibrated 20% HIR controller comparison.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--run-seeds", default=None)
    parser.add_argument("--target-hir", type=float, default=0.20)
    parser.add_argument("--hir-lower", type=float, default=0.18)
    parser.add_argument("--hir-upper", type=float, default=0.22)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    run_seeds = tuple(int(value) for value in args.run_seeds.split(",")) if args.run_seeds else seeds
    if not set(run_seeds).issubset(seeds):
        raise SystemExit("run-seeds must be a subset of frozen seeds")
    checkpoint_manifest = json.loads((args.checkpoints / "manifest.json").read_text(encoding="utf-8"))
    frozen = protocol(args, checkpoint_manifest)
    if frozen["episodes_sha256"] != checkpoint_manifest["episode_sha256"]:
        raise SystemExit("episode data differs from frozen rescue run")
    protocol_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    frozen["protocol_sha256"] = protocol_hash
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["protocol_sha256"] != protocol_hash:
            raise SystemExit("refusing to resume under a changed protocol")
    else:
        frozen.update({"python": platform.python_version(), "torch": torch.__version__})
        atomic_json(manifest_path, frozen)

    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ("train", "validation", "test")}
    encoder = FeatureEncoder.load(args.checkpoints / "feature_encoder.json")
    reward = RewardConfig(**checkpoint_manifest["reward_config"])
    for seed in run_seeds:
        path = args.out / f"seed_{seed}.json"
        if path.exists():
            print(json.dumps({"seed": seed, "status": "already complete"}))
            continue
        payload = run_seed(
            seed, encoder, split, reward, args.checkpoints,
            args.target_hir, args.hir_lower, args.hir_upper,
        )
        atomic_json(path, payload)
        print(json.dumps({"seed": seed, "status": "complete", "out": str(path)}))

    seed_payloads = []
    for seed in seeds:
        path = args.out / f"seed_{seed}.json"
        if path.exists():
            seed_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    complete = len(seed_payloads) == len(seeds)
    if complete:
        aggregate = aggregate_rows(seed_payloads, args.hir_lower, args.hir_upper)
        paired = paired_differences(seed_payloads)
        pareto = pareto_assessment(aggregate)
        atomic_json(args.out / "aggregate.json", aggregate)
        atomic_json(args.out / "paired_differences.json", paired)
        atomic_json(args.out / "pareto.json", pareto)
        write_csvs(args.out, seed_payloads, aggregate, paired)
    print(json.dumps({"out": str(args.out), "completed_seeds": len(seed_payloads), "complete": complete}))


if __name__ == "__main__":
    main()
