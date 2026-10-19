from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_guard_experiments import load_historical_identifier_index, validate_locked_holdout
from scripts.run_share_online_hard_cap import BINARY_POLICIES, OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.environment import base_reward
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


SEEDS = (17, 29, 43, 71, 101)
BUDGETS = (0.10, 0.20, 0.30)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share")
PRIMARY_METRICS = ("repair_sr", "hir", "uar", "stopped_rate")
SECONDARY_METRICS = ("normalized_return", "collaboration_regret")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260826


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


def verify_lock(lock: Mapping[str, Any]) -> None:
    for item in lock["files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen artifact mismatch: {path}")


def load_budget_policies(
    lock: Mapping[str, Any], budget: float, seed: int, encoder: FeatureEncoder, reward: RewardConfig
) -> list[Any]:
    files = lock["operating_points"][f"{budget:.1f}"][str(seed)]
    pack = joblib.load(files["comparators"]["path"])
    ddqn = DQNPolicy.load(files["share_ddqn"]["path"], encoder, reward, require_reward_config=True)
    ddqn.name = "share_ddqn"
    share = DQNPolicy.load(files["share"]["path"], encoder, reward, require_reward_config=True)
    share.name = "share"
    if budget == 0.20:
        calibration = json.loads(Path(files["calibration"]["path"]).read_text(encoding="utf-8"))["calibrations"]
        ddqn.human_calibration_penalty = float(calibration["share_ddqn"]["selected_scalar"])
        ddqn.ai_calibration_penalty = 0.0
    return [pack[name] for name in POLICIES[:3]] + [ddqn, share]


def run_policy(policy: Any, episodes: Sequence[RepairEpisode], reward: RewardConfig, budget: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, policy.name in BINARY_POLICIES)
    result = evaluate_policy(wrapper, episodes, reward, budget, None)
    human_actions = sum(int(action == int(AuthorityAction.HUMAN)) for record in result["records"] for action in record["actions"])
    if wrapper.human_interventions > cap:
        raise RuntimeError(f"hard-cap violation: {budget}/{seed}/{policy.name}")
    result.update({
        "nominal_budget": budget, "seed": seed, "hard_human_cap": cap,
        "human_interventions": wrapper.human_interventions, "human_actions": human_actions,
        "cap_respected": wrapper.human_interventions <= cap, "locked_holdout": True,
    })
    audit = {
        "cap": cap, "human_interventions": wrapper.human_interventions, "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "forced_autonomous_fallbacks": wrapper.forced_autonomous_fallbacks,
        "value_policy_replans": wrapper.value_policy_replans, "events": wrapper.events,
    }
    return result, audit


def record_metric(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    if metric == "repair_sr":
        return mean(float(row["success"]) for row in records)
    if metric == "hir":
        return mean(float(row["human_used"]) for row in records)
    if metric == "stopped_rate":
        return mean(float(int(row["actions"][-1]) == int(AuthorityAction.STOP)) for row in records)
    if metric == "normalized_return":
        return mean(sum(float(value) for value in row["rewards"]) for row in records)
    if metric == "uar":
        eligible = [row for row in records if not row["human_used"] and int(row["actions"][-1]) != int(AuthorityAction.STOP)]
        return mean(float(row["unsafe"] or not row["success"]) for row in eligible) if eligible else float("nan")
    if metric == "collaboration_regret":
        values = [float(oracle) - float(selected) for row in records for selected, oracle in zip(row["selected_returns"], row["oracle_returns"])]
        return mean(values)
    raise ValueError(metric)


def exact_sign_flip(values: Sequence[float]) -> float:
    observed = abs(mean(values))
    return sum(
        abs(mean([sign * value for sign, value in zip(signs, values)])) >= observed - 1e-15
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ) / (2 ** len(values))


def holm_adjust(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        output[index] = min(1.0, running)
    return output


def primary_comparisons(payloads: Sequence[Mapping[str, Any]], episodes: Sequence[RepairEpisode]) -> list[dict[str, Any]]:
    by_seed = {
        int(payload["seed"]): {row["policy"]: row for row in payload["results"] if row["nominal_budget"] == 0.20}
        for payload in payloads
    }
    groups = sorted({episode.group_id for episode in episodes})
    group_ids = {group: [episode.episode_id for episode in episodes if episode.group_id == group] for group in groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for comparator in POLICIES[:-1]:
        for metric in PRIMARY_METRICS:
            per_seed = [float(by_seed[seed][comparator][metric]) - float(by_seed[seed]["share"][metric]) for seed in SEEDS]
            boot = []
            for _ in range(BOOTSTRAP_SAMPLES):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                seed_differences = []
                for seed in SEEDS:
                    values = {}
                    for policy in (comparator, "share"):
                        by_id = {row["episode_id"]: row for row in by_seed[seed][policy]["records"]}
                        records = [by_id[episode_id] for group in sampled for episode_id in group_ids[group]]
                        values[policy] = record_metric(records, metric)
                    if not any(math.isnan(value) for value in values.values()):
                        seed_differences.append(values[comparator] - values["share"])
                boot.append(mean(seed_differences))
            output.append({
                "comparison": f"{comparator}-minus-share", "metric": metric,
                "direction": "higher_is_better" if metric == "repair_sr" else "lower_is_better",
                "per_seed_differences": per_seed, "mean_difference": mean(per_seed),
                "std_difference": stdev(per_seed),
                "workflow_group_cluster_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "exact_seed_sign_flip_p_two_sided": exact_sign_flip(per_seed),
            })
    adjusted = holm_adjust([row["exact_seed_sign_flip_p_two_sided"] for row in output])
    for row, value in zip(output, adjusted):
        row["holm_adjusted_p_across_16_primary_tests"] = value
    return output


def aggregate(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload["results"]:
            buckets[(float(row["nominal_budget"]), row["policy"])].append(row)
    output = []
    for budget in BUDGETS:
        for policy in POLICIES:
            rows = buckets[(budget, policy)]
            item: dict[str, Any] = {"nominal_budget": budget, "policy": policy, "seeds": len(rows)}
            for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
                values = [float(row[metric]) for row in rows]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values)
            item["all_runs_respected_cap"] = all(bool(row["cap_respected"]) for row in rows)
            output.append(item)
    return output


def ai_return(episode: RepairEpisode, step: int, reward: RewardConfig, gamma: float = 0.95) -> float:
    outcome = episode.steps[step].outcome(AuthorityAction.AI)
    value = base_reward(outcome, AuthorityAction.AI, reward)
    if not outcome.terminal and outcome.next_step is not None:
        value += gamma * ai_return(episode, outcome.next_step, reward, gamma)
    return value


def voh_analysis(lock: Mapping[str, Any], episodes: Sequence[RepairEpisode], encoder: FeatureEncoder, reward: RewardConfig) -> dict[str, Any]:
    rows = []
    seed_rhos = []
    seed_slopes = []
    for seed in SEEDS:
        checkpoint = lock["operating_points"]["0.2"][str(seed)]["share"]["path"]
        policy = DQNPolicy.load(checkpoint, encoder, reward, require_reward_config=True)
        states = [step.state for episode in episodes for step in episode.steps]
        q = policy._raw_values_batch(states)
        seed_rows = []
        offset = 0
        for episode in episodes:
            for step, paired in enumerate(episode.steps):
                human = paired.outcome(AuthorityAction.HUMAN)
                advantage = base_reward(human, AuthorityAction.HUMAN, reward) - ai_return(episode, step, reward)
                seed_rows.append({
                    "seed": seed, "episode_id": episode.episode_id, "group_id": episode.group_id, "step": step,
                    "predicted_voh": float(q[offset, AuthorityAction.HUMAN] - q[offset, AuthorityAction.AI]),
                    "realized_advantage": advantage,
                })
                offset += 1
        ranked = sorted(seed_rows, key=lambda row: (row["predicted_voh"], row["episode_id"], row["step"]))
        for rank, row in enumerate(ranked):
            row["voh_bin"] = min(5, math.floor(5 * rank / len(ranked)) + 1)
        means = [mean(row["realized_advantage"] for row in ranked if row["voh_bin"] == bin_index) for bin_index in range(1, 6)]
        seed_slopes.append(float(np.polyfit(np.arange(1, 6), means, 1)[0]))
        seed_rhos.append(float(spearmanr([row["predicted_voh"] for row in seed_rows], [row["realized_advantage"] for row in seed_rows]).statistic))
        rows.extend(seed_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = sorted({episode.group_id for episode in episodes})
    by_seed_group: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed_group[(row["seed"], row["group_id"])].append(row)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rhos = []
        for seed in SEEDS:
            sample = [row for group in sampled for row in by_seed_group[(seed, group)]]
            rhos.append(float(spearmanr([row["predicted_voh"] for row in sample], [row["realized_advantage"] for row in sample]).statistic))
        boot.append(mean(rhos))
    return {
        "estimand": "raw Q(H)-Q(AI) versus terminal Human reward minus recursively continued AI return (gamma=.95)",
        "binning": "within-seed evaluation prediction rank quintiles; sizes differ by at most one; outcomes not used",
        "per_seed_spearman": dict(zip(map(str, SEEDS), seed_rhos)),
        "mean_spearman": mean(seed_rhos), "std_spearman": stdev(seed_rhos),
        "group_cluster_bootstrap_mean_spearman_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "per_seed_bin_slope": dict(zip(map(str, SEEDS), seed_slopes)),
        "one_sided_exact_sign_flip_p_for_positive_slope": sum(
            mean([sign * value for sign, value in zip(signs, seed_slopes)]) >= mean(seed_slopes) - 1e-15
            for signs in itertools.product((-1.0, 1.0), repeat=len(seed_slopes))
        ) / (2 ** len(seed_slopes)),
        "state_rows": rows,
    }


def exploratory_strata(payloads: Sequence[Mapping[str, Any]], episodes: Sequence[RepairEpisode]) -> list[dict[str, Any]]:
    by_id = {episode.episode_id: episode for episode in episodes}
    rows = []
    for payload in payloads:
        for result in payload["results"]:
            if result["nominal_budget"] != 0.20:
                continue
            for kind, levels in (
                ("complexity", ("C_low", "C_mid", "C_high")),
                ("initial_horizon", ("H1", "H2", "H3")),
            ):
                for level in levels:
                    selected = []
                    for record in result["records"]:
                        episode = by_id[record["episode_id"]]
                        complexity = episode.steps[0].state.complexity
                        observed = ("C_low" if complexity < .5 else "C_mid" if complexity < .8 else "C_high") if kind == "complexity" else f"H{len(episode.steps)}"
                        if observed == level:
                            selected.append(record)
                    if selected:
                        rows.append({
                            "seed": payload["seed"], "policy": result["policy"], "stratum_type": kind, "stratum": level,
                            "episodes": len(selected), **{metric: record_metric(selected, metric) for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS)},
                        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Open and evaluate the preregistered SHARE locked holdout exactly once.")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--historical-identifiers", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    lock = json.loads(args.artifact_lock.read_text(encoding="utf-8"))
    if sha256_file(args.preregistration) != seal["preregistration_sha256"] or sha256_file(args.artifact_lock) != seal["artifact_lock_sha256"]:
        raise SystemExit("preregistration seal mismatch")
    expected_manifest = prereg.get("locked_holdout_manifest_sha256")
    if not expected_manifest:
        raise SystemExit("preregistration is not yet bound to a locked-holdout manifest; refusing to open outcomes")
    if sha256_file(args.holdout_manifest) != expected_manifest:
        raise SystemExit("supplied holdout manifest differs from preregistered digest")
    verify_lock(lock)
    args.out.mkdir(parents=True, exist_ok=True)
    attempt = args.out / "UNBLINDING_ATTEMPT.json"
    if attempt.exists():
        raise SystemExit("locked evaluation has already been attempted; refusing a second opening")
    atomic_json(attempt, {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": seal["preregistration_sha256"],
        "holdout_manifest_sha256": expected_manifest,
        "status": "holdout_opened_once",
    })
    development = load_episodes(args.development)
    historical = load_historical_identifier_index(args.historical_identifiers, args.development, development)
    holdout = load_episodes(args.holdout)
    provenance = validate_locked_holdout(args.holdout, args.holdout_manifest, holdout, development, historical)
    episodes = sorted(holdout, key=lambda episode: episode.episode_id)
    encoder = FeatureEncoder.load(Path(lock["encoder"]["path"]))
    reward = RewardConfig(**lock["reward_config"])
    payloads = []
    for seed in SEEDS:
        results, audits = [], {}
        for budget in BUDGETS:
            for policy in load_budget_policies(lock, budget, seed, encoder, reward):
                result, audit = run_policy(policy, episodes, reward, budget, seed)
                results.append(result)
                audits[f"{budget:.1f}/{policy.name}"] = audit
        payloads.append({"seed": seed, "results": results, "budget_audit": audits})
    primary = primary_comparisons(payloads, episodes)
    voh = voh_analysis(lock, episodes, encoder, reward)
    voh_rows = voh.pop("state_rows")
    atomic_json(args.out / "raw_results.json", {"payloads": payloads})
    atomic_json(args.out / "aggregate.json", aggregate(payloads))
    atomic_json(args.out / "primary_comparisons.json", primary)
    atomic_json(args.out / "voh_analysis.json", {**voh, "state_rows": voh_rows})
    atomic_json(args.out / "exploratory_strata.json", exploratory_strata(payloads, episodes))
    atomic_json(args.out / "evaluation_manifest.json", {
        "completed_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
        "preregistration_sha256": seal["preregistration_sha256"], "artifact_lock_sha256": seal["artifact_lock_sha256"],
        "episode_order": [episode.episode_id for episode in episodes],
        "episode_order_sha256": hashlib.sha256("\n".join(episode.episode_id for episode in episodes).encode()).hexdigest(),
        "runs": len(SEEDS) * len(BUDGETS) * len(POLICIES), "all_caps_respected": True,
        "retraining": False, "recalibration": False, "posthoc_selection": False,
    })
    attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
    attempt_payload.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(attempt, attempt_payload)
    print(json.dumps({"out": str(args.out), "episodes": len(episodes), "runs": 75, "complete": True}))


if __name__ == "__main__":
    main()
