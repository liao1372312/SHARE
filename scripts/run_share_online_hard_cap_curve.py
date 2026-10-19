from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import BINARY_POLICIES, OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig


BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.5)
SEEDS = (17, 29, 43, 71, 101)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
METRICS = (
    "repair_sr", "hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate",
    "normalized_return", "collaboration_regret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slug(budget: float) -> str:
    return f"{budget:g}".replace(".", "p")


def load_policies(budget_dir: Path, seed: int, encoder: FeatureEncoder, reward: RewardConfig):
    pack = joblib.load(budget_dir / f"comparators_seed_{seed}.joblib")
    ddqn = DQNPolicy.load(str(budget_dir / f"share_ddqn_seed_{seed}.pt"), encoder, reward, require_reward_config=True)
    ddqn.name = "share_ddqn"
    share_h = DQNPolicy.load(str(budget_dir / f"share_h_seed_{seed}.pt"), encoder, reward, require_reward_config=True)
    share_h.name = "share_h"
    return [pack[name] for name in POLICIES[:3]] + [ddqn, share_h]


def run_unit(policy, episodes: Sequence[RepairEpisode], cap: int, budget: float, seed: int) -> tuple[dict, dict]:
    wrapper = OnlineHardHumanCapPolicy(policy, cap, policy.name in BINARY_POLICIES)
    result = evaluate_policy(wrapper, episodes, human_budget=budget)
    human_actions = sum(
        int(action == int(AuthorityAction.HUMAN))
        for record in result["records"] for action in record["actions"]
    )
    human_episodes = sum(int(record["human_used"]) for record in result["records"])
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError(f"cap failure: budget={budget}, seed={seed}, policy={policy.name}")
    exhausted_at = next(
        (index for index, event in enumerate(wrapper.events) if event["budget_after"] >= cap),
        None,
    ) if cap > 0 else 0
    result.update(
        {
            "nominal_budget": budget,
            "seed": seed,
            "hard_human_cap": cap,
            "human_interventions": human_episodes,
            "human_actions": human_actions,
            "cap_respected": human_episodes <= cap,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "forced_autonomous_fallbacks": wrapper.forced_autonomous_fallbacks,
            "value_policy_replans": wrapper.value_policy_replans,
            "first_exhaustion_decision_index": exhausted_at,
            "synthetic_development_evaluation": True,
        }
    )
    audit = {
        "cap": cap,
        "human_interventions": human_episodes,
        "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "forced_autonomous_fallbacks": wrapper.forced_autonomous_fallbacks,
        "value_policy_replans": wrapper.value_policy_replans,
        "first_exhaustion_decision_index": exhausted_at,
        "events": wrapper.events,
    }
    return result, audit


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
            for metric in METRICS:
                values = [float(row[metric]) for row in rows]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            for metric in ("human_interventions", "attempted_human_after_exhaustion", "forced_autonomous_fallbacks", "value_policy_replans"):
                values = [int(row[metric]) for row in rows]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
            item["hard_human_cap"] = rows[0]["hard_human_cap"]
            item["all_runs_respected_cap"] = all(row["cap_respected"] for row in rows)
            output.append(item)
    return output


def nondominated(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        ("repair_sr_mean", "max"), ("normalized_return_mean", "max"),
        ("uar_mean", "min"), ("unsafe_autonomy_cost_rate_mean", "min"),
        ("stopped_rate_mean", "min"), ("collaboration_regret_mean", "min"),
    )
    output = []
    for budget in BUDGETS:
        group = [row for row in rows if float(row["nominal_budget"]) == budget]
        front = []
        for candidate in group:
            dominated = False
            for other in group:
                if other["policy"] == candidate["policy"]:
                    continue
                weak = all(
                    other[key] >= candidate[key] - 1e-12 if direction == "max" else other[key] <= candidate[key] + 1e-12
                    for key, direction in dimensions
                )
                strict = any(
                    other[key] > candidate[key] + 1e-12 if direction == "max" else other[key] < candidate[key] - 1e-12
                    for key, direction in dimensions
                )
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate["policy"])
        output.append({"nominal_budget": budget, "hard_cap": math.floor(120 * budget), "nondominated": front})
    return output


def write_csv(out: Path, payloads: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    per_seed = [
        {key: value for key, value in row.items() if key != "records"}
        for payload in payloads for row in payload["results"]
    ]
    for name, values in (("per_seed.csv", per_seed), ("aggregate.csv", list(rows))):
        fields = list(dict.fromkeys(key for row in values for key in row))
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered online hard-Human-budget curve.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--base-checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    parser.add_argument("--run-budgets", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    budgets = tuple(float(value) for value in args.budgets.split(","))
    run_budgets = tuple(float(value) for value in args.run_budgets.split(",")) if args.run_budgets else budgets
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if budgets != BUDGETS or seeds != SEEDS or not set(run_budgets).issubset(budgets):
        raise SystemExit("budgets and seeds must match the frozen curve protocol")
    base_manifest = json.loads((args.base_checkpoints / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != base_manifest["episode_sha256"]:
        raise SystemExit("dataset mismatch")
    episodes = load_episodes(args.episodes)
    evaluation = sorted((episode for episode in episodes if episode.split == "test"), key=lambda episode: episode.episode_id)
    order = [episode.episode_id for episode in evaluation]
    encoder = FeatureEncoder.load(args.base_checkpoints / "feature_encoder.json")
    reward = RewardConfig(**base_manifest["reward_config"])
    artifact_manifests = {}
    for budget in budgets:
        budget_dir = args.artifacts_root / f"budget_{slug(budget)}"
        manifest = json.loads((budget_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["budget"] != budget or manifest["episodes_sha256"] != base_manifest["episode_sha256"]:
            raise SystemExit(f"artifact manifest mismatch at budget {budget}")
        for seed in seeds:
            for item in manifest["files"][str(seed)].values():
                if sha256_file(budget_dir / item["path"]) != item["sha256"]:
                    raise SystemExit(f"artifact checksum mismatch at budget={budget}, seed={seed}")
        artifact_manifests[str(budget)] = sha256_file(budget_dir / "manifest.json")
    protocol = {
        "protocol": "share-online-hard-cap-curve-v1",
        "evidence": "synthetic development curve; evaluation split already inspected",
        "budgets": list(budgets),
        "seeds": list(seeds),
        "policies": list(POLICIES),
        "caps": {str(budget): math.floor(budget * len(evaluation)) for budget in budgets},
        "budget_unit": "episodes with at least one executed Human intervention",
        "evaluation_order": order,
        "evaluation_order_sha256": sha256_json(order),
        "hard_cap_evaluator_source_sha256": sha256_file(ROOT / "scripts/run_share_online_hard_cap.py"),
        "artifact_manifest_sha256": artifact_manifests,
        "locked_holdout_accessed": False,
        "evaluation_outcome_recalibration": False,
        "share_h_formulation_changed": False,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["protocol_sha256"] != protocol["protocol_sha256"]:
            raise SystemExit("refusing changed-protocol resume")
    else:
        manifest_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for budget in run_budgets:
        budget_dir = args.artifacts_root / f"budget_{slug(budget)}"
        cap = math.floor(budget * len(evaluation))
        for seed in seeds:
            path = args.out / f"budget_{slug(budget)}_seed_{seed}.json"
            if path.exists():
                continue
            results, audits = [], {}
            for policy in load_policies(budget_dir, seed, encoder, reward):
                result, audit = run_unit(policy, evaluation, cap, budget, seed)
                results.append(result)
                audits[policy.name] = audit
            path.write_text(json.dumps({"budget": budget, "seed": seed, "results": results, "budget_audit": audits}, indent=2), encoding="utf-8")
            print(json.dumps({"budget": budget, "seed": seed, "status": "complete"}), flush=True)
    payloads = []
    for budget in budgets:
        for seed in seeds:
            path = args.out / f"budget_{slug(budget)}_seed_{seed}.json"
            if path.exists():
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
    complete = len(payloads) == len(budgets) * len(seeds)
    if complete:
        rows = aggregate(payloads)
        (args.out / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (args.out / "nondominated.json").write_text(json.dumps(nondominated(rows), indent=2), encoding="utf-8")
        write_csv(args.out, payloads, rows)
    print(json.dumps({"out": str(args.out), "complete": complete, "budget_seed_units": len(payloads)}))


if __name__ == "__main__":
    main()
