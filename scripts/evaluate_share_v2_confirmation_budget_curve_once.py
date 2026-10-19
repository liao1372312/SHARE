from __future__ import annotations

"""Evaluate frozen V2 and frozen comparators at 10%/30% exactly once."""

import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_locked_holdout_once import load_budget_policies
from scripts.run_share_online_hard_cap import BINARY_POLICIES, OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig

DATASET = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
DATASET_SHA256 = "70c58b9aa2335177282382adae618797f4c578216e8eabfcd9f4267c15f0d7c4"
LOCK = ROOT / "results/share_locked_evaluation_prereg_20260826/artifact_lock.json"
OUT = ROOT / "results/share_v2_confirmation_budget_curve_20260827"
SEEDS = (17, 29, 43, 71, 101)
BUDGETS = (0.10, 0.30)
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")
V2_ROOTS = {
    0.10: ROOT / "results/share_v2_budget_0p1_development_20260827",
    0.30: ROOT / "results/share_v2_budget_0p3_development_20260827",
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


def v2_checkpoint(budget: float, seed: int) -> Path:
    return V2_ROOTS[budget] / f"share_v2_budget_{budget:.1f}_seed_{seed}.pt"


def run(policy: Any, episodes: Sequence[RepairEpisode], reward: RewardConfig, budget: float, seed: int) -> dict[str, Any]:
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, policy.name in BINARY_POLICIES)
    result = evaluate_policy(wrapper, episodes, reward, budget, None)
    if wrapper.human_interventions > cap:
        raise RuntimeError(f"hard-cap violation: {budget}/{seed}/{policy.name}")
    result.update(
        nominal_budget=budget,
        seed=seed,
        hard_human_cap=cap,
        human_interventions=wrapper.human_interventions,
        cap_respected=True,
    )
    return result


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in BUDGETS:
        for policy in ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_v2"):
            items = [row for row in rows if row["nominal_budget"] == budget and row["policy"] == policy]
            value: dict[str, Any] = {"budget": budget, "policy": policy, "seeds": len(items)}
            for metric in METRICS:
                samples = [float(item[metric]) for item in items]
                value[f"{metric}_mean"] = mean(samples)
                value[f"{metric}_std"] = stdev(samples)
            value["all_caps_respected"] = all(bool(item["cap_respected"]) for item in items)
            output.append(value)
    return output


def main() -> None:
    if sha256_file(DATASET) != DATASET_SHA256:
        raise SystemExit("confirmation dataset hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "FINAL_RESULTS.json").exists():
        raise SystemExit("budget-curve confirmation already executed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2_encoder_paths = {budget: V2_ROOTS[budget] / "feature_encoder.json" for budget in BUDGETS}
    manifest = {
        "protocol": "share-v2-confirmation-budget-curve-once-v1",
        "dataset": {"path": str(DATASET), "sha256": DATASET_SHA256, "episodes": 400},
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "v2_training_reward": {"rollback_penalty": 0.60},
        "v2_encoders": {str(budget): sha256_file(path) for budget, path in v2_encoder_paths.items()},
        "v2_checkpoints": {f"{budget:.1f}/{seed}": sha256_file(v2_checkpoint(budget, seed)) for budget in BUDGETS for seed in SEEDS},
        "comparator_artifact_lock": {"path": str(LOCK), "sha256": sha256_file(LOCK)},
        "hard_cap_evaluator_sha256": sha256_file(ROOT / "scripts/run_share_online_hard_cap.py"),
        "status": "sealed_before_execution",
        "retraining": False,
        "recalibration": False,
    }
    atomic_json(OUT / "pre_evaluation_manifest.json", manifest)
    episodes = sorted(load_episodes(DATASET), key=lambda episode: episode.episode_id)
    comparator_encoder = FeatureEncoder.load(Path(lock["encoder"]["path"]))
    comparator_reward = RewardConfig(**lock["reward_config"])
    rows = []
    for budget in BUDGETS:
        v2_encoder = FeatureEncoder.load(v2_encoder_paths[budget])
        v2_reward = RewardConfig(rollback_penalty=0.60)
        for seed in SEEDS:
            comparators = load_budget_policies(lock, budget, seed, comparator_encoder, comparator_reward)[:4]
            for policy in comparators:
                rows.append(run(policy, episodes, comparator_reward, budget, seed))
            v2 = DQNPolicy.load(str(v2_checkpoint(budget, seed)), v2_encoder, v2_reward, require_reward_config=True)
            v2.name = "share_v2"
            rows.append(run(v2, episodes, RewardConfig(), budget, seed))
    summaries = [{key: value for key, value in row.items() if key != "records"} for row in rows]
    result = {
        "evidence_class": "new_workflow_group_disjoint_replayable_synthetic_confirmation",
        "manifest_sha256": sha256_file(OUT / "pre_evaluation_manifest.json"),
        "aggregate": aggregate(summaries),
        "per_seed": summaries,
        "all_caps_respected": all(bool(row["cap_respected"]) for row in summaries),
    }
    atomic_json(OUT / "FINAL_RESULTS.json", result)
    atomic_json(OUT / "raw_records.json", [{"budget": row["nominal_budget"], "seed": row["seed"], "policy": row["policy"], "records": row["records"]} for row in rows])
    print(json.dumps({"out": str(OUT), "runs": len(rows), "all_caps_respected": result["all_caps_respected"]}, indent=2))


if __name__ == "__main__":
    main()
