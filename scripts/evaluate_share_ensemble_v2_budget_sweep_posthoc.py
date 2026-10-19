"""Retrospectively evaluate the frozen SHARE Q ensemble on the old 400-group holdout.

This script performs no fitting, calibration, or model selection.  It applies
three deployment-time hard caps to the same validation-frozen five-member
ensemble and reports leave-one-member-out (LOO) sensitivity separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RewardConfig

DATASET = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
OUT = ROOT / "results/share_ensemble_v2_budget_sweep_posthoc_20260829"
EXPECTED_DATASET_SHA256 = "70c58b9aa2335177282382adae618797f4c578216e8eabfcd9f4267c15f0d7c4"
HUMAN_PENALTY = 1.9403412342071533
BUDGETS = (0.10, 0.20, 0.30)
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(members, encoder, episodes, budget: float, label: str) -> dict:
    training_reward = RewardConfig(rollback_penalty=0.60)
    policy = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    policy.name = label
    policy.human_calibration_penalty = HUMAN_PENALTY
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), budget, None)
    result = {key: value for key, value in raw.items() if key != "records"}
    result.update(
        nominal_budget=budget,
        hard_human_cap=cap,
        human_interventions=wrapper.human_interventions,
        attempted_human_after_exhaustion=wrapper.attempted_human_after_exhaustion,
        cap_respected=wrapper.human_interventions <= cap,
    )
    return result


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("post-hoc ensemble budget sweep already exists")
    if sha256_file(DATASET) != EXPECTED_DATASET_SHA256:
        raise SystemExit("old 400-group dataset hash mismatch")
    OUT.mkdir(parents=True, exist_ok=False)
    episodes = sorted(load_episodes(DATASET), key=lambda episode: episode.episode_id)
    if len(episodes) != 400 or len({episode.group_id for episode in episodes}) != 400:
        raise SystemExit("unexpected old holdout size")

    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    members = {
        seed: DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True)
        for seed in SEEDS
    }
    manifest = {
        "protocol": "share-frozen-q-ensemble-old-v2-holdout-budget-sensitivity-v1",
        "evidence_class": "retrospective deployment-budget sensitivity; not confirmation or model selection",
        "retraining": False,
        "recalibration": False,
        "dataset": {"path": str(DATASET.relative_to(ROOT)), "sha256": sha256_file(DATASET), "episodes": 400},
        "budgets": list(BUDGETS),
        "members": list(SEEDS),
        "checkpoint_sha256": {str(seed): sha256_file(CHECKPOINTS[seed]) for seed in SEEDS},
        "encoder_sha256": sha256_file(ENCODER),
        "human_calibration_penalty": HUMAN_PENALTY,
        "ensemble_implementation_sha256": sha256_file(ROOT / "share/dqn.py"),
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    (OUT / "pre_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    full_rows = []
    loo_rows = []
    for budget in BUDGETS:
        full_rows.append(evaluate(list(members.values()), encoder, episodes, budget, "share_q_ensemble"))
        for omitted in SEEDS:
            row = evaluate(
                [member for seed, member in members.items() if seed != omitted],
                encoder,
                episodes,
                budget,
                f"share_q_ensemble_leave_out_{omitted}",
            )
            row["omitted_member_seed"] = omitted
            loo_rows.append(row)

    aggregate = []
    for full in full_rows:
        budget = full["nominal_budget"]
        relevant = [row for row in loo_rows if row["nominal_budget"] == budget]
        item = {"budget": budget, "full_ensemble": full, "loo_members": len(relevant)}
        for metric in METRICS:
            values = [float(row[metric]) for row in relevant]
            item[f"{metric}_loo_mean"] = mean(values)
            item[f"{metric}_loo_std"] = stdev(values)
            item[f"{metric}_loo_min"] = min(values)
            item[f"{metric}_loo_max"] = max(values)
        item["all_caps_respected"] = full["cap_respected"] and all(row["cap_respected"] for row in relevant)
        aggregate.append(item)

    result = {
        "evidence_class": manifest["evidence_class"],
        "manifest_sha256": sha256_file(OUT / "pre_evaluation_manifest.json"),
        "aggregate": aggregate,
        "full_ensemble": full_rows,
        "leave_one_member_out": loo_rows,
        "all_caps_respected": all(item["all_caps_respected"] for item in aggregate),
    }
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "all_caps_respected": result["all_caps_respected"]}, indent=2))


if __name__ == "__main__":
    main()
