"""Evaluate frozen 20% comparators on V4 after the ensemble confirmation."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy, load_frozen_policies
from share.data import load_episodes
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig

DATASET = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
OUT = ROOT / "results/share_v4_frozen_comparator_posthoc_20260829"
ARTIFACTS = ROOT / "results/share_budget_matched_frozen_artifacts_20260825"
CHECKPOINTS = ROOT / "results/share_controller_rescue_20pct_5seed_20260825"
CALIBRATION = ROOT / "results/share_budget_matched_20pct_5seed_20260825"
SEEDS = (17, 29, 43, 71, 101)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn")
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("V4 comparator post-hoc evaluation already exists")
    OUT.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(DATASET)
    cap = math.floor(0.20 * len(episodes))
    encoder = FeatureEncoder.load(CHECKPOINTS / "feature_encoder.json")
    reward = RewardConfig()
    rows = []
    for seed in SEEDS:
        frozen = load_frozen_policies(seed, ARTIFACTS, CHECKPOINTS, CALIBRATION, encoder, reward)
        for policy in frozen:
            if policy.name not in POLICIES:
                continue
            wrapper = OnlineHardHumanCapPolicy(policy, cap, policy.name in {"share_conf", "share_static"})
            raw = evaluate_policy(wrapper, episodes, reward, 0.20, None)
            row = {key: value for key, value in raw.items() if key != "records"}
            human_actions = sum(
                int(action == int(AuthorityAction.HUMAN))
                for record in raw["records"] for action in record["actions"]
            )
            row.update({
                "policy": policy.name, "seed": seed,
                "human_interventions": wrapper.human_interventions,
                "human_actions": human_actions,
                "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
                "cap_respected": wrapper.human_interventions <= cap,
            })
            rows.append(row)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["policy"]].append(row)
    aggregate = []
    for policy in POLICIES:
        item = {"policy": policy, "seeds": len(buckets[policy])}
        for metric in METRICS:
            values = [float(row[metric]) for row in buckets[policy]]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values)
        item["all_caps_respected"] = all(bool(row["cap_respected"]) for row in buckets[policy])
        aggregate.append(item)
    result = {
        "evidence_class": "post-confirmation evaluation of previously frozen comparators; no fitting or selection",
        "dataset": "V4 250 workflow-group-disjoint replay holdout",
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (OUT / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
