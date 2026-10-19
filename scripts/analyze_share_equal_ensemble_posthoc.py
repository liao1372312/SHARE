"""Post-hoc equal-aggregation comparison on the frozen V4 replay holdout."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS, sha256_file
from scripts.run_share_budget_matched_comparison import (
    breakpoint_candidates,
    calibrate_scalar_validation_only,
)
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig

HOLDOUT = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
DEVELOPMENT = ROOT / "datasets/share_synthetic/episodes.jsonl"
DDQN_ROOT = ROOT / "results/share_controller_rescue_20pct_5seed_20260825"
CALIBRATION_ROOT = ROOT / "results/share_budget_matched_20pct_5seed_20260825"
OUT = ROOT / "results/share_equal_ensemble_posthoc_20260929"
SHARE_ENSEMBLE_PENALTY = 1.9403412342071533
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def score(policy, episodes, reward):
    wrapper = OnlineHardHumanCapPolicy(policy, math.floor(0.20 * len(episodes)), False)
    raw = evaluate_policy(wrapper, episodes, reward, 0.20, None)
    return {
        **{key: raw[key] for key in METRICS},
        "human_interventions": wrapper.human_interventions,
        "cap_respected": wrapper.human_interventions <= wrapper.cap,
        "coverage": 1.0 - raw["stopped_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    holdout = load_episodes(HOLDOUT)
    validation = [episode for episode in load_episodes(DEVELOPMENT) if episode.split == "validation"]
    assert len(holdout) == 250 and len(validation) == 127
    encoder = FeatureEncoder.load(ENCODER)
    share_reward = RewardConfig(rollback_penalty=0.60)
    ddqn_reward = RewardConfig(rollback_penalty=0.25)
    evaluation_reward = RewardConfig()
    share_members = []
    ddqn_members = []
    rows = []
    ddqn_paths = {}
    for seed in SEEDS:
        share = DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, share_reward, require_reward_config=True)
        share.name = f"share_single_{seed}"
        share_members.append(share)
        rows.append({"method": "SHARE", "members": 1, "seed": seed,
                     "human_penalty": share.human_calibration_penalty,
                     **score(share, holdout, evaluation_reward)})

        path = DDQN_ROOT / f"share_ddqn_seed_{seed}.pt"
        ddqn_paths[str(seed)] = sha256_file(path)
        ddqn = DQNPolicy.load(str(path), encoder, ddqn_reward, require_reward_config=True)
        calibration = json.loads((CALIBRATION_ROOT / f"seed_{seed}.json").read_text())["calibrations"]["share_ddqn"]
        assert calibration["selection_split"] == "validation"
        ddqn.human_calibration_penalty = float(calibration["selected_scalar"])
        ddqn.ai_calibration_penalty = 0.0
        ddqn.name = f"ddqn_single_{seed}"
        ddqn_members.append(ddqn)
        rows.append({"method": "DDQN", "members": 1, "seed": seed,
                     "human_penalty": ddqn.human_calibration_penalty,
                     **score(ddqn, holdout, evaluation_reward)})

    share_ensemble = DQNEnsemblePolicy(share_members, encoder, share_members[0].config, share_reward)
    share_ensemble.name = "share_five"
    share_ensemble.human_calibration_penalty = SHARE_ENSEMBLE_PENALTY
    rows.append({"method": "SHARE", "members": 5, "seed": None,
                 "human_penalty": SHARE_ENSEMBLE_PENALTY,
                 **score(share_ensemble, holdout, evaluation_reward)})

    ddqn_ensemble = DQNEnsemblePolicy(ddqn_members, encoder, ddqn_members[0].config, ddqn_reward)
    ddqn_ensemble.name = "ddqn_five"
    validation_states = [step.state for episode in validation for step in episode.steps]
    raw_values = ddqn_ensemble._raw_values_batch(validation_states)
    advantages = raw_values[:, AuthorityAction.HUMAN] - np.maximum(
        raw_values[:, AuthorityAction.AI], raw_values[:, AuthorityAction.STOP]
    )
    ddqn_calibration = calibrate_scalar_validation_only(
        ddqn_ensemble, validation, breakpoint_candidates(advantages.tolist()),
        lambda value: setattr(ddqn_ensemble, "human_calibration_penalty", value),
        target=0.20, lower=0.18, upper=0.22,
    )
    rows.append({"method": "DDQN", "members": 5, "seed": None,
                 "human_penalty": ddqn_ensemble.human_calibration_penalty,
                 **score(ddqn_ensemble, holdout, evaluation_reward)})

    aggregate = []
    for method in ("SHARE", "DDQN"):
        items = [row for row in rows if row["method"] == method and row["members"] == 1]
        summary = {"method": method, "members": 1, "seeds": len(items)}
        for metric in (*METRICS, "coverage"):
            values = [row[metric] for row in items]
            summary[metric] = mean(values)
            summary[f"{metric}_sd"] = stdev(values)
        aggregate.append(summary)
        aggregate.append(next(row for row in rows if row["method"] == method and row["members"] == 5))
    result = {
        "evidence_class": "post-confirmation frozen-checkpoint analysis on a previously used simulated replay holdout",
        "interpretation": "Controls aggregation count, not the differing training reward and calibration selection protocols; five singles are not an ensemble.",
        "holdout_sha256": sha256_file(HOLDOUT),
        "development_sha256": sha256_file(DEVELOPMENT),
        "share_checkpoint_sha256": {str(seed): sha256_file(CHECKPOINTS[seed]) for seed in SEEDS},
        "ddqn_checkpoint_sha256": ddqn_paths,
        "encoder_sha256": sha256_file(ENCODER),
        "ddqn_ensemble_validation_calibration": ddqn_calibration,
        "rows": rows,
        "aggregate": aggregate,
    }
    (args.out / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
