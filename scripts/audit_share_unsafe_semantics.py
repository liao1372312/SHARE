"""Audit branch and trajectory unsafe flags in the frozen primary replay."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy, oracle_state_values
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig

DATASET = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
OUT = ROOT / "results/share_unsafe_semantics_audit_20260930.json"
ARCHIVED = ROOT / "results/share_v4_ensemble_confirmation_20260829/FINAL_RESULTS.json"


def main() -> None:
    episodes = load_episodes(DATASET)
    branch_counts = {}
    for action in AuthorityAction:
        branches = [step.outcome(action) for episode in episodes for step in episode.steps]
        counts = Counter((branch.success, branch.unsafe) for branch in branches)
        branch_counts[action.name.lower()] = {
            f"success_{int(success)}_unsafe_{int(unsafe)}": counts[success, unsafe]
            for success in (False, True) for unsafe in (False, True)
        }
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    members = [
        DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True)
        for seed in SEEDS
    ]
    policy = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    policy.human_calibration_penalty = 1.9403412342071533
    wrapper = OnlineHardHumanCapPolicy(policy, math.floor(0.20 * len(episodes)), False)
    result = evaluate_policy(wrapper, episodes, RewardConfig(), 0.20, None)
    records = result["records"]
    autonomous = [
        record for record in records
        if not record["human_used"] and record["actions"][-1] != AuthorityAction.STOP
    ]
    cross_all = Counter((bool(r["success"]), bool(r["unsafe"])) for r in records)
    cross_autonomous = Counter((bool(r["success"]), bool(r["unsafe"])) for r in autonomous)

    def cross_table(counts):
        return {
            f"success_{int(success)}_unsafe_{int(unsafe)}": counts[success, unsafe]
            for success in (False, True) for unsafe in (False, True)
        }

    selected = [value for record in records for value in record["selected_returns"]]
    oracle = [value for record in records for value in record["oracle_returns"]]
    initial_oracle = [oracle_state_values(ep, RewardConfig())[0] for ep in episodes]
    audit = {
        "episodes": len(episodes),
        "branch_counts": branch_counts,
        "policy_summary": {
            key: result[key]
            for key in ("repair_sr", "hir", "uar", "stopped_rate", "unsafe_autonomy_cost_rate")
        },
        "trajectory_counts": {
            "final_success": sum(bool(r["success"]) for r in records),
            "human_episodes": sum(bool(r["human_used"]) for r in records),
            "stop_episodes": sum(r["actions"][-1] == AuthorityAction.STOP for r in records),
            "success_and_unsafe_all": sum(bool(r["success"] and r["unsafe"]) for r in records),
            "success_and_unsafe_autonomous_nonstop": sum(bool(r["success"] and r["unsafe"]) for r in autonomous),
            "unsafe_all": sum(bool(r["unsafe"]) for r in records),
            "unsafe_action_events": sum(r["unsafe_actions"] for r in records),
            "autonomous_nonstop_episodes": len(autonomous),
            "success_unsafe_cross_all": cross_table(cross_all),
            "success_unsafe_cross_autonomous_nonstop": cross_table(cross_autonomous),
        },
        "return_regret_audit": {
            "episode_mean_undiscounted_return": result["normalized_return"],
            "visited_decisions": len(selected),
            "visited_state_mean_oracle_discounted_value": sum(oracle) / len(oracle),
            "visited_state_mean_selected_discounted_return": sum(selected) / len(selected),
            "visited_state_mean_regret": sum(best - chosen for chosen, best in zip(selected, oracle)) / len(selected),
            "episode_mean_initial_oracle_discounted_value": sum(initial_oracle) / len(initial_oracle),
            "episode_mean_initial_selected_discounted_return": sum(r["selected_returns"][0] for r in records) / len(records),
        },
    }
    archived = json.loads(ARCHIVED.read_text(encoding="utf-8"))
    for metric in ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret"):
        assert math.isclose(result[metric], archived[metric], abs_tol=1e-12), metric
    assert math.isclose(
        audit["return_regret_audit"]["visited_state_mean_regret"],
        result["collaboration_regret"], abs_tol=1e-12,
    )
    assert cross_autonomous[True, True] == 10
    assert sum(not r["success"] or r["unsafe"] for r in autonomous) == 13
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
