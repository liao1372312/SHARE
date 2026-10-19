"""Calibrate and gate root-masked SHARE before any retrospective holdout replay."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, SEEDS, sha256_file
from scripts.run_share_component_e2_232 import ENCODER, HOLDOUT, OUT as COMPONENT_OUT, choose_threshold, state_values
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import RewardConfig


MASKED = ROOT / "datasets/share_root_masked_development_20260930/episodes.jsonl"
NEW_OUT = ROOT / "results/share_root_masked_development_20260930"
EXPECTED_MASKED = "e50f7a1e24483bdb8123d3d4587d6612ae48d1ff0b28afd88b7bfc63e89d3b56"
EXPECTED_HOLDOUT = "52e528dc385368af0329a0d9711c43fba0e9ee4827862341dd4a19f301631989"
PROTOCOL = "share-training-root-oov-alignment-v1"
CALIBRATION = NEW_OUT / "calibration_matched.json"
GATE = NEW_OUT / "development_test_gate.json"
RESULT = NEW_OUT / "exploratory_232_result.json"
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")
VARIANTS = {
    "root": ("share_root_masked_development_20260930", "e50f7a1e24483bdb8123d3d4587d6612ae48d1ff0b28afd88b7bfc63e89d3b56", "share-training-root-oov-alignment-v1"),
    "root_domain": ("share_root_domain_masked_development_20260930", "80a78b51996eb0e9d54c37fa23a860245e56d299889ff3c53b5e35a93fe0ce11", "share-training-root-domain-oov-alignment-v1"),
}


def set_variant(variant: str) -> None:
    global MASKED, NEW_OUT, EXPECTED_MASKED, PROTOCOL, CALIBRATION, GATE, RESULT
    name, EXPECTED_MASKED, PROTOCOL = VARIANTS[variant]
    MASKED = ROOT / "datasets" / name / "episodes.jsonl"
    NEW_OUT = ROOT / "results" / name
    CALIBRATION = NEW_OUT / "calibration_matched.json"
    GATE = NEW_OUT / "development_test_gate.json"
    RESULT = NEW_OUT / "exploratory_232_result.json"


def load_ensembles():
    encoder = FeatureEncoder.load(ENCODER)
    reward = RewardConfig(rollback_penalty=0.60)
    old_members = [DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, reward, require_reward_config=True) for seed in SEEDS]
    new_members = []
    for seed in SEEDS:
        path = NEW_OUT / "checkpoints" / f"root_masked_seed_{seed}.pt"
        audit = NEW_OUT / "training_audit" / f"seed_{seed}.json"
        if not path.exists() or not audit.exists() or json.loads(audit.read_text())["gradient_steps"] != 30_000:
            raise SystemExit(f"missing verified new checkpoint {seed}")
        new_members.append(DQNPolicy.load(str(path), encoder, reward, require_reward_config=True))
    old = DQNEnsemblePolicy(old_members, encoder, old_members[0].config, reward)
    new = DQNEnsemblePolicy(new_members, encoder, new_members[0].config, reward)
    old.human_calibration_penalty = json.loads((COMPONENT_OUT / "calibration.json").read_text())["thresholds"]["Full SHARE:ensemble"]["threshold"]
    return old, new


def score(policy, episodes) -> tuple[dict, list[dict]]:
    cap = math.floor(0.20 * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, False)
    result = evaluate_policy(wrapper, episodes, RewardConfig(), cap / len(episodes), None)
    values = {key: result[key] for key in METRICS}
    values.update({"episodes": len(episodes), "human_count": wrapper.human_interventions, "success_count": sum(bool(row["success"]) for row in result["records"]), "stop_count": sum(row["actions"][-1] == 2 for row in result["records"])})
    return values, result["records"]


def check_data() -> None:
    if sha256_file(MASKED) != EXPECTED_MASKED or sha256_file(HOLDOUT) != EXPECTED_HOLDOUT:
        raise SystemExit("dataset hash changed")


def calibrate() -> None:
    if CALIBRATION.exists():
        raise SystemExit("refusing to overwrite calibration")
    check_data()
    episodes = load_episodes(MASKED)
    validation = [ep for ep in episodes if ep.split == "validation"]
    old, new = load_ensembles()
    states = [step.state for ep in validation for step in ep.steps]
    old_values = old._raw_values_batch(states)
    old_selected = choose_threshold(validation, state_values(validation, old_values), "neural")
    old.human_calibration_penalty = old_selected["threshold"]
    values = new._raw_values_batch(states)
    selected = choose_threshold(validation, state_values(validation, values), "neural")
    new.human_calibration_penalty = selected["threshold"]
    old_metrics, _ = score(old, validation)
    new_metrics, _ = score(new, validation)
    old_uncapped = evaluate_policy(old, validation, RewardConfig())["hir"]
    uncapped = evaluate_policy(new, validation, RewardConfig())["hir"]
    if not 0.18 <= uncapped <= 0.22 or not 0.18 <= old_uncapped <= 0.22:
        raise SystemExit(f"uncapped validation HIR outside band: old={old_uncapped}, new={uncapped}")
    artifact = {
        "protocol": PROTOCOL,
        "status": "calibration_on_root_masked_development_validation_only",
        "masked_development_sha256": EXPECTED_MASKED,
        "holdout_sha256_for_identity_only": EXPECTED_HOLDOUT,
        "old_checkpoint_sha256": {str(seed): sha256_file(CHECKPOINTS[seed]) for seed in SEEDS},
        "new_checkpoint_sha256": {str(seed): sha256_file(NEW_OUT / "checkpoints" / f"root_masked_seed_{seed}.pt") for seed in SEEDS},
        "old_selected": old_selected,
        "selected": selected,
        "old_realized_uncapped_validation_hir": old_uncapped,
        "realized_uncapped_validation_hir": uncapped,
        "old_capped_validation": old_metrics,
        "new_capped_validation": new_metrics,
        "holdout_outcomes_read": False,
    }
    CALIBRATION.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


def gate() -> None:
    if not CALIBRATION.exists() or GATE.exists():
        raise SystemExit("missing calibration or refusing to overwrite gate")
    check_data()
    calibration = json.loads(CALIBRATION.read_text())
    old, new = load_ensembles()
    old.human_calibration_penalty = calibration["old_selected"]["threshold"]
    new.human_calibration_penalty = calibration["selected"]["threshold"]
    test = [ep for ep in load_episodes(MASKED) if ep.split == "test"]
    before, _ = score(old, test)
    after, _ = score(new, test)
    passes = (after["repair_sr"] >= before["repair_sr"] - 1e-12 and after["uar"] <= before["uar"] + 1e-12 and after["stopped_rate"] <= before["stopped_rate"] + 1e-12 and (after["repair_sr"] > before["repair_sr"] + 1e-12 or after["uar"] < before["uar"] - 1e-12 or after["stopped_rate"] < before["stopped_rate"] - 1e-12))
    artifact = {"status": "development_test_gate_no_reselection", "calibration_sha256": sha256_file(CALIBRATION), "old": before, "new": after, "passes_joint_objective": passes, "holdout_outcomes_read": False}
    GATE.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


def evaluate() -> None:
    if not GATE.exists() or RESULT.exists():
        raise SystemExit("missing gate or refusing to overwrite result")
    check_data()
    gate_record = json.loads(GATE.read_text())
    if not gate_record["passes_joint_objective"]:
        raise SystemExit("development-test gate failed; holdout remains unevaluated")
    calibration = json.loads(CALIBRATION.read_text())
    old, new = load_ensembles()
    old.human_calibration_penalty = calibration["old_selected"]["threshold"]
    new.human_calibration_penalty = calibration["selected"]["threshold"]
    episodes = load_episodes(HOLDOUT)
    before, before_records = score(old, episodes)
    after, after_records = score(new, episodes)
    result = {"status": "exploratory_on_previously_inspected_232_group_holdout", "gate_sha256": sha256_file(GATE), "old": before, "new": after, "workflow_paired": [r["episode_id"] for r in before_records] == [r["episode_id"] for r in after_records]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="root")
    parser.add_argument("phase", choices=("calibrate", "gate", "evaluate"))
    args = parser.parse_args()
    set_variant(args.variant)
    {"calibrate": calibrate, "gate": gate, "evaluate": evaluate}[args.phase]()
