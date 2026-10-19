"""Validation-only Stop/AI preference sweep for a retrospective 232-group check.

The 232-group holdout was already inspected for the original component table.
Any result from this sweep on that holdout is exploratory and cannot replace
the frozen comparison or be called independent confirmation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.run_share_component_e2_232 import (
    CAP, DEVELOPMENT, ENCODER, EXPECTED_DEV, EXPECTED_HOLDOUT, HOLDOUT,
    OUT, choose_threshold, compact, load_policies, state_values,
)
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import deterministic_action
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import AuthorityAction, RepairState, RewardConfig


AI_BONUSES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
TUNE_PATH = OUT / "validation_router_tuning.json"
RESULT_PATH = OUT / "exploratory_validation_router_232.json"


class CachedQBiasPolicy(Policy):
    name = "full_share_validation_tuned_ai_bias"

    def __init__(self, raw: dict[tuple[str, int], np.ndarray], ai_bonus: float, human_penalty: float):
        self.raw = raw
        self.ai_bonus = float(ai_bonus)
        self.human_penalty = float(human_penalty)

    def _values(self, state: RepairState) -> np.ndarray:
        values = self.raw[(state.episode_id, state.step)].copy()
        values[AuthorityAction.AI] += self.ai_bonus
        values[AuthorityAction.HUMAN] -= self.human_penalty
        return values

    def select(self, state: RepairState, feasible) -> AuthorityAction:
        return deterministic_action(self._values(state), feasible)

    def escalation_score(self, state: RepairState) -> float:
        values = self._values(state)
        return float(values[AuthorityAction.HUMAN] - values[AuthorityAction.AI])


def raw_values(ensemble, episodes) -> dict[tuple[str, int], np.ndarray]:
    return {(ep.episode_id, step.state.step): ensemble._raw_values(step.state) for ep in episodes for step in ep.steps}


def metrics(policy: Policy, episodes, cap: int) -> tuple[dict, list[dict]]:
    wrapped = OnlineHardHumanCapPolicy(policy, cap, False)
    result = evaluate_policy(wrapped, episodes, RewardConfig(), cap / len(episodes), None)
    return {**compact(result), "human_count": wrapped.human_interventions}, result["records"]


def tune() -> None:
    if TUNE_PATH.exists():
        raise SystemExit("refusing to overwrite validation-only tuning")
    if sha256_file(DEVELOPMENT) != EXPECTED_DEV or sha256_file(HOLDOUT) != EXPECTED_HOLDOUT:
        raise SystemExit("dataset hash mismatch")
    calibration_path = OUT / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    validation = [ep for ep in load_episodes(DEVELOPMENT) if ep.split == "validation"]
    ensemble = load_policies(FeatureEncoder.load(ENCODER), calibration)["Full SHARE"][-1]
    raw = raw_values(ensemble, validation)
    baseline_penalty = calibration["thresholds"]["Full SHARE:ensemble"]["threshold"]
    baseline = CachedQBiasPolicy(raw, 0.0, baseline_penalty)
    baseline_metrics, baseline_records = metrics(baseline, validation, 25)
    original_metrics, original_records = metrics(ensemble, validation, 25)
    if [row["actions"] for row in baseline_records] != [row["actions"] for row in original_records]:
        raise RuntimeError("cached baseline does not reproduce frozen ensemble actions")
    if any(abs(baseline_metrics[k] - original_metrics[k]) > 1e-12 for k in ("repair_sr", "hir", "uar", "stopped_rate")):
        raise RuntimeError("cached baseline does not reproduce frozen ensemble metrics")
    candidates = [{"ai_bonus": 0.0, "human_penalty": baseline_penalty, "validation": baseline_metrics, "uncapped_validation_hir": calibration["thresholds"]["Full SHARE:ensemble"]["validation_hir"], "baseline": True}]
    for bonus in AI_BONUSES[1:]:
        biased = {key: value + np.asarray([bonus, 0.0, 0.0], dtype=value.dtype) for key, value in raw.items()}
        choice = choose_threshold(validation, biased, "neural")
        policy = CachedQBiasPolicy(raw, bonus, choice["threshold"])
        uncapped, _ = metrics(policy, validation, len(validation))
        capped, _ = metrics(policy, validation, 25)
        candidates.append({"ai_bonus": bonus, "human_penalty": choice["threshold"], "validation": capped, "uncapped_validation_hir": uncapped["hir"], "baseline": False})
    admissible = [row for row in candidates if 0.18 <= row["uncapped_validation_hir"] <= 0.22 and row["validation"]["uar"] <= baseline_metrics["uar"] + 1e-12 and row["validation"]["stopped_rate"] <= baseline_metrics["stopped_rate"] + 1e-12 and row["validation"]["repair_sr"] >= baseline_metrics["repair_sr"] - 1e-12]
    selected = max(admissible, key=lambda row: (row["validation"]["repair_sr"], -row["validation"]["uar"], -row["validation"]["stopped_rate"], row["validation"]["normalized_return"], -row["ai_bonus"]))
    artifact = {
        "status": "validation_only_selection_after_original_232_holdout_inspection",
        "interpretation": "any later 232-group result is exploratory, not independent confirmation",
        "development_sha256": EXPECTED_DEV,
        "holdout_sha256": EXPECTED_HOLDOUT,
        "calibration_sha256": sha256_file(calibration_path),
        "candidate_ai_bonuses": list(AI_BONUSES),
        "selection_rule": "uncapped validation HIR in [0.18,0.22]; capped UAR and Stop no worse than baseline, Repair.SR no worse; maximize Repair.SR, then minimize UAR, Stop, then maximize Return",
        "baseline": candidates[0], "candidates": candidates, "selected": selected,
        "holdout_outcomes_read_during_tuning": False,
    }
    TUNE_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"baseline": candidates[0], "selected": selected, "candidates": candidates}, indent=2))


def evaluate() -> None:
    if not TUNE_PATH.exists() or RESULT_PATH.exists():
        raise SystemExit("missing tuning or refusing to overwrite exploratory result")
    tuning = json.loads(TUNE_PATH.read_text())
    if tuning["holdout_sha256"] != sha256_file(HOLDOUT) or tuning["calibration_sha256"] != sha256_file(OUT / "calibration.json"):
        raise SystemExit("frozen input changed")
    episodes = load_episodes(HOLDOUT)
    calibration = json.loads((OUT / "calibration.json").read_text())
    ensemble = load_policies(FeatureEncoder.load(ENCODER), calibration)["Full SHARE"][-1]
    raw = raw_values(ensemble, episodes)
    choice = tuning["selected"]
    policy = CachedQBiasPolicy(raw, choice["ai_bonus"], choice["human_penalty"])
    observed, records = metrics(policy, episodes, CAP)
    artifact = {
        "status": "exploratory_repeat_on_previously_inspected_232_group_holdout",
        "tuning_sha256": sha256_file(TUNE_PATH), "holdout_sha256": EXPECTED_HOLDOUT,
        "selected": choice, "holdout_metrics": observed,
        "success": sum(bool(row["success"]) for row in records),
        "stop": sum(row["actions"][-1] == int(AuthorityAction.STOP) for row in records),
        "autonomous_completed": sum(not row["human_used"] and row["actions"][-1] != int(AuthorityAction.STOP) for row in records),
    }
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tune", "evaluate"))
    args = parser.parse_args()
    tune() if args.phase == "tune" else evaluate()
