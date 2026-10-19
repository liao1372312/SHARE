"""Train an AI-risk-assisted SHARE router without using holdout outcomes.

The public-API holdouts have already been inspected in earlier experiments.
The development-test and holdout checks here are retrospective diagnostics,
not independent confirmation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.run_share_component_e2_232 import DEVELOPMENT, ENCODER, EXPECTED_DEV, EXPECTED_HOLDOUT, HOLDOUT, OUT, choose_threshold, load_policies
from scripts.tune_share_component_validation_router import CachedQBiasPolicy, metrics, raw_values
from share.data import load_episodes
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairState


LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
AI_BONUSES = (0.0, 0.25, 0.5)
MODEL_PATH = OUT / "risk_router_model.joblib"
TUNE_PATH = OUT / "risk_router_validation_tuning.json"
CHECK_PATH = OUT / "risk_router_development_test_check.json"
RESULT_PATH = OUT / "exploratory_risk_router_232.json"


class CachedRiskQPolicy(CachedQBiasPolicy):
    name = "full_share_ai_risk_assisted"

    def __init__(self, raw, risk, ai_bonus: float, risk_weight: float, human_penalty: float):
        super().__init__(raw, ai_bonus, human_penalty)
        self.risk = risk
        self.risk_weight = float(risk_weight)

    def _values(self, state: RepairState) -> np.ndarray:
        values = super()._values(state)
        values[AuthorityAction.HUMAN] += self.risk_weight * self.risk[(state.episode_id, state.step)]
        return values


def risk_labels(episodes):
    return np.asarray([
        int(outcome.unsafe or (outcome.terminal and not outcome.success))
        for ep in episodes for step in ep.steps
        for outcome in [step.outcome(AuthorityAction.AI)]
    ])


def predict_risk(model, encoder, episodes):
    states = [step.state for ep in episodes for step in ep.steps]
    scores = model.predict_proba(encoder.transform(states))[:, list(model.classes_).index(1)]
    return {(ep.episode_id, step.state.step): float(score)
            for ep, group in zip(episodes, np.split(scores, np.cumsum([len(ep.steps) for ep in episodes])[:-1]))
            for step, score in zip(ep.steps, group)}


def load_share_ensemble(encoder):
    calibration = json.loads((OUT / "calibration.json").read_text())
    ensemble = load_policies(encoder, calibration)["Full SHARE"][-1]
    baseline_penalty = calibration["thresholds"]["Full SHARE:ensemble"]["threshold"]
    return ensemble, baseline_penalty


def train_and_tune() -> None:
    if MODEL_PATH.exists() or TUNE_PATH.exists() or CHECK_PATH.exists():
        raise SystemExit("refusing to overwrite risk-router training artifacts")
    if sha256_file(DEVELOPMENT) != EXPECTED_DEV or sha256_file(HOLDOUT) != EXPECTED_HOLDOUT:
        raise SystemExit("dataset changed")
    development = load_episodes(DEVELOPMENT)
    train = [ep for ep in development if ep.split == "train"]
    validation = [ep for ep in development if ep.split == "validation"]
    dev_test = [ep for ep in development if ep.split == "test"]
    encoder = FeatureEncoder.load(ENCODER)
    states = [step.state for ep in train for step in ep.steps]
    labels = risk_labels(train)
    risk_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=2026100101, n_jobs=4)
    risk_model.fit(encoder.transform(states), labels)
    joblib.dump(risk_model, MODEL_PATH)
    validation_risk = predict_risk(risk_model, encoder, validation)
    val_labels = risk_labels(validation)
    val_auc = float(roc_auc_score(val_labels, list(validation_risk.values())))
    ensemble, baseline_penalty = load_share_ensemble(encoder)
    raw = raw_values(ensemble, validation)
    baseline = CachedRiskQPolicy(raw, validation_risk, 0.0, 0.0, baseline_penalty)
    baseline_metrics, _ = metrics(baseline, validation, 25)
    candidates = [{"risk_weight": 0.0, "ai_bonus": 0.0, "human_penalty": baseline_penalty, "validation": baseline_metrics, "uncapped_hir": 25 / 127, "baseline": True}]
    for weight in LAMBDAS:
        for bonus in AI_BONUSES:
            if weight == 0.0 and bonus == 0.0:
                continue
            adjusted = {}
            for key, value in raw.items():
                row = value.copy()
                row[AuthorityAction.HUMAN] += weight * validation_risk[key]
                row[AuthorityAction.AI] += bonus
                adjusted[key] = row
            choice = choose_threshold(validation, adjusted, "neural")
            policy = CachedRiskQPolicy(raw, validation_risk, bonus, weight, choice["threshold"])
            uncapped, _ = metrics(policy, validation, len(validation))
            capped, _ = metrics(policy, validation, 25)
            candidates.append({"risk_weight": weight, "ai_bonus": bonus, "human_penalty": choice["threshold"], "validation": capped, "uncapped_hir": uncapped["hir"], "baseline": False})
    admissible = [row for row in candidates if 0.18 <= row["uncapped_hir"] <= 0.22 and row["validation"]["uar"] <= baseline_metrics["uar"] + 1e-12 and row["validation"]["repair_sr"] >= baseline_metrics["repair_sr"] - 1e-12 and row["validation"]["stopped_rate"] <= baseline_metrics["stopped_rate"] + 1e-12]
    selected = max(admissible, key=lambda row: (row["validation"]["repair_sr"], -row["validation"]["uar"], -row["validation"]["stopped_rate"], row["validation"]["normalized_return"], -row["risk_weight"], -row["ai_bonus"]))
    tuning = {
        "status": "development_validation_selection_after_prior_holdout_inspection",
        "holdout_use": "none during this selection; later checks remain retrospective",
        "development_sha256": EXPECTED_DEV, "holdout_sha256": EXPECTED_HOLDOUT,
        "model_sha256": sha256_file(MODEL_PATH), "calibration_sha256": sha256_file(OUT / "calibration.json"),
        "risk_target": "AI branch unsafe OR terminal unsuccessful",
        "training_states": len(states), "training_positive": int(labels.sum()),
        "validation_states": len(val_labels), "validation_risk_auroc": val_auc,
        "risk_weights": list(LAMBDAS), "ai_bonuses": list(AI_BONUSES),
        "selection_rule": "uncapped HIR 0.18-0.22; capped Repair.SR no lower, UAR and Stop no higher than baseline; maximize Repair.SR, then minimize UAR and Stop, then maximize Return",
        "baseline": candidates[0], "candidates": candidates, "selected": selected,
    }
    TUNE_PATH.write_text(json.dumps(tuning, indent=2) + "\n")
    raw_test = raw_values(ensemble, dev_test)
    risk_test = predict_risk(risk_model, encoder, dev_test)
    baseline_test, _ = metrics(CachedRiskQPolicy(raw_test, risk_test, 0, 0, baseline_penalty), dev_test, 24)
    selected_test, _ = metrics(CachedRiskQPolicy(raw_test, risk_test, selected["ai_bonus"], selected["risk_weight"], selected["human_penalty"]), dev_test, 24)
    passes = selected_test["repair_sr"] >= baseline_test["repair_sr"] - 1e-12 and selected_test["uar"] <= baseline_test["uar"] + 1e-12 and selected_test["stopped_rate"] <= baseline_test["stopped_rate"] + 1e-12
    check = {"status": "retrospective_development_test_no_reselection", "episodes": len(dev_test), "baseline": baseline_test, "selected": selected_test, "passes_joint_objective": passes}
    CHECK_PATH.write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps({"validation_risk_auroc": val_auc, "baseline": candidates[0], "selected": selected, "development_test_check": check}, indent=2))


def evaluate() -> None:
    if not TUNE_PATH.is_file() or not CHECK_PATH.is_file() or RESULT_PATH.exists():
        raise SystemExit("missing frozen tuning or refusing to overwrite result")
    tuning = json.loads(TUNE_PATH.read_text())
    check = json.loads(CHECK_PATH.read_text())
    if not check["passes_joint_objective"]:
        raise SystemExit("risk router failed the development-test joint objective; no holdout evaluation")
    if sha256_file(MODEL_PATH) != tuning["model_sha256"] or sha256_file(HOLDOUT) != tuning["holdout_sha256"]:
        raise SystemExit("frozen artifact changed")
    encoder = FeatureEncoder.load(ENCODER)
    ensemble, baseline_penalty = load_share_ensemble(encoder)
    model = joblib.load(MODEL_PATH)
    episodes = load_episodes(HOLDOUT)
    raw = raw_values(ensemble, episodes)
    risk = predict_risk(model, encoder, episodes)
    baseline, _ = metrics(CachedRiskQPolicy(raw, risk, 0, 0, baseline_penalty), episodes, 46)
    selected = tuning["selected"]
    candidate, records = metrics(CachedRiskQPolicy(raw, risk, selected["ai_bonus"], selected["risk_weight"], selected["human_penalty"]), episodes, 46)
    result = {"status": "exploratory_repeat_on_previously_inspected_232_group_holdout", "tuning_sha256": sha256_file(TUNE_PATH), "development_test_sha256": sha256_file(CHECK_PATH), "baseline": baseline, "candidate": candidate, "selected": selected, "success_count": sum(bool(row["success"]) for row in records), "stop_count": sum(row["actions"][-1] == 2 for row in records)}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tune", "evaluate"))
    args = parser.parse_args()
    train_and_tune() if args.phase == "tune" else evaluate()
