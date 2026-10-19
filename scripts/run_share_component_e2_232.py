"""Calibrate and evaluate the frozen 232-group matched component comparison."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, SEEDS, sha256_file
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy, deterministic_action
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy, SupervisedBenefitPolicy
from share.types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


OUT = ROOT / "results/share_component_confirmation_e2_232_20260930"
DEVELOPMENT = ROOT / "datasets/share_synthetic/episodes.jsonl"
HOLDOUT = ROOT / "datasets/share_component_confirmation_232_20260930/episodes.jsonl"
ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
EXPECTED_DEV = "174a2c387aa8fe4cece6374a54a9d23c5db062bde6349c71578384c386b5a694"
EXPECTED_HOLDOUT = "52e528dc385368af0329a0d9711c43fba0e9ee4827862341dd4a19f301631989"
METHODS = ("Full SHARE", "SHARE-Myopic", "SHARE-w/o-Paired", "Matched DDQN", "Static")
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")
CONTRAST_METRICS = ("repair_sr", "uar", "stopped_rate")
CAP = 46
BOOTSTRAP_DRAWS = 20_000


class StaticMeanPolicy(Policy):
    name = "matched_static_five"

    def __init__(self, members: list[SupervisedBenefitPolicy]):
        self.members = members
        self.threshold = 0.5

    def escalation_score(self, state: RepairState) -> float:
        return mean(member._score(state) for member in self.members)

    def select(self, state: RepairState, feasible) -> AuthorityAction:
        if self.escalation_score(state) >= self.threshold:
            return AuthorityAction.HUMAN
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP


def checkpoint_path(method: str, seed: int) -> Path:
    if method == "Full SHARE":
        return CHECKPOINTS[seed]
    if method == "SHARE-Myopic":
        return ROOT / f"results/share_v2_final_development_ablation_20pct_20260827/checkpoints/share_v2_myopic_seed_{seed}.pt"
    variant = {"SHARE-w/o-Paired": "wo_paired", "Matched DDQN": "ddqn"}[method]
    return OUT / "checkpoints" / f"{variant}_seed_{seed}.pt"


def check_inputs() -> None:
    if sha256_file(DEVELOPMENT) != EXPECTED_DEV or sha256_file(HOLDOUT) != EXPECTED_HOLDOUT:
        raise SystemExit("development or frozen holdout hash changed")
    if not (OUT / "protocol.json").is_file():
        raise SystemExit("missing frozen protocol")
    for method in METHODS[:-1]:
        for seed in SEEDS:
            if not checkpoint_path(method, seed).is_file():
                raise SystemExit(f"missing checkpoint: {method}/{seed}")
            if method in ("SHARE-w/o-Paired", "Matched DDQN"):
                audit = OUT / "training_audit" / f"{('wo_paired' if method == 'SHARE-w/o-Paired' else 'ddqn')}_seed_{seed}.json"
                if not audit.is_file() or json.loads(audit.read_text())["gradient_steps"] != 30_000:
                    raise SystemExit(f"unverified matched training: {method}/{seed}")


def neural_models(encoder: FeatureEncoder) -> dict[str, list[DQNPolicy]]:
    reward = RewardConfig(rollback_penalty=0.60)
    result = {}
    for method in METHODS[:-1]:
        items = []
        for seed in SEEDS:
            model = DQNPolicy.load(str(checkpoint_path(method, seed)), encoder, reward, require_reward_config=True)
            expected_gamma = 0.0 if method == "SHARE-Myopic" else 0.95
            expected_paired = method in ("Full SHARE", "SHARE-Myopic")
            if model.config.gamma != expected_gamma or model.config.paired_replay != expected_paired:
                raise SystemExit(f"checkpoint config mismatch: {method}/{seed}")
            if model.config.training_epochs != 30:
                raise SystemExit(f"checkpoint epoch mismatch: {method}/{seed}")
            model.human_calibration_penalty = 0.0
            model.ai_calibration_penalty = 0.0
            items.append(model)
        result[method] = items
    return result


def validation_hir(episodes: list[RepairEpisode], values: dict[tuple[str, int], np.ndarray | float], threshold: float, kind: str) -> float:
    used = 0
    for episode in episodes:
        step = 0
        while True:
            state = episode.steps[step].state
            value = values[(episode.episode_id, step)]
            if kind == "neural":
                scores = np.asarray(value, dtype=float).copy()
                scores[AuthorityAction.HUMAN] -= threshold
                feasible = [AuthorityAction.HUMAN, AuthorityAction.STOP]
                if state.remaining_budget > 0:
                    feasible.insert(0, AuthorityAction.AI)
                action = deterministic_action(scores, feasible)
            else:
                action = AuthorityAction.HUMAN if float(value) >= threshold else AuthorityAction.AI
            if action == AuthorityAction.HUMAN:
                used += 1
                break
            outcome = episode.steps[step].outcome(action)
            if action != AuthorityAction.AI or outcome.terminal or outcome.next_step is None:
                break
            step = outcome.next_step
    return used / len(episodes)


def choose_threshold(episodes: list[RepairEpisode], values: dict[tuple[str, int], np.ndarray | float], kind: str) -> dict:
    if kind == "neural":
        advantages = [float(v[AuthorityAction.HUMAN] - max(v[AuthorityAction.AI], v[AuthorityAction.STOP])) for v in values.values()]
    else:
        advantages = [float(v) for v in values.values()]
    unique = sorted(set(advantages))
    candidates = sorted({unique[0] - 1.0, unique[-1] + 1.0, *unique, *(float(np.nextafter(x, -np.inf)) for x in unique), *(float(np.nextafter(x, np.inf)) for x in unique)})
    cache: dict[int, float] = {}

    def evaluate(index: int) -> float:
        if index not in cache:
            cache[index] = validation_hir(episodes, values, candidates[index], kind)
        return cache[index]

    low, high = 0, len(candidates) - 1
    while low <= high:
        mid = (low + high) // 2
        if evaluate(mid) > 0.20:
            low = mid + 1
        else:
            high = mid - 1
    inspect = {0, len(candidates) - 1, *range(max(0, low - 4), min(len(candidates), low + 5))}
    for index in inspect:
        evaluate(index)
    selected = min(cache, key=lambda index: (not (0.18 - 1e-12 <= cache[index] <= 0.22 + 1e-12), abs(cache[index] - 0.20), cache[index], index))
    return {"threshold": candidates[selected], "validation_hir": cache[selected], "candidate_count": len(candidates), "evaluated_candidates": len(cache)}


def state_values(episodes: list[RepairEpisode], predictions: np.ndarray) -> dict[tuple[str, int], np.ndarray | float]:
    if len(predictions) != sum(len(ep.steps) for ep in episodes):
        raise ValueError("prediction count differs from validation state count")
    result = {}
    position = 0
    for ep in episodes:
        for step_index, step in enumerate(ep.steps):
            if step.state.step != step_index:
                raise ValueError(f"nonsequential step IDs in {ep.episode_id}")
            result[(ep.episode_id, step_index)] = predictions[position]
            position += 1
    return result


def calibrate() -> None:
    check_inputs()
    path = OUT / "calibration.json"
    if path.exists():
        raise SystemExit("refusing to overwrite frozen calibration")
    development = load_episodes(DEVELOPMENT)
    training = [ep for ep in development if ep.split == "train"]
    validation = [ep for ep in development if ep.split == "validation"]
    encoder = FeatureEncoder.load(ENCODER)
    models = neural_models(encoder)
    states = [step.state for ep in validation for step in ep.steps]
    calibrated = {}
    for method, members in models.items():
        member_values = [member._raw_values_batch(states) for member in members]
        for seed, values in zip(SEEDS, member_values):
            calibrated[f"{method}:seed_{seed}"] = choose_threshold(validation, state_values(validation, values), "neural")
        ensemble_values = np.mean(np.stack(member_values), axis=0)
        calibrated[f"{method}:ensemble"] = choose_threshold(validation, state_values(validation, ensemble_values), "neural")
    static_members = []
    for seed in SEEDS:
        model = SupervisedBenefitPolicy(encoder, 0.20, seed).fit(training)
        static_path = OUT / "static_models" / f"static_seed_{seed}.joblib"
        static_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, static_path)
        static_members.append(model)
        values = np.asarray([model._score(state) for state in states], dtype=float)
        calibrated[f"Static:seed_{seed}"] = choose_threshold(validation, state_values(validation, values), "static")
    mean_values = np.mean(np.stack([[model._score(state) for state in states] for model in static_members]), axis=0)
    calibrated["Static:ensemble"] = choose_threshold(validation, state_values(validation, mean_values), "static")
    artifact = {
        "protocol": "share-matched-e2-232-v1",
        "selection_data": "development validation split only",
        "selection_metric": "episode-level HIR alone",
        "target_hir": 0.20,
        "allowed_band": [0.18, 0.22],
        "holdout_sha256": EXPECTED_HOLDOUT,
        "encoder_sha256": sha256_file(ENCODER),
        "checkpoint_sha256": {f"{method}:seed_{seed}": sha256_file(checkpoint_path(method, seed)) for method in METHODS[:-1] for seed in SEEDS},
        "static_model_sha256": {f"seed_{seed}": sha256_file(OUT / "static_models" / f"static_seed_{seed}.joblib") for seed in SEEDS},
        "thresholds": calibrated,
        "holdout_outcomes_read": False,
    }
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"calibrated_policies": len(calibrated), "validation_episodes": len(validation), "path": str(path.relative_to(ROOT))}))


def load_policies(encoder: FeatureEncoder, calibration: dict) -> dict[str, list[Policy]]:
    neural = neural_models(encoder)
    result: dict[str, list[Policy]] = {}
    for method, members in neural.items():
        for seed, member in zip(SEEDS, members):
            member.name = f"{method}_seed_{seed}"
            member.human_calibration_penalty = calibration["thresholds"][f"{method}:seed_{seed}"]["threshold"]
        ensemble = DQNEnsemblePolicy(members, encoder, members[0].config, members[0].reward_config)
        ensemble.name = f"{method}_ensemble"
        ensemble.human_calibration_penalty = calibration["thresholds"][f"{method}:ensemble"]["threshold"]
        result[method] = [*members, ensemble]
    static_members = [joblib.load(OUT / "static_models" / f"static_seed_{seed}.joblib") for seed in SEEDS]
    for seed, member in zip(SEEDS, static_members):
        member.name = f"Static_seed_{seed}"
        member.threshold = calibration["thresholds"][f"Static:seed_{seed}"]["threshold"]
    static_ensemble = StaticMeanPolicy(static_members)
    static_ensemble.threshold = calibration["thresholds"]["Static:ensemble"]["threshold"]
    result["Static"] = [*static_members, static_ensemble]
    return result


def compact(raw: dict) -> dict:
    return {key: raw[key] for key in METRICS}


def record_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    success = np.asarray([bool(row["success"]) for row in records])
    human = np.asarray([bool(row["human_used"]) for row in records])
    stop = np.asarray([int(row["actions"][-1]) == int(AuthorityAction.STOP) for row in records])
    autonomous = ~human & ~stop
    adverse = np.asarray([bool(row["unsafe"]) for row in records]) | ~success
    return {"repair_sr": success.astype(float), "stopped_rate": stop.astype(float), "autonomous": autonomous.astype(float), "autonomous_adverse": (autonomous & adverse).astype(float)}


def paired_contrasts(ensembles: dict[str, list[dict]]) -> list[dict]:
    ids = [row["episode_id"] for row in ensembles["Full SHARE"]]
    if any([row["episode_id"] for row in rows] != ids for rows in ensembles.values()):
        raise ValueError("ensemble records are not workflow paired")
    arrays = {method: record_arrays(rows) for method, rows in ensembles.items()}
    rng = np.random.default_rng(2026093004)
    draws = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))
    rows = []
    for other in METHODS[1:]:
        for metric in CONTRAST_METRICS:
            def metric_samples(parts):
                if metric == "uar":
                    numer = parts["autonomous_adverse"][draws].sum(axis=1)
                    denom = parts["autonomous"][draws].sum(axis=1)
                    return np.divide(numer, denom, out=np.full(len(denom), np.nan), where=denom > 0)
                return parts[metric][draws].mean(axis=1)
            left, right = metric_samples(arrays["Full SHARE"]), metric_samples(arrays[other])
            diff = left - right
            valid = diff[np.isfinite(diff)]
            if len(valid) != BOOTSTRAP_DRAWS:
                raise ValueError("undefined UAR in bootstrap draw")
            p = min(1.0, 2.0 * (min(np.count_nonzero(valid <= 0), np.count_nonzero(valid >= 0)) + 1) / (BOOTSTRAP_DRAWS + 1))
            def observed(parts):
                if metric == "uar":
                    return parts["autonomous_adverse"].sum() / parts["autonomous"].sum()
                return parts[metric].mean()
            delta = float(observed(arrays["Full SHARE"]) - observed(arrays[other]))
            rows.append({"comparison": f"Full SHARE - {other}", "metric": metric, "delta": delta, "ci95": [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))], "p_raw": float(p), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "workflow_group_fixed_trajectories"})
    previous = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda x: x["p_raw"])):
        previous = max(previous, min(1.0, (len(rows) - rank) * row["p_raw"]))
        row["p_holm_12"] = previous
    return rows


def tex_table(single: dict[str, list[dict]], ensemble: dict[str, dict]) -> str:
    def cells(values):
        return " & ".join(values) + r"\\"
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Component comparison on 232 new group-disjoint public-API replay workflows at the 20\% Human cap. Neural policies use a locally frozen matched-training protocol; single-model cells are mean $\pm$ SD over five seeds and ensemble cells are one five-member endpoint. Outcomes are deterministic simulations. Twelve workflow-paired ensemble contrasts with Holm correction are reported separately.}",
        r"\label{tab:component-confirmation-232}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        cells(["Method", "Repair.SR", "HIR", "UAR", "Stop", "Return", "Regret"]),
        r"\midrule", r"\multicolumn{7}{l}{\emph{Five individually evaluated models}}\\",
    ]
    for method in METHODS:
        rows = single[method]
        values = [f"${mean(row[metric] for row in rows):.3f}\\pm{stdev(row[metric] for row in rows):.3f}$" for metric in METRICS]
        lines.append(cells([method, *values]))
    lines.extend([r"\midrule", r"\multicolumn{7}{l}{\emph{One five-member ensemble per method}}\\"])
    for method in METHODS:
        values = [f"${ensemble[method][metric]:.3f}$" for metric in METRICS]
        lines.append(cells([method, *values]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def tex_contrast_table(rows: list[dict]) -> str:
    labels = {"repair_sr": "Repair.SR", "uar": "UAR", "stopped_rate": "Stop"}
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Workflow-paired ensemble contrasts on the 232-group component holdout. Differences are Full SHARE minus the comparator. Intervals use 20,000 paired resamples of the frozen trajectories under the original online quota order; the twelve two-sided tests form one Holm family. These intervals describe conditional outcome uncertainty, not quota-order or training-seed variation.}",
        r"\label{tab:component-confirmation-contrasts-232}", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Comparator & Metric & Difference & Paired 95\% CI & Holm $p$\\",
        r"\midrule",
    ]
    for row in rows:
        other = row["comparison"].removeprefix("Full SHARE - ")
        lo, hi = row["ci95"]
        lines.append(f"{other} & {labels[row['metric']]} & ${row['delta']:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & ${row['p_holm_12']:.4f}$" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def evaluate() -> None:
    check_inputs()
    calibration_path = OUT / "calibration.json"
    if not calibration_path.is_file():
        raise SystemExit("calibration must be frozen before holdout evaluation")
    calibration = json.loads(calibration_path.read_text())
    for method in METHODS[:-1]:
        for seed in SEEDS:
            key = f"{method}:seed_{seed}"
            if sha256_file(checkpoint_path(method, seed)) != calibration["checkpoint_sha256"][key]:
                raise SystemExit(f"checkpoint changed since calibration: {key}")
    for seed in SEEDS:
        static_path = OUT / "static_models" / f"static_seed_{seed}.joblib"
        if sha256_file(static_path) != calibration["static_model_sha256"][f"seed_{seed}"]:
            raise SystemExit(f"Static model changed since calibration: {seed}")
    result_path = OUT / "FINAL_RESULTS.json"
    workflow_path = OUT / "per_workflow_records.jsonl"
    if result_path.exists() or workflow_path.exists():
        raise SystemExit("refusing to overwrite existing results")
    holdout = load_episodes(HOLDOUT)
    if len(holdout) != 232 or len({ep.group_id for ep in holdout}) != 232:
        raise SystemExit("unexpected holdout size or group duplication")
    encoder = FeatureEncoder.load(ENCODER)
    policies = load_policies(encoder, calibration)
    reward = RewardConfig()
    single = {}
    ensembles = {}
    ensemble_records = {}
    workflow_records = []
    for method in METHODS:
        singles = []
        for index, policy in enumerate(policies[method]):
            wrapper = OnlineHardHumanCapPolicy(policy, CAP, method == "Static")
            raw = evaluate_policy(wrapper, holdout, reward, CAP / len(holdout), None)
            if wrapper.human_interventions > CAP:
                raise RuntimeError("Human cap violated")
            row = compact(raw)
            row.update({"method": method, "members": 1 if index < 5 else 5, "seed": SEEDS[index] if index < 5 else None, "human_interventions": wrapper.human_interventions, "decision_states": raw["decision_states"]})
            workflow_records.extend({"method": method, "members": row["members"], "seed": row["seed"], **record} for record in raw["records"])
            if index < 5:
                singles.append(row)
            else:
                ensembles[method] = row
                ensemble_records[method] = raw["records"]
        single[method] = singles
    if len(workflow_records) != len(METHODS) * 6 * len(holdout):
        raise RuntimeError("incomplete per-workflow result record set")
    contrasts = paired_contrasts(ensemble_records)
    result = {
        "protocol": "share-matched-e2-232-v1",
        "evidence_class": "new group-disjoint deterministic replay; 232 workflows",
        "holdout_sha256": EXPECTED_HOLDOUT,
        "calibration_sha256": sha256_file(calibration_path),
        "single_model_rows": single,
        "five_member_ensembles": ensembles,
        "paired_ensemble_contrasts_holm_12": contrasts,
    }
    workflow_path.write_text("".join(json.dumps(row) + "\n" for row in workflow_records), encoding="utf-8")
    result["per_workflow_records_sha256"] = sha256_file(workflow_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "component_confirmation_table.tex").write_text(tex_table(single, ensembles), encoding="utf-8")
    (OUT / "component_confirmation_contrasts.tex").write_text(tex_contrast_table(contrasts), encoding="utf-8")
    print(json.dumps({"result": str(result_path.relative_to(ROOT)), "records": str(workflow_path.relative_to(ROOT)), "table": str((OUT / 'component_confirmation_table.tex').relative_to(ROOT)), "contrasts": str((OUT / 'component_confirmation_contrasts.tex').relative_to(ROOT)), "ensembles": ensembles}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    args = parser.parse_args()
    calibrate() if args.phase == "calibrate" else evaluate()
