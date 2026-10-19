"""Reproducible post-confirmation analyses for the SHARE TOSEM manuscript.

The script never fits, selects, or recalibrates a model.  It evaluates frozen
artifacts on the already opened V4/V5 replay holdouts and labels every output
as post-hoc/post-confirmation evidence.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, ENCODER, SEEDS
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy, load_frozen_policies
from scripts.run_share_voh_auxiliary_development import pairwise_rows
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy, deterministic_action
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import AuthorityAction, RepairEpisode, RepairState, RewardConfig
from share.voh import expected_voh


V4 = ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl"
V5 = ROOT / "datasets/share_v5_expected_voh_holdout_20260829/episodes.jsonl"
COMPARATOR_ARTIFACTS = ROOT / "results/share_budget_matched_frozen_artifacts_20260825"
COMPARATOR_CHECKPOINTS = ROOT / "results/share_controller_rescue_20pct_5seed_20260825"
COMPARATOR_CALIBRATION = ROOT / "results/share_budget_matched_20pct_5seed_20260825"
AUXILIARY_HEAD = ROOT / "results/share_expected_voh_filter_development_20260829/expected_voh_filter.joblib"
OUT = ROOT / "results/share_acceptance_supplement_20260831"
FIGURE_OUT = ROOT / "TOSEM/TOSEM_FLARE/figures"
HUMAN_PENALTY = 1.9403412342071533
BOOTSTRAP_DRAWS = 20_000
ORDER_DRAWS = 1_000
RNG_SEED = 20260831
METRICS = ("repair_sr", "uar", "stopped_rate", "normalized_return")


def without_records(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "records"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models() -> tuple[FeatureEncoder, RewardConfig, DQNEnsemblePolicy, dict[str, list[Policy]]]:
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    members = [
        DQNPolicy.load(str(CHECKPOINTS[seed]), encoder, training_reward, require_reward_config=True)
        for seed in SEEDS
    ]
    ensemble = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
    ensemble.name = "SHARE"
    ensemble.human_calibration_penalty = HUMAN_PENALTY
    comparators: dict[str, list[Policy]] = {"Static": [], "DDQN": []}
    for seed in SEEDS:
        policies = load_frozen_policies(
            seed,
            COMPARATOR_ARTIFACTS,
            COMPARATOR_CHECKPOINTS,
            COMPARATOR_CALIBRATION,
            encoder,
            RewardConfig(),
        )
        comparators["Static"].append(next(policy for policy in policies if policy.name == "share_static"))
        comparators["DDQN"].append(next(policy for policy in policies if policy.name == "share_ddqn"))
    return encoder, training_reward, ensemble, comparators


def cluster_bootstrap_correlation(
    scores: np.ndarray,
    target: np.ndarray,
    episode_sizes: Sequence[int],
    rng: np.random.Generator,
) -> list[float]:
    offsets = np.cumsum([0, *episode_sizes])
    state_indices = [np.arange(offsets[index], offsets[index + 1]) for index in range(len(episode_sizes))]
    output: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = rng.integers(0, len(episode_sizes), len(episode_sizes))
        indices = np.concatenate([state_indices[index] for index in selected])
        value = spearmanr(scores[indices], target[indices]).statistic
        if np.isfinite(value):
            output.append(float(value))
    return output


def validity_row(
    signal: str,
    estimand: str,
    scores: np.ndarray,
    target: np.ndarray,
    episode_sizes: Sequence[int],
    rng: np.random.Generator,
) -> dict[str, Any]:
    rho = float(spearmanr(scores, target).statistic)
    bootstrap = cluster_bootstrap_correlation(scores, target, episode_sizes, rng)
    order = np.argsort(scores, kind="stable")
    quintiles = [float(np.mean(target[index])) for index in np.array_split(order, 5)]
    return {
        "signal": signal,
        "estimand": estimand,
        "rho": rho,
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "quintile_means": quintiles,
        "strictly_monotone": all(left < right for left, right in zip(quintiles, quintiles[1:])),
        "states": len(target),
        "workflow_groups": len(episode_sizes),
        "bootstrap_unit": "workflow_group",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def q_voh_analysis(
    episodes: Sequence[RepairEpisode],
    encoder: FeatureEncoder,
    ensemble: DQNEnsemblePolicy,
) -> list[dict[str, Any]]:
    states = [step.state for episode in episodes for step in episode.steps]
    raw_values = ensemble._raw_values_batch(states)
    q_voh = raw_values[:, AuthorityAction.HUMAN] - raw_values[:, AuthorityAction.AI]
    expected = expected_voh(episodes, RewardConfig())
    _, realized, _ = pairwise_rows(episodes, encoder, RewardConfig(), 0.95)
    payload = joblib.load(AUXILIARY_HEAD)
    auxiliary = np.asarray(payload["head"].predict(encoder.transform(states)), dtype=np.float64)
    sizes = [len(episode.steps) for episode in episodes]
    rng = np.random.default_rng(RNG_SEED)
    return [
        validity_row("Q-VoH", "expected", q_voh, expected, sizes, rng),
        validity_row("Q-VoH", "realized", q_voh, realized, sizes, rng),
        validity_row("Auxiliary expected-VoH head", "expected", auxiliary, expected, sizes, rng),
        validity_row("Auxiliary expected-VoH head", "realized", auxiliary, realized, sizes, rng),
    ]


def run_frozen_policies(
    episodes: Sequence[RepairEpisode],
    ensemble: DQNEnsemblePolicy,
    comparators: Mapping[str, Sequence[Policy]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cap = math.floor(0.20 * len(episodes))
    share = evaluate_policy(
        OnlineHardHumanCapPolicy(ensemble, cap, False), episodes, RewardConfig(), 0.20, None
    )
    output: dict[str, list[dict[str, Any]]] = {"Static": [], "DDQN": []}
    for method, policies in comparators.items():
        for policy in policies:
            binary = method == "Static"
            result = evaluate_policy(
                OnlineHardHumanCapPolicy(policy, cap, binary), episodes, RewardConfig(), 0.20, None
            )
            output[method].append(result)
    return share, output


@dataclass(frozen=True)
class MetricComponents:
    repair: np.ndarray
    stop: np.ndarray
    adverse: np.ndarray
    eligible: np.ndarray


def record_components(records: Sequence[Mapping[str, Any]]) -> MetricComponents:
    return MetricComponents(
        repair=np.asarray([float(record["success"]) for record in records]),
        stop=np.asarray([float(record["actions"][-1] == int(AuthorityAction.STOP)) for record in records]),
        adverse=np.asarray(
            [
                float((record["unsafe"] or not record["success"]) and not record["human_used"] and record["actions"][-1] != int(AuthorityAction.STOP))
                for record in records
            ]
        ),
        eligible=np.asarray(
            [float(not record["human_used"] and record["actions"][-1] != int(AuthorityAction.STOP)) for record in records]
        ),
    )


def average_components(items: Sequence[MetricComponents]) -> MetricComponents:
    return MetricComponents(*[np.mean(np.stack([getattr(item, field) for item in items]), axis=0) for field in ("repair", "stop", "adverse", "eligible")])


def metric_value(parts: MetricComponents, metric: str, indices: np.ndarray | None = None) -> float:
    index = slice(None) if indices is None else indices
    if metric == "repair_sr":
        return float(np.mean(parts.repair[index]))
    if metric == "stopped_rate":
        return float(np.mean(parts.stop[index]))
    if metric == "uar":
        denominator = float(np.sum(parts.eligible[index]))
        return float(np.sum(parts.adverse[index]) / denominator) if denominator else 0.0
    raise KeyError(metric)


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_bootstrap"])
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(row["p_bootstrap"])))
        rows[index]["p_holm"] = running


def paired_statistics(
    share: Mapping[str, Any],
    baselines: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    share_parts = record_components(share["records"])
    rng = np.random.default_rng(RNG_SEED + 1)
    rows: list[dict[str, Any]] = []
    n = len(share["records"])
    for method in ("Static", "DDQN"):
        # Match the manuscript's estimand: compute each frozen seed's metric
        # first and then average the five metrics.  Pooling numerators and
        # denominators would overweight DDQN seeds with more autonomous
        # completions and would not reproduce Table 8's mean UAR.
        baseline_parts = [record_components(row["records"]) for row in baselines[method]]
        for metric in ("repair_sr", "uar", "stopped_rate"):
            share_value = metric_value(share_parts, metric)
            baseline_value = mean(metric_value(parts, metric) for parts in baseline_parts)
            delta = share_value - baseline_value
            bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
            for draw in range(BOOTSTRAP_DRAWS):
                indices = rng.integers(0, n, n)
                bootstrap[draw] = metric_value(share_parts, metric, indices) - mean(
                    metric_value(parts, metric, indices) for parts in baseline_parts
                )
            tail = min(float(np.mean(bootstrap <= 0.0)), float(np.mean(bootstrap >= 0.0)))
            p_value = min(1.0, 2.0 * (tail * BOOTSTRAP_DRAWS + 1.0) / (BOOTSTRAP_DRAWS + 1.0))
            rows.append(
                {
                    "comparison": f"SHARE - {method}",
                    "metric": metric,
                    "share": share_value,
                    "baseline": baseline_value,
                    "delta": delta,
                    "ci95_low": float(np.percentile(bootstrap, 2.5)),
                    "ci95_high": float(np.percentile(bootstrap, 97.5)),
                    "p_bootstrap": p_value,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_unit": "workflow_group",
                    "baseline_estimand": "mean of five frozen training-seed artifacts",
                }
            )
    holm_adjust(rows)
    return rows


def resource_matched_hir(
    episodes: Sequence[RepairEpisode],
    ensemble: DQNEnsemblePolicy,
    static_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed, static in zip(SEEDS, static_results):
        interventions = sum(int(record["human_used"]) for record in static["records"])
        wrapper = OnlineHardHumanCapPolicy(ensemble, interventions, False)
        share = evaluate_policy(
            wrapper, episodes, RewardConfig(), interventions / len(episodes), None
        )
        row: dict[str, Any] = {
            "static_seed": seed,
            "matched_human_episodes": interventions,
            "target_hir": interventions / len(episodes),
        }
        for prefix, result in (("static", static), ("share", share)):
            for metric in ("hir", "repair_sr", "uar", "stopped_rate", "normalized_return"):
                row[f"{prefix}_{metric}"] = float(result[metric])
        rows.append(row)
    summary: dict[str, Any] = {"matches": len(rows), "rows": rows}
    for prefix in ("static", "share"):
        for metric in ("hir", "repair_sr", "uar", "stopped_rate", "normalized_return"):
            values = [row[f"{prefix}_{metric}"] for row in rows]
            summary[f"{prefix}_{metric}_mean"] = mean(values)
            summary[f"{prefix}_{metric}_std"] = stdev(values)
    for metric in ("repair_sr", "uar", "stopped_rate", "normalized_return"):
        differences = [row[f"share_{metric}"] - row[f"static_{metric}"] for row in rows]
        summary[f"delta_{metric}_mean"] = mean(differences)
        summary[f"delta_{metric}_std"] = stdev(differences)
    summary.update(
        {
            "evidence_class": "post-hoc resource-matched sensitivity",
            "matching_rule": "For each frozen Static artifact, set only the SHARE online cap to Static's realized Human-episode count; no outcome metric selects the cap.",
        }
    )
    return summary


class CachedValuePolicy(Policy):
    def __init__(self, name: str, values: Mapping[tuple[str, int], np.ndarray]):
        self.name = name
        self.values = values

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        return deterministic_action(self.values[(state.episode_id, state.step)], feasible)

    def escalation_score(self, state: RepairState) -> float:
        value = self.values[(state.episode_id, state.step)]
        return float(value[AuthorityAction.HUMAN] - value[AuthorityAction.AI])


class CachedBinaryPolicy(Policy):
    def __init__(self, name: str, actions: Mapping[tuple[str, int], AuthorityAction]):
        self.name = name
        self.actions = actions

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        action = self.actions[(state.episode_id, state.step)]
        if action in feasible:
            return action
        return AuthorityAction.AI if AuthorityAction.AI in feasible else AuthorityAction.STOP

    def escalation_score(self, state: RepairState) -> float:
        return float(self.actions[(state.episode_id, state.step)] == AuthorityAction.HUMAN)


def cached_policies(
    episodes: Sequence[RepairEpisode],
    ensemble: DQNEnsemblePolicy,
    comparators: Mapping[str, Sequence[Policy]],
) -> dict[str, list[tuple[Policy, bool]]]:
    states = [step.state for episode in episodes for step in episode.steps]
    keys = [(state.episode_id, state.step) for state in states]
    output: dict[str, list[tuple[Policy, bool]]] = {"SHARE": [], "Static": [], "DDQN": []}
    share_values = ensemble.values_batch(states)
    output["SHARE"].append((CachedValuePolicy("SHARE", dict(zip(keys, share_values))), False))
    for index, policy in enumerate(comparators["Static"]):
        actions = {
            key: policy.select(state, (AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP))
            for key, state in zip(keys, states)
        }
        output["Static"].append((CachedBinaryPolicy(f"Static-{SEEDS[index]}", actions), True))
    for index, policy in enumerate(comparators["DDQN"]):
        values = policy.values_batch(states)
        output["DDQN"].append((CachedValuePolicy(f"DDQN-{SEEDS[index]}", dict(zip(keys, values))), False))
    return output


def arrival_order_analysis(
    episodes: Sequence[RepairEpisode],
    ensemble: DQNEnsemblePolicy,
    comparators: Mapping[str, Sequence[Policy]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = cached_policies(episodes, ensemble, comparators)
    rng = np.random.default_rng(RNG_SEED + 2)
    permutations = [rng.permutation(len(episodes)) for _ in range(ORDER_DRAWS)]
    long_rows: list[dict[str, Any]] = []
    cap = math.floor(0.20 * len(episodes))
    for draw, permutation in enumerate(permutations):
        ordered = [episodes[index] for index in permutation]
        for method in ("SHARE", "Static", "DDQN"):
            seed_results = []
            for policy, binary in policies[method]:
                result = evaluate_policy(
                    OnlineHardHumanCapPolicy(policy, cap, binary), ordered, RewardConfig(), 0.20, None
                )
                seed_results.append(result)
            row: dict[str, Any] = {"permutation": draw, "method": method}
            for metric in METRICS:
                row[metric] = mean(float(result[metric]) for result in seed_results)
            row["hir"] = mean(float(result["hir"]) for result in seed_results)
            long_rows.append(row)
    summaries: list[dict[str, Any]] = []
    for method in ("SHARE", "Static", "DDQN"):
        method_rows = [row for row in long_rows if row["method"] == method]
        row = {"method": method, "permutations": ORDER_DRAWS}
        for metric in (*METRICS, "hir"):
            values = np.asarray([item[metric] for item in method_rows], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p05"] = float(np.percentile(values, 5))
            row[f"{metric}_p95"] = float(np.percentile(values, 95))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        summaries.append(row)
    return long_rows, summaries


def plot_arrival_order(
    rows: Sequence[Mapping[str, Any]],
    nominal: Mapping[str, Mapping[str, float]],
    output: Path,
) -> None:
    methods = ("Static", "DDQN", "SHARE")
    colors = {"Static": "#56B4E9", "DDQN": "#0072B2", "SHARE": "#D55E00"}
    labels = {"repair_sr": "Repair.SR", "uar": "UAR", "stopped_rate": "Stop"}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45))
    for axis, metric in zip(axes, labels):
        data = [[float(row[metric]) for row in rows if row["method"] == method] for method in methods]
        box = axis.boxplot(
            data,
            labels=methods,
            widths=0.60,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            boxprops={"linewidth": 0.8},
        )
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.78)
        axis.scatter(
            np.arange(1, len(methods) + 1),
            [nominal[method][metric] for method in methods],
            marker="D",
            s=18,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=4,
        )
        axis.set_ylabel(labels[metric])
        axis.set_ylim(bottom=0.0, top=1.0 if metric == "repair_sr" else None)
        axis.tick_params(axis="x", rotation=18)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.98, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    v4 = load_episodes(V4)
    v5 = load_episodes(V5)
    encoder, _, ensemble, comparators = load_models()

    voh = q_voh_analysis(v5, encoder, ensemble)
    write_json(OUT / "e1_q_voh.json", {"evidence_class": "post-hoc mechanism analysis", "rows": voh})
    write_csv(OUT / "e1_q_voh.csv", voh)

    share, baselines = run_frozen_policies(v4, ensemble, comparators)
    paired = paired_statistics(share, baselines)
    write_json(OUT / "e2_paired_statistics.json", {"evidence_class": "post-confirmation paired analysis", "rows": paired})
    write_csv(OUT / "e2_paired_statistics.csv", paired)

    matched = resource_matched_hir(v4, ensemble, baselines["Static"])
    write_json(OUT / "e3_resource_matched_hir.json", matched)
    write_csv(OUT / "e3_resource_matched_hir.csv", matched["rows"])

    order_rows, order_summary = arrival_order_analysis(v4, ensemble, comparators)
    write_csv(OUT / "e4_arrival_order_permutations.csv", order_rows)
    write_json(
        OUT / "e4_arrival_order_summary.json",
        {"evidence_class": "post-confirmation fixed-policy order robustness", "rows": order_summary},
    )
    write_csv(OUT / "e4_arrival_order_summary.csv", order_summary)
    nominal = {
        "SHARE": {metric: float(share[metric]) for metric in ("repair_sr", "uar", "stopped_rate")},
        **{
            method: {
                metric: mean(float(result[metric]) for result in baselines[method])
                for metric in ("repair_sr", "uar", "stopped_rate")
            }
            for method in ("Static", "DDQN")
        },
    }
    plot_arrival_order(order_rows, nominal, FIGURE_OUT / "fig_arrival_order_robustness")

    manifest = {
        "evidence_class": "post-confirmation/post-hoc; no fitting, selection, or recalibration",
        "v4": str(V4.relative_to(ROOT)),
        "v5": str(V5.relative_to(ROOT)),
        "seeds": list(SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "arrival_order_draws": ORDER_DRAWS,
        "rng_seed": RNG_SEED,
        "outputs": sorted(path.name for path in OUT.iterdir()),
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"out": str(OUT), "figure": str(FIGURE_OUT / 'fig_arrival_order_robustness.pdf')}, indent=2))


if __name__ == "__main__":
    main()
