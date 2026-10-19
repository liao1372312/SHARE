from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


METRICS = ("feasibility", "safety", "input_coverage")


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or row.get("kind") != "share_proposal_assessment_export_row":
            raise ValueError(f"invalid export row {number}")
        rows.append(row)
    if not rows:
        raise ValueError("empty annotation export")
    return rows


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def safe_wilcoxon(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = left - right
    if np.allclose(difference, 0.0):
        return 0.0, 1.0
    result = wilcoxon(left, right, alternative="two-sided", zero_method="pratt")
    return float(result.statistic), float(result.pvalue)


def cluster_bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    difference = left - right
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    draws = difference[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Case-clustered inference for completed SHARE proposal assessments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.out.exists() or args.table.exists():
        raise SystemExit("output already exists")
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap samples must be positive")

    rows = load_rows(args.input)
    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    preferences: dict[str, Counter[str]] = defaultdict(Counter)
    annotator_preferences: dict[str, Counter[str]] = defaultdict(Counter)
    verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    models = set()
    for row in rows:
        case = row["case"]["case_id"]
        annotator = row["annotator_id"]
        mapping = row["model_by_blind_label"]
        response = row["response"]
        for label, review in response["proposal_reviews"].items():
            model = mapping[label]["model"]
            models.add(model)
            for metric in METRICS:
                scores[case][model][metric].append(float(review[metric]))
            verdicts[model][review["verdict"]] += 1
        preferred = response["preferred_proposal"]
        if preferred.upper() in mapping:
            model = mapping[preferred.upper()]["model"]
        else:
            model = preferred
        preferences[case][model] += 1
        annotator_preferences[annotator][model] += 1

    ordered_models = sorted(models)
    cases = sorted(scores)
    if len(ordered_models) != 3 or any(set(scores[case]) != set(ordered_models) for case in cases):
        raise ValueError("every case must contain all three models")
    case_counts = {case: sum(preferences[case].values()) for case in cases}
    if len(set(case_counts.values())) != 1:
        raise ValueError("cases do not have a common annotation count")

    table_rows = []
    total_preferences = sum(sum(value.values()) for value in preferences.values())
    for model in ordered_models:
        flattened = {
            metric: [value for case in cases for value in scores[case][model][metric]]
            for metric in METRICS
        }
        preferred_count = sum(value[model] for value in preferences.values())
        table_rows.append({
            "model": model,
            "proposal_ratings": len(flattened[METRICS[0]]),
            **{f"{metric}_mean": float(np.mean(flattened[metric])) for metric in METRICS},
            **{f"{metric}_sd": float(np.std(flattened[metric], ddof=1)) for metric in METRICS},
            "preferred_count": preferred_count,
            "preferred_share": preferred_count / total_preferences,
            "accept_count": verdicts[model]["accept"],
            "accept_share": verdicts[model]["accept"] / len(flattened[METRICS[0]]),
        })

    analyses: dict[str, Any] = {}
    for metric in (*METRICS, "preferred_share"):
        arrays = {}
        for model in ordered_models:
            if metric == "preferred_share":
                arrays[model] = np.asarray([preferences[case][model] / case_counts[case] for case in cases])
            else:
                arrays[model] = np.asarray([np.mean(scores[case][model][metric]) for case in cases])
        friedman = friedmanchisquare(*(arrays[model] for model in ordered_models))
        pairs = []
        raw_pvalues = []
        for pair_index, (left, right) in enumerate(combinations(ordered_models, 2)):
            statistic, pvalue = safe_wilcoxon(arrays[left], arrays[right])
            ci = cluster_bootstrap_difference(
                arrays[left], arrays[right], samples=args.bootstrap_samples, seed=args.seed + pair_index
            )
            pairs.append({
                "model_a": left,
                "model_b": right,
                "mean_difference_a_minus_b": float(np.mean(arrays[left] - arrays[right])),
                "cluster_bootstrap_95_ci": list(ci),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": pvalue,
            })
            raw_pvalues.append(pvalue)
        for pair, adjusted in zip(pairs, holm_adjust(raw_pvalues)):
            pair["wilcoxon_p_holm"] = adjusted
        analyses[metric] = {
            "unit": "case mean across three human annotators",
            "cases": len(cases),
            "friedman_statistic": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "pairwise": pairs,
        }

    report = {
        "schema_version": 1,
        "kind": "share_proposal_assessment_case_clustered_inference",
        "warning": "Anonymous proposal-quality judgments only; not executed repair, paired authority, HIR, UAR, or practitioner-outcome evidence.",
        "input": str(args.input),
        "assessments": len(rows),
        "cases": len(cases),
        "ratings_per_case": next(iter(case_counts.values())),
        "models": ordered_models,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "raw_table": table_rows,
        "annotator_preference_counts": {
            annotator: dict(sorted(counts.items())) for annotator, counts in sorted(annotator_preferences.items())
        },
        "analyses": analyses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({"out": str(args.out), "table": str(args.table), "assessments": len(rows), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
