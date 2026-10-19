from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import (
    ASSESSMENT_SCOPE,
    CALIBRATION_MANIFEST_KIND,
    canonical_json,
    ensure_private_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed SHARE expert proposal assessments without treating them as executed service outcomes."
    )
    parser.add_argument("--input", type=Path, required=True, help="Owner-only JSONL from export_share_proposal_annotations.py")
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only JSON summary path.")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=None,
        help="Optional human-only calibration manifest used to report that subset separately.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError("--input must be one existing JSONL export")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"input line {number} is not JSON") from error
        if not isinstance(row, Mapping) or row.get("kind") != "share_proposal_assessment_export_row":
            raise ValueError(f"input line {number} is not an assessment export row")
        if row.get("assessment_scope") != ASSESSMENT_SCOPE:
            raise ValueError(f"input line {number} has the wrong assessment scope")
        rows.append(row)
    if not rows:
        raise ValueError("assessment export is empty")
    return rows


def load_calibration_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("--calibration-manifest must be one existing JSON file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("--calibration-manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or manifest.get("kind") != CALIBRATION_MANIFEST_KIND:
        raise ValueError("--calibration-manifest has the wrong kind")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("--calibration-manifest has no cases")
    case_ids = []
    for item in cases:
        if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str) or not item["case_id"]:
            raise ValueError("--calibration-manifest has an invalid case ID")
        case_ids.append(item["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("--calibration-manifest has duplicate case IDs")
    return {
        "study_id": manifest.get("study_id"),
        "collection_id": manifest.get("collection_id"),
        "case_ids": set(case_ids),
    }


def pairwise_kappa(values: Mapping[str, list[str]]) -> dict[str, Any]:
    pairs = []
    for ratings in values.values():
        for left_index in range(len(ratings)):
            for right_index in range(left_index + 1, len(ratings)):
                pairs.append((ratings[left_index], ratings[right_index]))
    if not pairs:
        return {"pairs": 0, "exact_agreement": None, "cohen_kappa": None}
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _right in pairs)
    right_counts = Counter(right for _left, right in pairs)
    expected = sum((left_counts[key] / len(pairs)) * (right_counts[key] / len(pairs)) for key in set(left_counts) | set(right_counts))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"pairs": len(pairs), "exact_agreement": observed, "cohen_kappa": kappa}


def summary(rows: Iterable[Mapping[str, Any]], calibration_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    preferred: Counter[str] = Counter()
    engineer_review: Counter[str] = Counter()
    confidence = []
    per_case_preference: dict[str, list[str]] = defaultdict(list)
    per_case_model_verdict: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    annotators = set()
    study_ids = set()
    collection_ids = set()
    annotation_modes: Counter[str] = Counter()
    human_only_preference: dict[str, list[str]] = defaultdict(list)
    human_only_model_verdict: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    calibration_preference: dict[str, list[str]] = defaultdict(list)
    calibration_model_verdict: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    calibration_case_ids = set(calibration_manifest["case_ids"]) if calibration_manifest is not None else None
    human_only_count = 0
    calibration_human_only_count = 0
    count = 0
    for row in rows:
        response = row.get("response")
        mapping = row.get("model_by_blind_label")
        case = row.get("case")
        if not isinstance(response, Mapping) or not isinstance(mapping, Mapping) or not isinstance(case, Mapping):
            raise ValueError("assessment export row lacks response, case, or model mapping")
        case_id = case.get("case_id")
        reviews = response.get("proposal_reviews")
        if not isinstance(case_id, str) or not isinstance(reviews, Mapping):
            raise ValueError("assessment export row has invalid case or proposal reviews")
        annotator = row.get("annotator_id")
        if isinstance(annotator, str):
            annotators.add(annotator)
        if isinstance(row.get("study_id"), str):
            study_ids.add(row["study_id"])
        if isinstance(row.get("collection_id"), str):
            collection_ids.add(row["collection_id"])
        assisted = isinstance(response.get("llm_assisted_expert_audit"), Mapping)
        annotation_modes["llm_assisted_expert_audit" if assisted else "human_only"] += 1
        if not assisted:
            human_only_count += 1
        in_calibration = calibration_case_ids is not None and case_id in calibration_case_ids
        for label, review in reviews.items():
            model_binding = mapping.get(label)
            if not isinstance(model_binding, Mapping) or not isinstance(model_binding.get("model"), str) or not isinstance(review, Mapping):
                raise ValueError("assessment export row has an invalid blinded model mapping")
            model = model_binding["model"]
            for metric in ("feasibility", "safety", "input_coverage"):
                value = review.get(metric)
                if isinstance(value, int):
                    model_scores[model][metric].append(float(value))
            verdict = review.get("verdict")
            if isinstance(verdict, str):
                model_verdicts[model][verdict] += 1
                per_case_model_verdict[case_id][model].append(verdict)
                if not assisted:
                    human_only_model_verdict[case_id][model].append(verdict)
                    if in_calibration:
                        calibration_model_verdict[case_id][model].append(verdict)
        preference = response.get("preferred_proposal")
        if isinstance(preference, str):
            blind_label = preference.upper()
            if blind_label in mapping and isinstance(mapping[blind_label], Mapping):
                selected_model = mapping[blind_label].get("model")
                if isinstance(selected_model, str):
                    preferred[selected_model] += 1
                    per_case_preference[case_id].append(selected_model)
                    if not assisted:
                        human_only_preference[case_id].append(selected_model)
                        if in_calibration:
                            calibration_preference[case_id].append(selected_model)
            elif preference in {"tie", "none"}:
                preferred[preference] += 1
                per_case_preference[case_id].append(preference)
                if not assisted:
                    human_only_preference[case_id].append(preference)
                    if in_calibration:
                        calibration_preference[case_id].append(preference)
        review_decision = response.get("engineer_review")
        if isinstance(review_decision, str):
            engineer_review[review_decision] += 1
        case_confidence = response.get("case_confidence")
        if isinstance(case_confidence, int):
            confidence.append(float(case_confidence))
        if not assisted and in_calibration:
            calibration_human_only_count += 1
        count += 1
    models = {
        model: {
            "proposal_ratings": sum(model_verdicts[model].values()),
            "feasibility_mean": mean(model_scores[model]["feasibility"]) if model_scores[model]["feasibility"] else None,
            "safety_mean": mean(model_scores[model]["safety"]) if model_scores[model]["safety"] else None,
            "input_coverage_mean": mean(model_scores[model]["input_coverage"]) if model_scores[model]["input_coverage"] else None,
            "verdict_counts": dict(sorted(model_verdicts[model].items())),
            "preferred_count": preferred[model],
        }
        for model in sorted(model_verdicts)
    }
    verdict_agreement = {
        model: pairwise_kappa({case_id: by_model.get(model, []) for case_id, by_model in per_case_model_verdict.items()})
        for model in sorted(models)
    }
    agreement = {
        "all_completed_assessments_descriptive_only": {
            "preferred_choice": pairwise_kappa(per_case_preference),
            "verdict_by_model": verdict_agreement,
        },
        "all_human_only_completed_assessments": {
            "completed_assessments": human_only_count,
            "preferred_choice": pairwise_kappa(human_only_preference),
            "verdict_by_model": {
                model: pairwise_kappa({case_id: by_model.get(model, []) for case_id, by_model in human_only_model_verdict.items()})
                for model in sorted(models)
            },
        },
    }
    if calibration_case_ids is not None:
        agreement["manifested_human_only_calibration"] = {
            "manifest_cases": len(calibration_case_ids),
            "cases_with_completed_assessments": len(calibration_preference),
            "human_only_completed_assessments": calibration_human_only_count,
            "preferred_choice": pairwise_kappa(calibration_preference),
            "verdict_by_model": {
                model: pairwise_kappa({case_id: by_model.get(model, []) for case_id, by_model in calibration_model_verdict.items()})
                for model in sorted(models)
            },
        }
    return {
        "schema_version": 1,
        "kind": "share_proposal_assessment_summary",
        "assessment_scope": ASSESSMENT_SCOPE,
        "warning": "These results describe expert judgments of anonymous LLM proposal quality. They are not service-execution outcomes, paired Human/Stop outcomes, Repair.SR, HIR, UAR, or evidence for those paper claims. When LLM-assisted expert audits are present, their agreement is descriptive only and must not be presented as independent human inter-rater reliability.",
        "study_ids": sorted(study_ids),
        "collection_ids": sorted(collection_ids),
        "completed_assessments": count,
        "annotation_mode_counts": dict(sorted(annotation_modes.items())),
        "annotators": len(annotators),
        "unique_cases": len(per_case_preference),
        "case_confidence_mean": mean(confidence) if confidence else None,
        "engineer_review_counts": dict(sorted(engineer_review.items())),
        "preferred_counts": dict(sorted(preferred.items())),
        "models": models,
        "interrater_agreement": agreement,
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    rows = load_rows(args.input)
    calibration_manifest = load_calibration_manifest(args.calibration_manifest) if args.calibration_manifest is not None else None
    if calibration_manifest is not None:
        row_study_ids = {row.get("study_id") for row in rows}
        row_collection_ids = {row.get("collection_id") for row in rows}
        if row_study_ids != {calibration_manifest["study_id"]} or row_collection_ids != {calibration_manifest["collection_id"]}:
            raise SystemExit("--calibration-manifest does not match the exported study")
    report = summary(rows, calibration_manifest)
    ensure_private_directory(args.out.parent)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "completed_assessments": report["completed_assessments"]}, indent=2))


if __name__ == "__main__":
    main()
