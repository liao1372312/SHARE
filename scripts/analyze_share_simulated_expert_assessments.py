from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import StudyStore, canonical_json, ensure_private_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LLM-simulated SHARE assessments without presenting them as human data."
    )
    parser.add_argument("--input", type=Path, required=True, help="Owner-only simulated_assessments.jsonl file.")
    parser.add_argument("--study", type=Path, required=True, help="Private study database used only to resolve blind labels after simulation.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only JSON output path.")
    return parser.parse_args()


def load_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError("--input must be one existing simulated-assessment JSONL file")
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"input line {line_number} is not JSON") from error
        if not isinstance(row, Mapping) or row.get("kind") != "share_llm_simulated_expert_assessment":
            raise ValueError(f"input line {line_number} is not a simulated assessment")
        rows.append(row)
    if not rows:
        raise ValueError("simulated-assessment input is empty")
    return rows


def resolved_preference(assessment: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    """Resolve a blinded preference to its proposal model or a special response."""
    preference = assessment.get("preferred_proposal")
    if not isinstance(preference, str):
        return None
    blind_label = preference.upper()
    binding = mapping.get(blind_label)
    if isinstance(binding, Mapping) and isinstance(binding.get("model"), str):
        return binding["model"]
    if preference in {"tie", "none"}:
        return preference
    return None


def descriptive_persona_repeatability(
    rows: list[Mapping[str, Any]], mappings: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize same-model, two-slot repeatability without implying human IRR."""
    by_case: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        case_id = row.get("case_id")
        reviewer_slot = row.get("reviewer_slot")
        assessment = row.get("assessment")
        if isinstance(case_id, str) and isinstance(reviewer_slot, int) and isinstance(assessment, Mapping):
            by_case[case_id][reviewer_slot] = assessment

    paired = [
        (case_id, slots[1], slots[2])
        for case_id, slots in by_case.items()
        if case_id in mappings and 1 in slots and 2 in slots
    ]
    preference_pairs: list[tuple[str, str]] = []
    engineer_pairs: list[tuple[str, str]] = []
    confidence_pairs: list[tuple[int, int]] = []
    verdict_pairs: list[tuple[str, str]] = []
    metric_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    unanimous_preferences: Counter[str] = Counter()
    split_preference_cases = 0

    for case_id, first, second in paired:
        mapping = mappings[case_id]
        first_preference = resolved_preference(first, mapping)
        second_preference = resolved_preference(second, mapping)
        if first_preference is not None and second_preference is not None:
            preference_pairs.append((first_preference, second_preference))
            if first_preference == second_preference:
                unanimous_preferences[first_preference] += 1
            else:
                split_preference_cases += 1
        first_engineer = first.get("engineer_review")
        second_engineer = second.get("engineer_review")
        if isinstance(first_engineer, str) and isinstance(second_engineer, str):
            engineer_pairs.append((first_engineer, second_engineer))
        first_confidence = first.get("case_confidence")
        second_confidence = second.get("case_confidence")
        if isinstance(first_confidence, int) and isinstance(second_confidence, int):
            confidence_pairs.append((first_confidence, second_confidence))
        first_reviews = first.get("proposal_reviews")
        second_reviews = second.get("proposal_reviews")
        if not isinstance(first_reviews, Mapping) or not isinstance(second_reviews, Mapping):
            continue
        for label in ("A", "B", "C"):
            first_review = first_reviews.get(label)
            second_review = second_reviews.get(label)
            if not isinstance(first_review, Mapping) or not isinstance(second_review, Mapping):
                continue
            first_verdict = first_review.get("verdict")
            second_verdict = second_review.get("verdict")
            if isinstance(first_verdict, str) and isinstance(second_verdict, str):
                verdict_pairs.append((first_verdict, second_verdict))
            for metric in ("feasibility", "safety", "input_coverage"):
                first_value = first_review.get(metric)
                second_value = second_review.get(metric)
                if isinstance(first_value, int) and isinstance(second_value, int):
                    metric_pairs[metric].append((first_value, second_value))

    def categorical(values: list[tuple[str, str]]) -> dict[str, Any]:
        matches = sum(left == right for left, right in values)
        return {
            "paired_observations": len(values),
            "exact_match_count": matches,
            "exact_match_rate": matches / len(values) if values else None,
        }

    def ordinal(values: list[tuple[int, int]]) -> dict[str, Any]:
        matches = sum(left == right for left, right in values)
        differences = [abs(left - right) for left, right in values]
        return {
            "paired_observations": len(values),
            "exact_match_count": matches,
            "exact_match_rate": matches / len(values) if values else None,
            "mean_absolute_difference": mean(differences) if differences else None,
        }

    return {
        "warning": (
            "This is descriptive repeatability across two predefined simulated-reviewer slots queried from the same judge model. "
            "It is not human inter-rater reliability, independent-rater agreement, or inferential evidence."
        ),
        "complete_two_slot_cases": len(paired),
        "cases_missing_a_slot": len(by_case) - len(paired),
        "preference": {
            **categorical(preference_pairs),
            "unanimous_case_counts": dict(sorted(unanimous_preferences.items())),
            "split_case_count": split_preference_cases,
        },
        "engineer_review": categorical(engineer_pairs),
        "case_confidence": ordinal(confidence_pairs),
        "proposal_verdict": categorical(verdict_pairs),
        "proposal_scores": {
            metric: ordinal(metric_pairs[metric])
            for metric in ("feasibility", "safety", "input_coverage")
        },
    }


def summary(rows: list[Mapping[str, Any]], mappings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    preferred: Counter[str] = Counter()
    engineer_review: Counter[str] = Counter()
    confidence: list[float] = []
    simulation_ids = set()
    reviewer_slots = set()
    case_ids = set()
    profiles = set()
    for row in rows:
        case_id = row.get("case_id")
        assessment = row.get("assessment")
        if not isinstance(case_id, str) or not isinstance(assessment, Mapping) or case_id not in mappings:
            raise ValueError("simulated assessment does not match the supplied study")
        mapping = mappings[case_id]
        reviews = assessment.get("proposal_reviews")
        if not isinstance(reviews, Mapping):
            raise ValueError("simulated assessment lacks proposal reviews")
        for label in ("A", "B", "C"):
            review = reviews.get(label)
            model_binding = mapping.get(label)
            if not isinstance(review, Mapping) or not isinstance(model_binding, Mapping) or not isinstance(model_binding.get("model"), str):
                raise ValueError("simulated assessment has an invalid blind-label mapping")
            model = model_binding["model"]
            for metric in ("feasibility", "safety", "input_coverage"):
                value = review.get(metric)
                if isinstance(value, int):
                    scores[model][metric].append(float(value))
            verdict = review.get("verdict")
            if isinstance(verdict, str):
                verdicts[model][verdict] += 1
        preference = resolved_preference(assessment, mapping)
        if preference is not None:
            preferred[preference] += 1
        decision = assessment.get("engineer_review")
        if isinstance(decision, str):
            engineer_review[decision] += 1
        value = assessment.get("case_confidence")
        if isinstance(value, int):
            confidence.append(float(value))
        if isinstance(row.get("simulation_id"), str):
            simulation_ids.add(row["simulation_id"])
        if isinstance(row.get("reviewer_slot"), int):
            reviewer_slots.add(row["reviewer_slot"])
        profile = row.get("profile_binding")
        if isinstance(profile, Mapping) and isinstance(profile.get("model"), str):
            profiles.add(profile["model"])
        case_ids.add(case_id)
    models = {
        model: {
            "proposal_ratings": sum(verdicts[model].values()),
            "feasibility_mean": mean(scores[model]["feasibility"]) if scores[model]["feasibility"] else None,
            "safety_mean": mean(scores[model]["safety"]) if scores[model]["safety"] else None,
            "input_coverage_mean": mean(scores[model]["input_coverage"]) if scores[model]["input_coverage"] else None,
            "verdict_counts": dict(sorted(verdicts[model].items())),
            "preferred_count": preferred[model],
        }
        for model in sorted(verdicts)
    }
    return {
        "schema_version": 1,
        "kind": "share_llm_simulated_expert_assessment_summary",
        "warning": "This is an LLM-simulated expert assessment summary. It is not a human study, human inter-rater reliability estimate, practitioner evidence, service-execution outcome, paired Human/Stop outcome, or evidence for Repair.SR/HIR/UAR claims.",
        "simulation_ids": sorted(simulation_ids),
        "simulated_assessments": len(rows),
        "simulated_reviewer_slots": sorted(reviewer_slots),
        "unique_cases": len(case_ids),
        "judge_models": sorted(profiles),
        "case_confidence_mean": mean(confidence) if confidence else None,
        "engineer_review_counts": dict(sorted(engineer_review.items())),
        "preferred_counts": dict(sorted(preferred.items())),
        "models": models,
        "descriptive_persona_repeatability": descriptive_persona_repeatability(rows, mappings),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    try:
        report = summary(load_rows(args.input), StudyStore(args.study, migrate=False).case_model_mappings())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    ensure_private_directory(args.out.parent)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "simulated_assessments": report["simulated_assessments"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
