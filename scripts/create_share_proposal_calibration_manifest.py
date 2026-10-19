from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import (
    CALIBRATION_MANIFEST_KIND,
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    StudyStore,
    sha256_text,
    utc_now,
    write_private_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reserve a double-rated human-only calibration set before showing LLM defaults to SHARE proposal experts."
    )
    parser.add_argument("--study", type=Path, required=True, help="Private SQLite study database.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only calibration manifest JSON.")
    parser.add_argument("--calibration-cases", type=int, default=12, help="Number of cases reserved for human-only calibration.")
    parser.add_argument("--human-ratings-per-case", type=int, default=2, help="Independent human-only ratings required before LLM prefilling.")
    parser.add_argument("--seed", type=int, default=17, help="Stable tie-breaker when cases have equal completion progress.")
    args = parser.parse_args()
    if args.calibration_cases < 1:
        parser.error("--calibration-cases must be positive")
    if args.human_ratings_per_case < 2:
        parser.error("--human-ratings-per-case must be at least two")
    return args


def select_cases(store: StudyStore, *, count: int, seed: int) -> list[Mapping[str, Any]]:
    cases = store.cases_for_preannotation()
    if count > len(cases):
        raise ValueError("--calibration-cases exceeds the study case count")
    case_ids = [item["case"]["case_id"] for item in cases]
    progress = store.human_only_calibration_progress(case_ids)
    # Preserve work that experts have already completed while keeping a stable order.
    ranked = sorted(
        cases,
        key=lambda item: (
            -progress[item["case"]["case_id"]]["human_only_ratings"],
            sha256_text(f"{seed}:{item['case']['case_id']}"),
        ),
    )
    return ranked[:count]


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    try:
        store = StudyStore(args.study)
        selected = select_cases(store, count=args.calibration_cases, seed=args.seed)
        metadata = store.metadata()
        manifest = {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "kind": CALIBRATION_MANIFEST_KIND,
            "created_at_utc": utc_now(),
            "study_id": metadata.get("study_id"),
            "collection_id": metadata.get("collection_id"),
            "required_human_only_ratings_per_case": args.human_ratings_per_case,
            "cases": [
                {
                    "case_id": item["case"]["case_id"],
                    "group_id": item["case"]["group_id"],
                    "case_sha256": item["case_sha256"],
                }
                for item in selected
            ],
            "selection": {
                "strategy": "prioritize_existing_human_only_completions_then_stable_seeded_order",
                "seed": args.seed,
            },
            "warning": (
                "These cases must receive the required number of independent human-only ratings before any LLM default is shown. "
                "They remain excluded from LLM prefilling so they can serve as an unanchored calibration baseline."
            ),
        }
        write_manifest(args.out, manifest)
        progress = store.human_only_calibration_progress([item["case_id"] for item in manifest["cases"]])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    completed = sum(item["human_only_ratings"] for item in progress.values())
    required = args.calibration_cases * args.human_ratings_per_case
    print(
        json.dumps(
            {
                "out": str(args.out),
                "calibration_cases": args.calibration_cases,
                "required_human_only_ratings": required,
                "completed_human_only_ratings": completed,
                "remaining_human_only_ratings": max(0, required - completed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
