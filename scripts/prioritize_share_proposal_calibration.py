from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import StudyStore, load_human_only_calibration_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prioritize pending human-only calibration assignments before LLM-assisted SHARE proposal assessments."
    )
    parser.add_argument("--study", type=Path, required=True, help="Private SQLite study database.")
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        store = StudyStore(args.study)
        calibration = load_human_only_calibration_manifest(args.calibration_manifest, store)
        prioritized = store.prioritize_human_only_calibration(
            calibration["case_ids"],
            required_human_only_ratings=calibration["required_human_only_ratings_per_case"],
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "study": str(args.study),
                "calibration_cases": len(calibration["case_ids"]),
                "pending_assignments_prioritized": sum(len(case_ids) for case_ids in prioritized.values()),
                "assignments_by_annotator": prioritized,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
