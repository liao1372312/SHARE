from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import create_simulated_assessment_casebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a private blinded casebook for LLM-simulated SHARE proposal assessment. "
            "It creates no human accounts, assignments, or human-study records."
        )
    )
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only SQLite casebook path.")
    parser.add_argument("--study-id", required=True, help="Stable identifier bound into blinded case ordering.")
    parser.add_argument("--max-workflows", type=int, default=3000, help="Number of workflow cases, balanced by document group.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.max_workflows < 1:
        parser.error("--max-workflows must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"casebook database already exists: {args.out}")
    try:
        report = create_simulated_assessment_casebook(
            collection_dir=args.collection_dir,
            database_path=args.out,
            study_id=args.study_id,
            max_workflows=args.max_workflows,
            seed=args.seed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
