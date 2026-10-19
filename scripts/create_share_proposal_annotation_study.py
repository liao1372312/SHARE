from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import create_study, write_credentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a blinded, local expert-assessment study from a completed three-profile SHARE proposal collection."
    )
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh SQLite study database path.")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--annotator", action="append", required=True, help="Lowercase annotator ID. Repeat for every expert.")
    parser.add_argument("--admin-id", default="coordinator")
    parser.add_argument("--max-workflows", type=int, default=300, help="Number of workflow cases to sample, balanced by document group.")
    parser.add_argument("--ratings-per-workflow", type=int, default=2, help="Independent expert ratings to assign per case.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--credentials-out", type=Path, default=None, help="Fresh owner-only credential file. Defaults beside --out.")
    args = parser.parse_args()
    if args.max_workflows < 1:
        parser.error("--max-workflows must be positive")
    if args.ratings_per_workflow < 1:
        parser.error("--ratings-per-workflow must be positive")
    if args.credentials_out is None:
        args.credentials_out = args.out.with_suffix(args.out.suffix + ".credentials.txt")
    if args.out.resolve() == args.credentials_out.resolve():
        parser.error("--out and --credentials-out must differ")
    return args


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"study database already exists: {args.out}")
    if args.credentials_out.exists():
        raise SystemExit(f"credentials file already exists: {args.credentials_out}")
    summary = create_study(
        collection_dir=args.collection_dir,
        database_path=args.out,
        study_id=args.study_id,
        annotator_ids=args.annotator,
        admin_id=args.admin_id,
        max_workflows=args.max_workflows,
        ratings_per_workflow=args.ratings_per_workflow,
        seed=args.seed,
    )
    write_credentials(args.credentials_out, summary.pop("credentials"), args.admin_id.strip().lower())
    summary["credentials_file"] = str(args.credentials_out)
    summary["warning"] = (
        "Credentials are owner-only. Distribute each annotator password privately. The study measures anonymous proposal quality, "
        "not service execution or paired authority outcomes."
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
