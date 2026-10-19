from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import add_annotator, write_private_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add one human SHARE proposal-assessment account without changing existing cases or submitted responses."
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--annotator", required=True, help="New lowercase expert user ID.")
    parser.add_argument(
        "--assign-all-cases",
        action="store_true",
        help="Add one pending human assessment for every existing blinded case.",
    )
    parser.add_argument(
        "--credentials-out",
        type=Path,
        required=True,
        help="Fresh owner-only file containing this expert's generated password.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.credentials_out.exists():
        raise SystemExit(f"credentials file already exists: {args.credentials_out}")
    try:
        created = add_annotator(
            args.study,
            annotator_id=args.annotator,
            assign_all_cases=args.assign_all_cases,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    password = created.pop("password")
    write_private_text(
        args.credentials_out,
        "# Store privately. Give this password only to the named expert.\n"
        "user_id\trole\tone_time_password\n"
        f"{created['annotator_id']}\tannotator\t{password}\n",
    )
    created["credentials_file"] = str(args.credentials_out)
    created["warning"] = (
        "Existing cases, prior assignments, submitted responses, and LLM-only simulations are unchanged. "
        "New human responses are proposal-quality assessments only, not service-execution outcomes."
    )
    print(json.dumps(created, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
