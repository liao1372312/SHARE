from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import backup_study_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a consistent owner-only backup of a running SHARE proposal-assessment study.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh SQLite backup path.")
    args = parser.parse_args()
    try:
        backup_study_database(args.study, args.out)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"out": str(args.out), "source": str(args.study)}, indent=2))


if __name__ == "__main__":
    main()
