from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.proposal_annotation import StudyStore, canonical_json, ensure_private_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export completed blinded SHARE proposal assessments for offline analysis.")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh owner-only JSONL export path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    store = StudyStore(args.study)
    rows = list(store.export_completed_rows())
    ensure_private_directory(args.out.parent)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    print(json.dumps({"out": str(args.out), "completed_assessments": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
