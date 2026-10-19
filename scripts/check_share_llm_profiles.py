from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.llm_profiles import public_profile_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the three local SHARE LLM profiles without contacting any endpoint.")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, default=Path("results/share_llm_profile_preflight.json"))
    parser.add_argument("--allow-insecure-http", action="store_true", help="Allow HTTP only for a trusted local proxy; never use this on an untrusted network.")
    args = parser.parse_args()
    try:
        report = public_profile_report(args.env, allow_insecure_http=args.allow_insecure_http)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "profiles": len(report["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
