"""Report availability of updated paper inputs; never synthesizes missing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path):
    required = [f"datasets/paper/{role}/{name}" for role in ("development", "primary", "mechanism", "retrospective") for name in ("episodes.jsonl", "import_report.json")]
    required.append("artifacts/paper/artifacts.json")
    rows = [{"path": name, "present": (root / name).is_file()} for name in required]
    available = all(r["present"] for r in rows)
    manifest_ready = False
    manifest = root / "artifacts/paper/artifacts.json"
    if manifest.is_file():
        try:
            manifest_ready = json.loads(manifest.read_text(encoding="utf-8")).get("status") == "ready"
        except (ValueError, OSError):
            pass
    return {"status": "inputs_present_validation_required" if available and manifest_ready else "pending_inputs", "files": rows,
            "note": "Availability is not evidence validation. Run evaluate_share_paper.py after completing the checklist."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "inputs_present_validation_required" else 2


if __name__ == "__main__":
    sys.exit(main())
