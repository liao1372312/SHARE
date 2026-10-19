from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes


LOCKED_HOLDOUT_PROTOCOL_VERSION = "share-guard-frozen-replay-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def load_historical_identifier_index(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("--historical-identifier-index must be an existing JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit("--historical-identifier-index must contain JSON") from error
    required = {
        "schema_version", "purpose", "source_sha256", "source_episodes", "source_groups",
        "development_sha256", "development_episodes", "development_groups",
        "development_episode_ids_sha256", "development_group_ids_sha256",
        "historical_episode_ids", "historical_group_ids",
    }
    missing = sorted(required - set(value))
    if missing:
        raise SystemExit(f"--historical-identifier-index missing fields: {missing}")
    if value["schema_version"] != 1 or value["purpose"] != "SHARE-Guard metadata-only historical identifier index":
        raise SystemExit("unsupported historical identifier index schema")
    episode_ids = value["historical_episode_ids"]
    group_ids = value["historical_group_ids"]
    if not all(isinstance(items, list) and all(isinstance(item, str) and item for item in items) for items in (episode_ids, group_ids)):
        raise SystemExit("historical identifier index contains invalid identifiers")
    if len(episode_ids) != len(set(episode_ids)) or len(group_ids) != len(set(group_ids)):
        raise SystemExit("historical identifier index contains duplicate identifiers")
    if (
        not isinstance(value["source_sha256"], str)
        or len(value["source_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["source_sha256"])
        or value["source_episodes"] != len(episode_ids)
        or value["source_groups"] != len(group_ids)
    ):
        raise SystemExit("historical identifier index source metadata is inconsistent")
    return value


def load_real_paired_import_report(path: Path, episodes_path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("--import-report must be an existing JSON file")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit("--import-report must contain JSON") from error
    if report.get("schema_version") != 1 or report.get("evidence_kind") != "externally_collected_real_paired_outcomes":
        raise SystemExit("--import-report is not a supported real paired-outcome import report")
    expected_sha256 = sha256_file(episodes_path)
    if report.get("output_sha256") != expected_sha256:
        raise SystemExit("--import-report output_sha256 does not match --episodes")
    models = report.get("ai_model_episode_counts")
    if not isinstance(models, dict) or not models or not all(isinstance(key, str) and key and isinstance(value, int) and value > 0 for key, value in models.items()):
        raise SystemExit("--import-report requires non-empty AI model counts")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the auditable manifest required for a new SHARE locked holdout.")
    parser.add_argument("--episodes", type=Path, required=True, help="One JSONL file containing only previously uninspected holdout groups.")
    parser.add_argument("--frozen-at", required=True, help="ISO-8601 timestamp recorded when the holdout was locked.")
    parser.add_argument("--attestation", required=True, help="External provenance or preregistration reference asserting prior-unseen status.")
    parser.add_argument("--import-report", type=Path, required=True, help="Report emitted by import_share_real_paired_episodes.py for this exact input file.")
    parser.add_argument("--historical-identifier-index", type=Path, required=True, help="Identifier-only index of all historical development groups and source binding.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.episodes.is_file():
        raise SystemExit("--episodes must be a single JSONL file")
    try:
        frozen_at = datetime.fromisoformat(args.frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise SystemExit("--frozen-at must be an ISO-8601 timestamp") from error
    if frozen_at.tzinfo is None:
        raise SystemExit("--frozen-at must include a UTC offset or Z")
    if not args.attestation.strip():
        raise SystemExit("--attestation must be a non-empty external provenance reference")
    episodes = load_episodes(args.episodes)
    if any(episode.synthetic for episode in episodes):
        raise SystemExit("locked holdout requires externally collected paired outcomes with synthetic=false")
    import_report = load_real_paired_import_report(args.import_report, args.episodes)
    historical_index = load_historical_identifier_index(args.historical_identifier_index)
    manifest = {
        "schema_version": 1,
        "protocol_version": LOCKED_HOLDOUT_PROTOCOL_VERSION,
        "dataset_sha256": sha256_file(args.episodes),
        "episodes": len(episodes),
        "workflow_groups": len({episode.group_id for episode in episodes}),
        "episode_ids_sha256": identifier_digest([episode.episode_id for episode in episodes]),
        "group_ids_sha256": identifier_digest([episode.group_id for episode in episodes]),
        "historical_identifier_index_sha256": sha256_file(args.historical_identifier_index),
        "historical_source_sha256": historical_index["source_sha256"],
        "historical_source_episodes": historical_index["source_episodes"],
        "historical_source_groups": historical_index["source_groups"],
        "frozen_at": args.frozen_at,
        "prior_unseen_attestation": args.attestation.strip(),
        "evidence_kind": "externally_collected_real_paired_outcomes",
        "import_report_sha256": sha256_file(args.import_report),
        "ai_model_episode_counts": import_report["ai_model_episode_counts"],
        "warning": "This manifest records supplied provenance. It cannot itself prove that the data were previously unseen.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "episodes": manifest["episodes"], "groups": manifest["workflow_groups"]}, indent=2))


if __name__ == "__main__":
    main()
