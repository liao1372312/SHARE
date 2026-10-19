from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the train-only SHARE-Guard development file from a paired episode corpus.")
    parser.add_argument("--episodes", type=Path, required=True, help="Source paired JSONL used only to create a train-only development artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL containing only original split=train groups.")
    parser.add_argument("--metadata-out", type=Path, required=True, help="Metadata-only source and identifier index; it contains no branch outcomes.")
    args = parser.parse_args()
    if not args.episodes.is_file():
        raise SystemExit("--episodes must be a single JSONL file")
    episodes = load_episodes(args.episodes)
    development = [episode for episode in episodes if episode.split == "train"]
    if not development:
        raise SystemExit("source file has no split=train episodes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for episode in development:
            handle.write(json.dumps(episode.to_dict(), separators=(",", ":")) + "\n")
    index = {
        "schema_version": 1,
        "purpose": "SHARE-Guard metadata-only historical identifier index",
        "source_sha256": sha256_file(args.episodes),
        "source_episodes": len(episodes),
        "source_groups": len({episode.group_id for episode in episodes}),
        "development_sha256": sha256_file(args.out),
        "development_episodes": len(development),
        "development_groups": len({episode.group_id for episode in development}),
        "development_episode_ids_sha256": identifier_digest([episode.episode_id for episode in development]),
        "development_group_ids_sha256": identifier_digest([episode.group_id for episode in development]),
        "historical_episode_ids": sorted(episode.episode_id for episode in episodes),
        "historical_group_ids": sorted({episode.group_id for episode in episodes}),
        "warning": "The development JSONL is train-only. This metadata index contains identifiers and file hashes only, never authority-outcome branches.",
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "metadata_out": str(args.metadata_out), "episodes": len(development), "groups": index["development_groups"]}, indent=2))


if __name__ == "__main__":
    main()
