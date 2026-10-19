"""Freeze a source-aligned SHARE development set from disjoint historical groups.

This deliberately repurposes historical V1 replay groups for training and V2
groups for validation. It creates a new exploratory training regime; the old
V1/V2 results must no longer be described as independent tests for that model.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from share.data import load_episodes, write_episodes
from share.features import FeatureEncoder


OLD_DEV = ROOT / "datasets/share_synthetic/episodes.jsonl"
V1 = ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl"
V2 = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
V3 = ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl"
NEW232 = ROOT / "datasets/share_component_confirmation_232_20260930/episodes.jsonl"
OUT = ROOT / "datasets/share_source_adaptation_20260930"


def main() -> None:
    if OUT.exists():
        raise SystemExit("refusing to overwrite source-adaptation dataset")
    original_train = [ep for ep in load_episodes(OLD_DEV) if ep.split == "train"]
    source_train = [replace(ep, split="train") for ep in load_episodes(V1)]
    validation = [replace(ep, split="validation") for ep in load_episodes(V2)]
    gate = load_episodes(V3)
    holdout = load_episodes(NEW232)
    train = [*original_train, *source_train]
    sizes = (len(original_train), len(source_train), len(validation), len(gate), len(holdout))
    if sizes != (542, 400, 400, 400, 232):
        raise SystemExit(f"unexpected source split sizes: {sizes}")
    ids = [{ep.group_id for ep in part} for part in (train, validation, gate, holdout)]
    if any(len(ids[i]) != len(part) for i, part in enumerate((train, validation, gate, holdout)) if i > 0):
        raise SystemExit("duplicate group ID within a validation or evaluation split")
    if any(ids[i] & ids[j] for i in range(4) for j in range(i + 1, 4)):
        raise SystemExit("source-adaptation group overlap")
    episode_ids = [{ep.episode_id for ep in part} for part in (train, validation, gate, holdout)]
    if any(episode_ids[i] & episode_ids[j] for i in range(4) for j in range(i + 1, 4)):
        raise SystemExit("source-adaptation episode overlap")
    encoder = FeatureEncoder.fit(step.state for ep in train for step in ep.steps)
    OUT.mkdir(parents=True)
    train_path, validation_path, encoder_path = OUT / "train.jsonl", OUT / "validation.jsonl", OUT / "feature_encoder.json"
    write_episodes(train_path, train)
    write_episodes(validation_path, validation)
    encoder.save(encoder_path)
    manifest = {
        "protocol": "share-public-api-source-adaptation-v1",
        "status": "training_and_validation_frozen_before_new_training",
        "evidence_class": "exploratory_reuse_of_historical_replay_groups",
        "train": {"episodes": len(train), "groups": len(ids[0]), "states": sum(len(ep.steps) for ep in train), "sha256": sha256_file(train_path), "sources": ["original Flare development train", "historical public-API V1 replay"]},
        "validation": {"episodes": len(validation), "states": sum(len(ep.steps) for ep in validation), "sha256": sha256_file(validation_path), "source": "historical public-API V2 replay"},
        "gate": {"episodes": len(gate), "sha256": sha256_file(V3), "source": "historical public-API V3 replay; no model training or threshold search"},
        "future_diagnostic_232": {"episodes": len(holdout), "sha256": sha256_file(NEW232), "previously_inspected": True},
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (OLD_DEV, V1, V2, V3, NEW232)},
        "encoder_sha256": sha256_file(encoder_path), "encoder_dimension": encoder.dimension,
        "encoder_root_types": encoder.root_types,
        "encoder_domains": encoder.domains,
        "group_overlap": 0,
        "outcomes_and_labels_unchanged": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"train": manifest["train"], "validation": manifest["validation"], "gate": manifest["gate"], "encoder_dimension": encoder.dimension, "group_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
