"""Freeze a train/validation/test development copy with root_type out of vocabulary.

The intervention changes only the model input field root_type. It preserves
episode IDs, split membership, every branch outcome, and Human-benefit labels.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from share.data import load_episodes, write_episodes
from share.features import FeatureEncoder


SOURCE = ROOT / "datasets/share_synthetic/episodes.jsonl"
OUT = ROOT / "datasets/share_root_masked_development_20260930"
ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
EXPECTED_SOURCE = "174a2c387aa8fe4cece6374a54a9d23c5db062bde6349c71578384c386b5a694"
MASK = "__root_type_out_of_vocabulary__"


def main() -> None:
    if OUT.exists():
        raise SystemExit("refusing to overwrite root-masked development dataset")
    if sha256_file(SOURCE) != EXPECTED_SOURCE:
        raise SystemExit("source development dataset changed")
    encoder = FeatureEncoder.load(ENCODER)
    if MASK in encoder.root_types:
        raise SystemExit("mask token is not out of vocabulary")
    original = load_episodes(SOURCE)
    masked = [replace(ep, steps=[replace(step, state=replace(step.state, root_type=MASK)) for step in ep.steps]) for ep in original]
    for old, new in zip(original, masked):
        if old.episode_id != new.episode_id or old.group_id != new.group_id or old.split != new.split:
            raise AssertionError("episode identity or split changed")
        for a, b in zip(old.steps, new.steps):
            if a.outcomes != b.outcomes or a.human_beneficial != b.human_beneficial:
                raise AssertionError("branch outcome or label changed")
            old_state, new_state = asdict(a.state), asdict(b.state)
            old_state.pop("root_type"); new_state.pop("root_type")
            if old_state != new_state:
                raise AssertionError("non-root state field changed")
    OUT.mkdir(parents=True)
    path = OUT / "episodes.jsonl"
    write_episodes(path, masked)
    manifest = {
        "protocol": "share-training-root-oov-alignment-v1",
        "status": "frozen_before_model_training",
        "source_sha256": EXPECTED_SOURCE,
        "episodes_sha256": sha256_file(path),
        "encoder_sha256": sha256_file(ENCODER),
        "root_mask": MASK,
        "split_counts": {split: sum(ep.split == split for ep in masked) for split in ("train", "validation", "test")},
        "states": sum(len(ep.steps) for ep in masked),
        "outcomes_and_labels_unchanged": True,
        "holdout_outcomes_read": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
