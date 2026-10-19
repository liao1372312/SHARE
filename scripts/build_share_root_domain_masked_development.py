"""Freeze a development copy whose root_type and domain match OOV deployment."""
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
ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
OUT = ROOT / "datasets/share_root_domain_masked_development_20260930"
EXPECTED_SOURCE = "174a2c387aa8fe4cece6374a54a9d23c5db062bde6349c71578384c386b5a694"
ROOT_MASK = "__root_type_out_of_vocabulary__"
DOMAIN_MASK = "__domain_out_of_vocabulary__"


def main() -> None:
    if OUT.exists():
        raise SystemExit("refusing to overwrite root/domain-masked development dataset")
    if sha256_file(SOURCE) != EXPECTED_SOURCE:
        raise SystemExit("source development dataset changed")
    encoder = FeatureEncoder.load(ENCODER)
    if ROOT_MASK in encoder.root_types or DOMAIN_MASK in encoder.domains:
        raise SystemExit("a mask token is in the fitted vocabulary")
    original = load_episodes(SOURCE)
    masked = [replace(ep, steps=[replace(step, state=replace(step.state, root_type=ROOT_MASK, domain=DOMAIN_MASK)) for step in ep.steps]) for ep in original]
    for old, new in zip(original, masked):
        if (old.episode_id, old.group_id, old.split) != (new.episode_id, new.group_id, new.split):
            raise AssertionError("episode identity or split changed")
        for a, b in zip(old.steps, new.steps):
            if a.outcomes != b.outcomes or a.human_beneficial != b.human_beneficial:
                raise AssertionError("branch outcome or label changed")
            old_state, new_state = asdict(a.state), asdict(b.state)
            for field in ("root_type", "domain"):
                old_state.pop(field); new_state.pop(field)
            if old_state != new_state:
                raise AssertionError("nonmasked state field changed")
    OUT.mkdir(parents=True)
    path = OUT / "episodes.jsonl"
    write_episodes(path, masked)
    manifest = {
        "protocol": "share-training-root-domain-oov-alignment-v1",
        "status": "frozen_before_model_training",
        "source_sha256": EXPECTED_SOURCE,
        "episodes_sha256": sha256_file(path),
        "encoder_sha256": sha256_file(ENCODER),
        "root_mask": ROOT_MASK,
        "domain_mask": DOMAIN_MASK,
        "split_counts": {split: sum(ep.split == split for ep in masked) for split in ("train", "validation", "test")},
        "states": sum(len(ep.steps) for ep in masked),
        "outcomes_and_labels_unchanged": True,
        "holdout_outcomes_read": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
