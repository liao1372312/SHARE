"""Count controller categorical OOV features across development and replay sets."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from share.data import load_episodes
from share.features import FeatureEncoder

ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
SETS = {
    "original_development": ROOT / "datasets/share_synthetic/episodes.jsonl",
    "component_232": ROOT / "datasets/share_component_confirmation_232_20260930/episodes.jsonl",
    "prior_v4_250": ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl",
    "prior_v5_150": ROOT / "datasets/share_v5_expected_voh_holdout_20260829/episodes.jsonl",
}
OUT = ROOT / "results/share_feature_schema_audit_20260930.json"


def main() -> None:
    encoder = FeatureEncoder.load(ENCODER)
    result = {"encoder_sha256": sha256_file(ENCODER), "vocabulary_sizes": {"gap_type": len(encoder.gap_types), "root_type": len(encoder.root_types), "domain": len(encoder.domains)}, "datasets": {}}
    for name, path in SETS.items():
        episodes = load_episodes(path)
        fields = {field: Counter() for field in ("gap_type", "root_type", "domain")}
        by_split = Counter()
        for ep in episodes:
            by_split[ep.split] += 1
            for step in ep.steps:
                for field in fields:
                    fields[field][getattr(step.state, field)] += 1
        states = sum(len(ep.steps) for ep in episodes)
        result["datasets"][name] = {
            "episodes": len(episodes), "states": states, "split_counts": dict(by_split),
            "sha256": sha256_file(path),
            "oov": {field: {"count": sum(n for value, n in count.items() if value not in getattr(encoder, field.replace("_type", "_types") if field != "domain" else "domains")), "observed_values": dict(count)} for field, count in fields.items()},
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"episodes": row["episodes"], "states": row["states"], "oov_counts": {field: values["count"] for field, values in row["oov"].items()}} for name, row in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
