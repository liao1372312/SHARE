"""Audit the frozen expected-VoH head's episode and state provenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.train_share_expected_voh_filter import TRAIN, VALIDATION
from share.data import load_episodes

DEVELOPMENT = ROOT / "datasets/share_synthetic/episodes.jsonl"
MECHANISM = ROOT / "datasets/share_v5_expected_voh_holdout_20260829/episodes.jsonl"
FROZEN_RESULT = ROOT / "results/share_expected_voh_filter_development_20260829/FINAL_RESULTS.json"
OUT = ROOT / "results/share_auxiliary_voh_provenance_audit_20260929.json"


def summarize(path: Path, episodes) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "episodes": len(episodes),
        "decision_states": sum(len(episode.steps) for episode in episodes),
        "file_split_labels": sorted({episode.split for episode in episodes}),
        "unique_episode_ids": len({episode.episode_id for episode in episodes}),
        "unique_group_ids": len({episode.group_id for episode in episodes}),
    }


def main() -> None:
    frozen = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    training_parts = [(path, load_episodes(path)) for path in TRAIN]
    validation = load_episodes(VALIDATION)
    mechanism = load_episodes(MECHANISM)
    development = load_episodes(DEVELOPMENT)
    training = [episode for _, episodes in training_parts for episode in episodes]
    training_groups = {episode.group_id for episode in training}
    source_keys = (
        "source_document_sha256",
        "source_operation_sha256",
        "source_record_sha256",
        "normalized_source_fingerprint",
    )
    def source_overlap(other):
        return {
            key: len(
                {episode.metadata.get(key) for episode in training}
                & {episode.metadata.get(key) for episode in other}
            )
            for key in source_keys
        }
    result = {
        "auxiliary_head_training": {
            "sources": [summarize(path, episodes) for path, episodes in training_parts],
            "episodes": len(training),
            "decision_states": sum(len(episode.steps) for episode in training),
            "unique_group_ids": len(training_groups),
            "selection_rule": "all episodes and all steps from three previously evaluated replay holdouts",
        },
        "auxiliary_threshold_validation": summarize(VALIDATION, validation),
        "mechanism_evaluation": summarize(MECHANISM, mechanism),
        "original_controller_development": {
            split: summarize(DEVELOPMENT, [episode for episode in development if episode.split == split])
            for split in ("train", "validation", "test")
        },
        "group_overlap": {
            "training_vs_v4_validation": len(training_groups & {episode.group_id for episode in validation}),
            "training_vs_v5_mechanism": len(training_groups & {episode.group_id for episode in mechanism}),
            "v4_validation_vs_v5_mechanism": len({episode.group_id for episode in validation} & {episode.group_id for episode in mechanism}),
        },
        "source_hash_overlap": {
            "training_vs_v4_validation": source_overlap(validation),
            "training_vs_v5_mechanism": source_overlap(mechanism),
        },
        "frozen_result_cross_check": {
            "training_episodes": frozen["training_episodes"],
            "validation_episodes": frozen["validation_episodes"],
            "artifact_sha256": frozen["artifact_sha256"],
            "source_hashes_match": all(
                frozen["training_dataset_sha256"][str(path.relative_to(ROOT))] == sha256_file(path)
                for path in TRAIN
            ),
        },
    }
    assert result["auxiliary_head_training"]["episodes"] == frozen["training_episodes"] == 1200
    assert result["auxiliary_head_training"]["decision_states"] == 3996
    assert result["auxiliary_head_training"]["unique_group_ids"] == 1200
    assert result["auxiliary_threshold_validation"]["episodes"] == frozen["validation_episodes"] == 250
    assert result["frozen_result_cross_check"]["source_hashes_match"]
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
