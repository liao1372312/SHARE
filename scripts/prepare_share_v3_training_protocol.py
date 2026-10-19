"""Freeze SHARE-V3 data roles and fixed training protocol before training."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes


OUT = ROOT / "results/share_v3_training_protocol_20260827"
DEVELOPMENT = ROOT / "datasets/share_synthetic/episodes.jsonl"
OLD_LOCKED = ROOT / "datasets/share_locked_holdout_replayable_20260826/episodes.jsonl"
VALIDATION = ROOT / "datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl"
TRAINER = ROOT / "scripts/train_share_v3_fixed.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("refusing to overwrite V3 protocol")
    train = load_episodes(DEVELOPMENT) + load_episodes(OLD_LOCKED)
    validation = load_episodes(VALIDATION)
    train_episode_ids = {episode.episode_id for episode in train}
    validation_episode_ids = {episode.episode_id for episode in validation}
    train_groups = {episode.group_id for episode in train}
    validation_groups = {episode.group_id for episode in validation}
    episode_overlap = sorted(train_episode_ids & validation_episode_ids)
    group_overlap = sorted(train_groups & validation_groups)
    if episode_overlap or group_overlap:
        raise SystemExit("V3 train/validation overlap")
    manifest = {
        "protocol": "share-v3-fixed-old400-augmentation-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "method_change": "training-data augmentation only; add the opened original 400-workflow holdout to training",
        "fixed_from_v2": {
            "architecture": [128, 64],
            "double_dqn": True,
            "paired_replay": True,
            "gamma": 0.95,
            "rollback_penalty": 0.60,
            "epochs": 30,
            "learning_rate": 0.001,
            "batch_size": 64,
            "target_update": 200,
            "updates_per_epoch": 1000,
            "human_constraint": True,
            "unsafe_constraint": False,
            "support_mode": "posthoc_stop",
            "minimum_ai_support": 3,
        },
        "budgets": [0.10, 0.20, 0.30],
        "seeds": [17, 29, 43, 71, 101],
        "data_roles": {
            "training": [
                {"path": str(DEVELOPMENT), "sha256": sha256(DEVELOPMENT), "episodes": len(load_episodes(DEVELOPMENT)), "status": "previously inspected development"},
                {"path": str(OLD_LOCKED), "sha256": sha256(OLD_LOCKED), "episodes": len(load_episodes(OLD_LOCKED)), "status": "opened original locked set; permanently reclassified as V3 training"},
            ],
            "validation": {
                "path": str(VALIDATION),
                "sha256": sha256(VALIDATION),
                "episodes": len(validation),
                "status": "opened V2 confirmation set; V3 checkpoint selection/calibration only; ineligible for V3 final claims",
            },
            "future_final_test": "must be generated from a third set of previously unused workflow groups after V3 artifacts are frozen",
        },
        "counts": {
            "training_episodes": len(train),
            "training_unique_episode_ids": len(train_episode_ids),
            "training_unique_groups": len(train_groups),
            "validation_episodes": len(validation),
            "validation_unique_groups": len(validation_groups),
        },
        "overlap_audit": {
            "episode_id_overlap": len(episode_overlap),
            "group_id_overlap": len(group_overlap),
            "result": "PASS",
        },
        "selection_rule": "no architecture or reward search; train one fixed configuration per nominal budget; validation is used only for epoch checkpoint selection and existing constraint calibration",
        "prohibitions": [
            "do not report validation as V3 locked/final test",
            "do not change V3 after opening the third final test",
            "do not overwrite V2 artifacts or original locked evidence",
        ],
        "source_hashes": {
            "trainer": sha256(TRAINER),
            "dqn": sha256(ROOT / "share/dqn.py"),
            "hard_cap": sha256(ROOT / "scripts/run_share_online_hard_cap.py"),
        },
    }
    OUT.mkdir(parents=True)
    target = OUT / "data_role_and_training_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "SHA256SUMS").write_text(f"{sha256(target)}  {target.name}\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "training": len(train), "validation": len(validation), "overlap": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
