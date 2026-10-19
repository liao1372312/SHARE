from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.evaluation import immediate_reward_tensor
from share.features import FeatureEncoder
from share.policies import ContextualBanditPolicy, SupervisedBenefitPolicy, ThresholdPolicy
from share.types import RewardConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze deterministic artifacts for the hard-cap evaluation.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--calibration-results", type=Path, default=Path("results/share_budget_matched_20pct_5seed_20260825"))
    parser.add_argument("--checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rescue_manifest = json.loads((args.checkpoints / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != rescue_manifest["episode_sha256"]:
        raise SystemExit("episodes differ from the frozen rescue dataset")
    episodes = load_episodes(args.episodes)
    train = [episode for episode in episodes if episode.split == "train"]
    encoder = FeatureEncoder.load(args.checkpoints / "feature_encoder.json")
    reward = RewardConfig(**rescue_manifest["reward_config"])
    rewards = immediate_reward_tensor(train, reward)
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for seed in seeds:
        calibration_path = args.calibration_results / f"seed_{seed}.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["calibrations"]
        conf = ThresholdPolicy("uncertainty", float(calibration["share_conf"]["selected_scalar"]), True, "share_conf")
        static = SupervisedBenefitPolicy(encoder, 0.20, seed).fit(train)
        static.name = "share_static"
        static.threshold = float(calibration["share_static"]["selected_scalar"])
        bandit = ContextualBanditPolicy(encoder, 0.20).fit(train, rewards)
        bandit.name = "share_bandit"
        bandit.human_penalty = float(calibration["share_bandit"]["selected_scalar"])
        path = args.out / f"comparators_seed_{seed}.joblib"
        joblib.dump(
            {
                "share_conf": conf,
                "share_static": static,
                "share_bandit": bandit,
                "source_calibration": calibration_path.name,
            },
            path,
            compress=3,
        )
        artifacts[str(seed)] = {"path": path.name, "sha256": sha256_file(path)}
    manifest = {
        "purpose": "deterministic rehydration and immediate freeze of previously unsaved comparator parameters",
        "limitation": "Static and Bandit estimators were not serialized by the preceding run; they were reconstructed once from immutable train data and fixed seeds, without validation/evaluation outcome selection.",
        "episodes_sha256": sha256_file(args.episodes),
        "encoder_sha256": sha256_file(args.checkpoints / "feature_encoder.json"),
        "source_calibration_manifest_sha256": sha256_file(args.calibration_results / "manifest.json"),
        "seeds": seeds,
        "artifacts": artifacts,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "source_sha256": {
            "scripts/freeze_share_budget_matched_artifacts.py": sha256_file(Path(__file__)),
            "share/policies.py": sha256_file(ROOT / "share/policies.py"),
            "share/features.py": sha256_file(ROOT / "share/features.py"),
        },
        "evaluation_split_read_for_fitting_or_selection": False,
        "locked_holdout_accessed": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "artifacts": len(artifacts)}))


if __name__ == "__main__":
    main()
