"""Retrain V3 comparators on the exact V3 training/validation roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_budget_matched_comparison import breakpoint_candidates, calibrate_scalar_validation_only
from scripts.run_share_controller_rescue import dqn_config
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import immediate_reward_tensor
from share.features import FeatureEncoder
from share.policies import ContextualBanditPolicy, SupervisedBenefitPolicy, ThresholdPolicy
from share.types import RewardConfig

PROTOCOL = ROOT / "results/share_v3_training_protocol_20260827/data_role_and_training_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, choices=(0.10, 0.20, 0.30), required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite V3 comparator ledger")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training = [episode for role in protocol["data_roles"]["training"] for episode in load_episodes(Path(role["path"]))]
    validation = load_episodes(Path(protocol["data_roles"]["validation"]["path"]))
    encoder_path = args.v3_root / "feature_encoder.json"
    encoder = FeatureEncoder.load(encoder_path)
    reward = RewardConfig()
    rewards = immediate_reward_tensor(training, reward)
    validation_states = [step.state for episode in validation for step in episode.steps]
    args.out.mkdir(parents=True)
    files, calibrations = {}, {}
    for seed in protocol["seeds"]:
        conf = ThresholdPolicy("uncertainty", 0.0, True, "share_conf")
        conf_cal = calibrate_scalar_validation_only(
            conf, validation, breakpoint_candidates([state.uncertainty for state in validation_states]),
            lambda value: setattr(conf, "threshold", value), target=args.budget, lower=args.budget, upper=args.budget,
        )
        static = SupervisedBenefitPolicy(encoder, args.budget, int(seed)).fit(training)
        static.name = "share_static"
        static_cal = calibrate_scalar_validation_only(
            static, validation, breakpoint_candidates([static._score(state) for state in validation_states]),
            lambda value: setattr(static, "threshold", value), target=args.budget, lower=args.budget, upper=args.budget,
        )
        bandit = ContextualBanditPolicy(encoder, args.budget).fit(training, rewards)
        bandit.name = "share_bandit"
        bandit_cal = calibrate_scalar_validation_only(
            bandit, validation, breakpoint_candidates([bandit._advantage(state) for state in validation_states]),
            lambda value: setattr(bandit, "human_penalty", value), target=args.budget, lower=args.budget, upper=args.budget,
        )
        ddqn = DQNPolicy(
            encoder,
            dqn_config("share_ddqn", seed=int(seed), epochs=30, human_budget=args.budget, unsafe_budget=0.05, device=args.device),
            reward,
        )
        ddqn.name = "share_ddqn"
        ddqn.fit(training, validation)
        ddqn.calibrate_human_budget(validation, args.budget)
        pack_path = args.out / f"comparators_seed_{seed}.joblib"
        ddqn_path = args.out / f"share_ddqn_seed_{seed}.pt"
        joblib.dump({"share_conf": conf, "share_static": static, "share_bandit": bandit}, pack_path, compress=3)
        ddqn.save(str(ddqn_path))
        files[str(seed)] = {
            "comparators": {"path": str(pack_path), "sha256": sha256(pack_path)},
            "share_ddqn": {"path": str(ddqn_path), "sha256": sha256(ddqn_path)},
        }
        calibrations[str(seed)] = {
            "share_conf": conf_cal,
            "share_static": static_cal,
            "share_bandit": bandit_cal,
            "share_ddqn": {
                "selection_split": "V3 validation",
                "human_calibration_penalty": ddqn.human_calibration_penalty,
                "validation_hir": ddqn.calibration_validation_hir,
            },
        }
        print(json.dumps({"budget": args.budget, "seed": seed, "complete": True}), flush=True)
    manifest = {
        "protocol": "share-v3-fair-comparators-v1",
        "parent_protocol": str(PROTOCOL),
        "parent_protocol_sha256": sha256(PROTOCOL),
        "budget": args.budget,
        "seeds": protocol["seeds"],
        "training_episodes": len(training),
        "validation_episodes": len(validation),
        "encoder_sha256": sha256(encoder_path),
        "definitions": {
            "share_conf": "uncertainty threshold; validation HIR calibration",
            "share_static": "random-forest Human-benefit classifier; validation HIR calibration",
            "share_bandit": "ridge one-step branch-return models; validation HIR calibration",
            "share_ddqn": "unconstrained Double DQN with historical standard replay; validation Human penalty calibration",
        },
        "files": files,
        "calibrations": calibrations,
        "validation_is_not_final_test": True,
        "source_sha256": {
            "script": sha256(Path(__file__)),
            "policies": sha256(ROOT / "share/policies.py"),
            "dqn": sha256(ROOT / "share/dqn.py"),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
