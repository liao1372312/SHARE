from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_budget_matched_comparison import breakpoint_candidates, calibrate_scalar_validation_only
from scripts.run_share_controller_rescue import dqn_config
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


SEEDS = (17, 29, 43, 71, 101)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def budget_slug(budget: float) -> str:
    return f"{budget:g}".replace(".", "p")


def calibrate_comparators(pack: dict, validation, budget: float, ddqn: DQNPolicy) -> dict:
    states = [step.state for episode in validation for step in episode.steps]
    calibrations = {}
    conf = pack["share_conf"]
    calibrations["share_conf"] = calibrate_scalar_validation_only(
        conf, validation, breakpoint_candidates([state.uncertainty for state in states]),
        lambda value: setattr(conf, "threshold", value),
        target=budget, lower=budget, upper=budget,
    )
    static = pack["share_static"]
    calibrations["share_static"] = calibrate_scalar_validation_only(
        static, validation, breakpoint_candidates([static._score(state) for state in states]),
        lambda value: setattr(static, "threshold", value),
        target=budget, lower=budget, upper=budget,
    )
    bandit = pack["share_bandit"]
    calibrations["share_bandit"] = calibrate_scalar_validation_only(
        bandit, validation, breakpoint_candidates([bandit._advantage(state) for state in states]),
        lambda value: setattr(bandit, "human_penalty", value),
        target=budget, lower=budget, upper=budget,
    )
    ddqn.human_calibration_penalty = 0.0
    ddqn.ai_calibration_penalty = 0.0
    raw = ddqn._raw_values_batch(states)
    values = raw[:, AuthorityAction.HUMAN] - np.maximum(raw[:, AuthorityAction.AI], raw[:, AuthorityAction.STOP])
    calibrations["share_ddqn"] = calibrate_scalar_validation_only(
        ddqn, validation, breakpoint_candidates(values.tolist()),
        lambda value: setattr(ddqn, "human_calibration_penalty", value),
        target=budget, lower=budget, upper=budget,
    )
    return calibrations


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one frozen budget point for the hard-cap curve.")
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--base-artifacts", type=Path, default=Path("results/share_budget_matched_frozen_artifacts_20260825"))
    parser.add_argument("--base-checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--base-calibration-results", type=Path, default=Path("results/share_budget_matched_20pct_5seed_20260825"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    if args.budget not in {0.0, 0.1, 0.2, 0.3, 0.5}:
        raise SystemExit("budget must be one of 0,0.1,0.2,0.3,0.5")
    seeds = [int(value) for value in args.seeds.split(",")]
    base_manifest = json.loads((args.base_checkpoints / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != base_manifest["episode_sha256"]:
        raise SystemExit("dataset hash differs from frozen 20% run")
    episodes = load_episodes(args.episodes)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    encoder = FeatureEncoder.load(args.base_checkpoints / "feature_encoder.json")
    reward = RewardConfig(**base_manifest["reward_config"])
    out = args.out_root / f"budget_{budget_slug(args.budget)}"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    calibration_rows = {}
    for seed in seeds:
        comparator_out = out / f"comparators_seed_{seed}.joblib"
        ddqn_out = out / f"share_ddqn_seed_{seed}.pt"
        share_h_out = out / f"share_h_seed_{seed}.pt"
        if args.budget == 0.2:
            shutil.copy2(args.base_artifacts / f"comparators_seed_{seed}.joblib", comparator_out)
            ddqn = DQNPolicy.load(
                str(args.base_checkpoints / f"share_ddqn_seed_{seed}.pt"), encoder, reward,
                require_reward_config=True,
            )
            frozen_calibration = json.loads(
                (args.base_calibration_results / f"seed_{seed}.json").read_text(encoding="utf-8")
            )["calibrations"]["share_ddqn"]
            ddqn.human_calibration_penalty = float(frozen_calibration["selected_scalar"])
            ddqn.ai_calibration_penalty = 0.0
            ddqn.save(str(ddqn_out))
            shutil.copy2(args.base_checkpoints / f"share_h_seed_{seed}.pt", share_h_out)
            calibration_rows[str(seed)] = {
                "source": "frozen_validated_20pct_artifacts",
                "share_ddqn_selected_scalar": ddqn.human_calibration_penalty,
            }
        else:
            pack = joblib.load(args.base_artifacts / f"comparators_seed_{seed}.joblib")
            ddqn = DQNPolicy(
                encoder,
                dqn_config(
                    "share_ddqn", seed=seed, epochs=args.epochs,
                    human_budget=args.budget, unsafe_budget=0.05, device=args.device,
                ),
                reward,
            )
            ddqn.name = "share_ddqn"
            ddqn.fit(train, validation)
            calibrations = calibrate_comparators(pack, validation, args.budget, ddqn)
            joblib.dump(
                {name: pack[name] for name in ("share_conf", "share_static", "share_bandit")},
                comparator_out,
                compress=3,
            )
            ddqn.save(str(ddqn_out))
            config = dqn_config(
                "share_h", seed=seed, epochs=args.epochs,
                human_budget=args.budget, unsafe_budget=0.05, device=args.device,
            )
            share_h = DQNPolicy(encoder, config, reward)
            share_h.name = "share_h"
            share_h.fit(train, validation)
            share_h.calibrate_constraints(validation, args.budget, 0.05)
            share_h.save(str(share_h_out))
            calibrations["share_h"] = {
                "validation_hir": share_h.calibration_validation_hir,
                "human_calibration_penalty": share_h.human_calibration_penalty,
                "beta_h": share_h.beta_h,
                "beta_u": share_h.beta_u,
                "human_constraint": True,
                "unsafe_constraint": False,
            }
            calibration_rows[str(seed)] = calibrations
        files[str(seed)] = {
            "comparators": {"path": comparator_out.name, "sha256": sha256_file(comparator_out)},
            "share_ddqn": {"path": ddqn_out.name, "sha256": sha256_file(ddqn_out)},
            "share_h": {"path": share_h_out.name, "sha256": sha256_file(share_h_out)},
        }
        print(json.dumps({"budget": args.budget, "seed": seed, "status": "complete"}), flush=True)
    manifest = {
        "protocol": "share-hard-cap-curve-artifact-v1",
        "budget": args.budget,
        "seeds": seeds,
        "episodes_sha256": sha256_file(args.episodes),
        "training_splits_used": ["train", "validation"],
        "evaluation_split_used": False,
        "locked_holdout_accessed": False,
        "share_h_formulation": "Double DQN with human dual/calibration only; unsafe penalty remains in frozen reward",
        "share_h_reused_at_20pct": args.budget == 0.2,
        "ddqn_weights": "fresh budget-point training for 0/10/30/50%; frozen validated checkpoint reused at 20%; validation human penalty is budget-specific",
        "baseline_weights": "frozen Conf/Static/Bandit parameters; validation threshold/penalty is budget-specific except at 20%",
        "epochs": args.epochs,
        "device": args.device,
        "files": files,
        "calibrations": calibration_rows,
        "source_sha256": {
            "scripts/prepare_share_hard_cap_curve_budget.py": sha256_file(Path(__file__)),
            "scripts/run_share_budget_matched_comparison.py": sha256_file(ROOT / "scripts/run_share_budget_matched_comparison.py"),
            "scripts/run_share_controller_rescue.py": sha256_file(ROOT / "scripts/run_share_controller_rescue.py"),
            "share/dqn.py": sha256_file(ROOT / "share/dqn.py"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "budget": args.budget, "seeds": len(seeds)}))


if __name__ == "__main__":
    main()
