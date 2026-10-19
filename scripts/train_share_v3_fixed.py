"""Train frozen SHARE-V3 using opened old evidence as training and V2 confirmation as validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_v2_development_multistart import METRICS, atomic_json, config_for, summarize_without_records
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig

PROTOCOL = ROOT / "results/share_v3_training_protocol_20260827/data_role_and_training_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(policy: DQNPolicy, episodes: Sequence[RepairEpisode], budget: float) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), budget, None)
    human_episodes = sum(int(row["human_used"]) for row in raw["records"])
    human_actions = sum(int(action == int(AuthorityAction.HUMAN)) for row in raw["records"] for action in row["actions"])
    if human_episodes != wrapper.human_interventions or human_episodes > cap:
        raise RuntimeError("V3 validation hard-cap violation")
    summary = summarize_without_records(raw)
    summary.update(hard_human_cap=cap, human_interventions=human_episodes, human_actions=human_actions, cap_respected=True)
    return summary, {
        "hard_human_cap": cap,
        "human_interventions": human_episodes,
        "human_actions": human_actions,
        "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
        "value_policy_replans": wrapper.value_policy_replans,
        "cap_respected": True,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": len(rows)}
    for metric in METRICS:
        values = [float(row["validation"][metric]) for row in rows]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values)
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    output["all_caps_respected"] = all(row["validation"]["cap_respected"] for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, choices=(0.10, 0.20, 0.30), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite V3 training ledger")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != protocol["source_hashes"]["trainer"]:
        raise SystemExit("V3 trainer differs from frozen protocol")
    for row in protocol["data_roles"]["training"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise SystemExit("V3 training data hash mismatch")
    validation_role = protocol["data_roles"]["validation"]
    if sha256(Path(validation_role["path"])) != validation_role["sha256"]:
        raise SystemExit("V3 validation data hash mismatch")
    training = [episode for role in protocol["data_roles"]["training"] for episode in load_episodes(Path(role["path"]))]
    validation = load_episodes(Path(validation_role["path"]))
    encoder = FeatureEncoder.fit(step.state for episode in training for step in episode.steps)
    args.out.mkdir(parents=True)
    encoder.save(args.out / "feature_encoder.json")
    training_reward = RewardConfig(rollback_penalty=0.60)
    run_manifest = {
        "protocol": "share-v3-fixed-training-run-v1",
        "parent_protocol": str(PROTOCOL),
        "parent_protocol_sha256": sha256(PROTOCOL),
        "budget": args.budget,
        "device": args.device,
        "training_episodes": len(training),
        "validation_episodes": len(validation),
        "encoder_sha256": sha256(args.out / "feature_encoder.json"),
        "validation_is_not_final_test": True,
    }
    atomic_json(args.out / "manifest.json", run_manifest)
    rows = []
    for seed in protocol["seeds"]:
        config = replace(config_for(int(seed), "paired_e30", args.device), human_budget=args.budget)
        policy = DQNPolicy(encoder, config, training_reward)
        policy.name = "share_v3"
        policy.fit(training, validation)
        policy.calibrate_constraints(validation, args.budget, 0.05)
        validation_result, cap_audit = evaluate(policy, validation, args.budget)
        checkpoint = args.out / f"share_v3_budget_{args.budget:.1f}_seed_{seed}.pt"
        policy.save(str(checkpoint))
        row = {
            "seed": int(seed),
            "budget": args.budget,
            "validation": validation_result,
            "validation_cap_audit": cap_audit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config": asdict(config),
            "selected_checkpoint_pre_calibration": policy.selected_checkpoint_diagnostics,
        }
        atomic_json(args.out / f"seed_{seed}.json", row)
        rows.append(row)
        print(json.dumps({"budget": args.budget, "seed": seed, "complete": True}), flush=True)
    summary = {
        "evidence_class": "V3_validation_only_not_final",
        "budget": args.budget,
        "aggregate": aggregate(rows),
        "interpretation_boundary": "The 400 episodes were previously opened for V2 and are used for V3 validation/calibration. These values are ineligible as independent V3 test evidence.",
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"out": str(args.out), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
