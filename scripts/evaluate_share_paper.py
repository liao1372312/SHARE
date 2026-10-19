"""Evaluate supplied observed branches and a frozen five-member SHARE ensemble."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.ledger import OnlineHardHumanCapPolicy
from share.paper import bound_file, identifier_digest, load_observed_episodes, paper_metrics, require_disjoint, sha256_file
from share.support import SupportCheckedPolicy, support_key
from share.types import AuthorityAction, RewardConfig


def evaluate(args):
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    if artifacts.get("status") != "ready" or artifacts.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("frozen paper artifacts are not ready or use another protocol")
    if args.out.exists():
        raise ValueError("use a new --out directory to preserve earlier runs")
    base = args.artifacts.resolve().parent
    encoder_path = bound_file(artifacts["encoder"], base)
    development_report = bound_file(artifacts["training_import_report"], base)
    development_meta = json.loads(development_report.read_text(encoding="utf-8"))
    development = load_observed_episodes(Path(development_meta["output_path"]), development_report)
    episodes = load_observed_episodes(args.episodes, args.import_report)
    if any(e.split != "test" for e in episodes):
        raise ValueError("evaluation requires the declared test partition")
    require_disjoint(development, episodes)
    expected = protocol["datasets"][args.dataset_role]
    if len(episodes) != expected["episodes"] or sum(len(e.steps) for e in episodes) != expected["states"]:
        raise ValueError("evaluation sample size differs from the declared paper protocol")
    training = [e for e in development if e.split == "train"]
    validation = [e for e in development if e.split == "validation"]
    if Counter(e.split for e in development) != protocol["datasets"]["development"]["splits"]:
        raise ValueError("development split counts differ from the paper protocol")
    encoder = FeatureEncoder.load(encoder_path)
    if encoder.dimension != protocol["state_dimension"]:
        raise ValueError("encoder dimension differs from the paper protocol")
    fitted_encoder = FeatureEncoder.fit(step.state for episode in training for step in episode.steps)
    if (encoder.gap_types != fitted_encoder.gap_types or encoder.root_types != fitted_encoder.root_types
            or encoder.domains != fitted_encoder.domains or encoder.disabled_groups
            or not np.allclose(encoder.mean, fitted_encoder.mean)
            or not np.allclose(encoder.scale, fitted_encoder.scale)):
        raise ValueError("encoder preprocessing does not match training-only fitting")
    calibration_path = bound_file(artifacts["validation_calibration"], base)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (calibration.get("partition") != "validation"
            or calibration.get("selection_data_sha256") != development_meta["output_sha256"]
            or calibration.get("validation_episode_ids_sha256") != identifier_digest([e.episode_id for e in validation])
            or calibration.get("support_episode_ids_sha256") != identifier_digest([e.episode_id for e in training])):
        raise ValueError("calibration/support bindings do not match the development partitions")
    penalty = artifacts.get("human_calibration_penalty")
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or calibration.get("human_calibration_penalty") != penalty:
        raise ValueError("a matching finite validation-selected Human penalty is required")
    entries = artifacts["checkpoints"]
    if [e["seed"] for e in entries] != protocol["seeds"]:
        raise ValueError("checkpoint membership differs from the paper's five seeds")
    reward = RewardConfig(rollback_penalty=protocol["training"]["training_stop_coefficient"])
    paths = [bound_file(e, base) for e in entries]
    members = [DQNPolicy.load(str(path), encoder, reward, require_reward_config=True, device=args.device) for path in paths]
    for member, entry in zip(members, entries):
        if member.config.seed != entry["seed"] or not member.config.paired_replay:
            raise ValueError("checkpoint seed or paired-replay setting differs from its declared member")
        if member.config.unsafe_constraint:
            raise ValueError("paper SHARE uses no separate unsafe dual constraint")
    policy = DQNEnsemblePolicy(members, encoder, members[0].config, reward)
    policy.name = "share_paper_q_ensemble"
    policy.human_calibration_penalty = float(penalty)
    counts = Counter((support_key(step.state), int(AuthorityAction.AI)) for e in training for step in e.steps if step.state.remaining_budget > 0)
    supported = SupportCheckedPolicy(policy, counts, minimum_count=protocol["minimum_ai_support"])
    budget = protocol["human_budget"] if args.budget is None else args.budget
    if not 0 <= budget <= 1:
        raise ValueError("--budget must lie between zero and one")
    wrapper = OnlineHardHumanCapPolicy(supported, math.floor(budget * len(episodes)))
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), budget, None)
    result = paper_metrics(raw)
    result.update({"hard_human_cap": wrapper.cap, "human_interventions": wrapper.human_interventions,
                   "cap_respected": wrapper.human_interventions <= wrapper.cap,
                   "human_branches_available": sum(len(e.steps) for e in episodes),
                   "human_branch_successes": sum(s.outcomes["human"].success for e in episodes for s in e.steps),
                   "evidence_kind": "supplied_observed_branches_with_verified_internal_bindings"})
    manifest = {"protocol_id": protocol["protocol_id"], "dataset_role": args.dataset_role,
                "episodes_sha256": sha256_file(args.episodes), "import_report_sha256": sha256_file(args.import_report),
                "artifacts_manifest_sha256": sha256_file(args.artifacts), "protocol_sha256": sha256_file(args.protocol),
                "evaluator_sha256": sha256_file(Path(__file__)), "device": str(members[0].device),
                "budget": budget, "evaluation_reward": asdict(RewardConfig()),
                "checkpoint_sha256": [sha256_file(p) for p in paths],
                "source_authenticity": "document separately; file hashes verify internal consistency only"}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in [("metrics.json", result), ("manifest.json", manifest), ("decision_ledger.json", wrapper.events)]:
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in raw["records"]), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--import-report", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol.json")
    parser.add_argument("--dataset-role", choices=["primary", "mechanism", "retrospective"], default="primary")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Paper evaluation cannot start: {error}") from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
