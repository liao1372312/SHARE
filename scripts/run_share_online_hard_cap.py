from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


PROTOCOL_VERSION = "share-online-hard-human-cap-20pct-v1"
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)
BINARY_POLICIES = {"share_conf", "share_static"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


from share.ledger import OnlineHardHumanCapPolicy


def load_frozen_policies(
    seed: int,
    artifacts_root: Path,
    checkpoint_root: Path,
    calibration_root: Path,
    encoder: FeatureEncoder,
    reward: RewardConfig,
) -> list[Policy]:
    artifact = joblib.load(artifacts_root / f"comparators_seed_{seed}.joblib")
    calibration = json.loads((calibration_root / f"seed_{seed}.json").read_text(encoding="utf-8"))["calibrations"]
    ddqn = DQNPolicy.load(
        str(checkpoint_root / f"share_ddqn_seed_{seed}.pt"), encoder, reward,
        require_reward_config=True,
    )
    ddqn.name = "share_ddqn"
    ddqn.human_calibration_penalty = float(calibration["share_ddqn"]["selected_scalar"])
    ddqn.ai_calibration_penalty = 0.0
    share_h = DQNPolicy.load(
        str(checkpoint_root / f"share_h_seed_{seed}.pt"), encoder, reward,
        require_reward_config=True,
    )
    share_h.name = "share_h"
    policies = [artifact[name] for name in ("share_conf", "share_static", "share_bandit")]
    return [*policies, ddqn, share_h]


def run_seed(
    seed: int,
    ordered_evaluation: Sequence[RepairEpisode],
    cap: int,
    artifacts_root: Path,
    checkpoint_root: Path,
    calibration_root: Path,
    encoder: FeatureEncoder,
    reward: RewardConfig,
) -> dict[str, Any]:
    results = []
    audits = {}
    for base in load_frozen_policies(seed, artifacts_root, checkpoint_root, calibration_root, encoder, reward):
        wrapper = OnlineHardHumanCapPolicy(base, cap, base.name in BINARY_POLICIES)
        result = evaluate_policy(wrapper, ordered_evaluation, reward, cap / len(ordered_evaluation), None)
        human_actions = sum(
            int(action == int(AuthorityAction.HUMAN))
            for record in result["records"]
            for action in record["actions"]
        )
        human_episodes = sum(int(record["human_used"]) for record in result["records"])
        if human_episodes != wrapper.human_interventions or human_episodes > cap:
            raise RuntimeError(f"{base.name}/seed={seed}: Human cap audit failed")
        result.update(
            {
                "seed": seed,
                "hard_human_cap": cap,
                "human_interventions": wrapper.human_interventions,
                "human_actions": human_actions,
                "cap_respected": wrapper.human_interventions <= cap,
                "synthetic_development_evaluation": True,
            }
        )
        audits[base.name] = {
            "hard_human_cap": cap,
            "human_interventions": wrapper.human_interventions,
            "human_actions": human_actions,
            "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
            "forced_autonomous_fallbacks": wrapper.forced_autonomous_fallbacks,
            "value_policy_replans": wrapper.value_policy_replans,
            "events": wrapper.events,
        }
        results.append(result)
    return {"seed": seed, "results": results, "budget_audit": audits}


def aggregate(seed_payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for payload in seed_payloads:
        for row in payload["results"]:
            buckets[row["policy"]].append(row)
    output = []
    for policy in POLICIES:
        rows = buckets[policy]
        item: dict[str, Any] = {"policy": policy, "seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        for metric in ("human_interventions", "human_actions"):
            values = [int(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        item["all_runs_respected_cap"] = all(row["cap_respected"] for row in rows)
        output.append(item)
    return output


def write_csv(out: Path, seed_payloads: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]]) -> None:
    per_seed = [
        {key: value for key, value in row.items() if key != "records"}
        for payload in seed_payloads
        for row in payload["results"]
    ]
    for name, rows in (("per_seed.csv", per_seed), ("aggregate.csv", list(aggregate_rows))):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def write_report(out: Path, rows: Sequence[Mapping[str, Any]], cap: int, episodes: int) -> None:
    lines = [
        "# Uniform online hard-Human-budget evaluation",
        "",
        f"Hard cap: floor(0.20 × {episodes}) = {cap} Human-intervened episodes per method-seed run.",
        "Evaluation order: lexicographic episode_id, identical for every method and seed.",
        "No evaluation outcome was used for fitting, calibration, or action selection.",
        "",
        "| Policy | Repair.SR | Realized HIR | UAR | Unsafe cost | Stop | Return | Regret | Human interventions | Cap respected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        value = lambda metric: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
        lines.append(
            f"| {row['policy']} | {value('repair_sr')} | {value('hir')} | {value('uar')} | "
            f"{value('unsafe_autonomy_cost_rate')} | {value('stopped_rate')} | {value('normalized_return')} | "
            f"{value('collaboration_regret')} | {row['human_interventions_mean']:.1f} "
            f"[{row['human_interventions_min']}, {row['human_interventions_max']}] | "
            f"{'yes' if row['all_runs_respected_cap'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "Static and Bandit artifacts were deterministically reconstructed once because the preceding comparison did not serialize them; "
            "the reconstruction used only immutable training data, fixed seeds, and the previously selected validation calibration scalars. "
            "The hard-cap evaluator itself loaded frozen artifacts and checkpoints and performed no fitting or recalibration.",
            "",
            "This is synthetic development evidence; it is not a locked-holdout or practitioner result.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen SHARE policies under a uniform online Human cap.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--calibration-results", type=Path, default=Path("results/share_budget_matched_20pct_5seed_20260825"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--run-seeds", default=None)
    parser.add_argument("--human-budget", type=float, default=0.20)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    run_seeds = tuple(int(value) for value in args.run_seeds.split(",")) if args.run_seeds else seeds
    if not set(run_seeds).issubset(seeds):
        raise SystemExit("run-seeds must be a subset of frozen seeds")
    rescue_manifest = json.loads((args.checkpoints / "manifest.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads((args.artifacts / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(args.episodes) != rescue_manifest["episode_sha256"] or artifact_manifest["episodes_sha256"] != rescue_manifest["episode_sha256"]:
        raise SystemExit("dataset hash mismatch")
    for seed in seeds:
        artifact = args.artifacts / artifact_manifest["artifacts"][str(seed)]["path"]
        if sha256_file(artifact) != artifact_manifest["artifacts"][str(seed)]["sha256"]:
            raise SystemExit(f"artifact hash mismatch for seed {seed}")
    episodes = load_episodes(args.episodes)
    evaluation = sorted((episode for episode in episodes if episode.split == "test"), key=lambda episode: episode.episode_id)
    if not evaluation:
        raise SystemExit("development evaluation split is empty")
    order = [episode.episode_id for episode in evaluation]
    cap = math.floor(args.human_budget * len(evaluation))
    encoder = FeatureEncoder.load(args.checkpoints / "feature_encoder.json")
    reward = RewardConfig(**rescue_manifest["reward_config"])
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "synthetic_development_hard_cap_evaluation",
        "locked_holdout_accessed": False,
        "development_split_name_in_dataset": "test",
        "development_evaluation_already_inspected": True,
        "seeds": list(seeds),
        "policies": list(POLICIES),
        "human_budget": args.human_budget,
        "evaluation_episodes": len(evaluation),
        "hard_human_episode_cap": cap,
        "budget_unit": "episodes with at least one executed Human intervention",
        "episode_order": order,
        "episode_order_sha256": sha256_json(order),
        "binary_exhaustion_fallback": "AI if feasible, otherwise Stop",
        "value_exhaustion_fallback": "mask Human and select the best remaining AI/Stop under frozen values",
        "fitting_or_recalibration_during_evaluation": False,
        "artifacts_manifest_sha256": sha256_file(args.artifacts / "manifest.json"),
        "rescue_manifest_sha256": sha256_file(args.checkpoints / "manifest.json"),
        "calibration_manifest_sha256": sha256_file(args.calibration_results / "manifest.json"),
        "source_sha256": {
            "scripts/run_share_online_hard_cap.py": sha256_file(Path(__file__)),
            "share/evaluation.py": sha256_file(ROOT / "share/evaluation.py"),
            "share/dqn.py": sha256_file(ROOT / "share/dqn.py"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["protocol_sha256"] != protocol["protocol_sha256"]:
            raise SystemExit("refusing to resume under a changed protocol")
    else:
        atomic_json(manifest_path, protocol)
    for seed in run_seeds:
        path = args.out / f"seed_{seed}.json"
        if path.exists():
            print(json.dumps({"seed": seed, "status": "already complete"}))
            continue
        payload = run_seed(seed, evaluation, cap, args.artifacts, args.checkpoints, args.calibration_results, encoder, reward)
        atomic_json(path, payload)
        print(json.dumps({"seed": seed, "status": "complete"}))
    payloads = []
    for seed in seeds:
        path = args.out / f"seed_{seed}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    complete = len(payloads) == len(seeds)
    if complete:
        rows = aggregate(payloads)
        atomic_json(args.out / "aggregate.json", rows)
        write_csv(args.out, payloads, rows)
        write_report(args.out, rows, cap, len(evaluation))
    print(json.dumps({"out": str(args.out), "complete": complete, "completed_seeds": len(payloads), "cap": cap}))


if __name__ == "__main__":
    main()
