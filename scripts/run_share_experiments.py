from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy, immediate_reward_tensor
from share.features import FeatureEncoder
from share.policies import (
    AlwaysHumanPolicy,
    AutonomousPolicy,
    ContextualBanditPolicy,
    RandomBudgetPolicy,
    SupervisedBenefitPolicy,
    ThresholdPolicy,
    LoggedAssessmentPolicy,
)
from share.types import RepairEpisode, RewardConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate SHARE authority-allocation policies.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/share_experiments"))
    parser.add_argument("--budgets", default="0,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--primary-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--constraint-confidence-z", type=float, default=0.0, help="Validation-only safety margin; use 1.64 for a conservative operating point.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    parser.add_argument("--dual-learning-rate-human", type=float, default=5e-3)
    parser.add_argument("--dual-learning-rate-unsafe", type=float, default=5e-3)
    parser.add_argument("--updates-per-epoch", type=int, default=1_000)
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or cuda:<index>.")
    parser.add_argument("--paired-replay", action="store_true", help="Ablation: train from all paired counterfactual branches instead of logged exploratory transitions.")
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--sanity", action="store_true", help="Run one seed, one budget, and short training.")
    parser.add_argument("--llm-assessments", type=Path, default=None, help="Optional held-out JSONL scores for the LLM Self-Assessment baseline.")
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    split = {name: [episode for episode in episodes if episode.split == name] for name in ["train", "validation", "test"]}
    if any(not values for values in split.values()):
        raise SystemExit(f"all splits must be non-empty: { {key: len(value) for key, value in split.items()} }")
    train_states = [step.state for episode in split["train"] for step in episode.steps]
    encoder = FeatureEncoder.fit(train_states)
    reward_config = RewardConfig()
    budgets = [args.primary_budget] if args.sanity else [float(value) for value in args.budgets.split(",")]
    seeds = [int(args.seeds.split(",")[0])] if args.sanity else [int(value) for value in args.seeds.split(",")]
    epochs = min(args.epochs, 2) if args.sanity else args.epochs
    args.out.mkdir(parents=True, exist_ok=True)
    encoder.save(args.out / "feature_encoder.json")
    all_results: List[Dict] = []
    for budget in budgets:
        for seed in seeds:
            dqn_options = {
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "replay_capacity": args.replay_capacity,
                "target_update": args.target_update,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay_steps": args.epsilon_decay_steps,
                "dual_learning_rate_human": args.dual_learning_rate_human,
                "dual_learning_rate_unsafe": args.dual_learning_rate_unsafe,
                "updates_per_epoch": args.updates_per_epoch,
                "device": args.device,
                "paired_replay": args.paired_replay,
            }
            policies = build_policies(encoder, split, reward_config, budget, args.unsafe_budget, epochs, args.batch_size, seed, args.sanity, args.llm_assessments, args.constraint_confidence_z, dqn_options)
            for policy in policies:
                result = evaluate_policy(policy, split["test"], reward_config, budget, args.unsafe_budget)
                result.update({"target_budget": budget, "seed": seed, "synthetic_evidence": all(item.synthetic for item in split["test"])})
                if isinstance(policy, DQNPolicy):
                    result.update({
                        "beta_h": policy.beta_h,
                        "beta_u": policy.beta_u,
                        "human_calibration_penalty": policy.human_calibration_penalty,
                        "ai_calibration_penalty": policy.ai_calibration_penalty,
                        "feasible_validation_checkpoint_found": policy.feasible_checkpoint_found,
                        "calibration_human_target": policy.calibration_human_target,
                        "calibration_unsafe_target": policy.calibration_unsafe_target,
                        "calibration_validation_hir": policy.calibration_validation_hir,
                        "calibration_validation_unsafe": policy.calibration_validation_unsafe,
                        "replay_protocol": "paired_counterfactual_ablation" if policy.config.paired_replay else "exploratory_logged_transitions",
                        "human_constraint_enabled": policy.config.constrained and policy.config.human_constraint,
                        "unsafe_constraint_enabled": policy.config.constrained and policy.config.unsafe_constraint,
                    })
                detail_path = args.out / f"{policy.name}_budget_{budget:g}_seed_{seed}.json"
                detail_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                summary = {key: value for key, value in result.items() if key != "records"}
                all_results.append(summary)
                if isinstance(policy, DQNPolicy):
                    policy.save(str(args.out / f"{policy.name}_budget_{budget:g}_seed_{seed}.pt"))
                write_outputs(args.out, all_results, split, encoder, args.episodes, vars(args))
    write_outputs(args.out, all_results, split, encoder, args.episodes, vars(args))
    print(json.dumps({"out": str(args.out), "runs": len(all_results), "splits": {key: len(value) for key, value in split.items()}}, indent=2))


def build_policies(encoder: FeatureEncoder, split: Dict[str, Sequence[RepairEpisode]], reward_config: RewardConfig, budget: float, unsafe_budget: float, epochs: int, batch_size: int, seed: int, sanity: bool, llm_assessments: Path | None = None, constraint_confidence_z: float = 0.0, dqn_options: Dict | None = None):
    rewards = immediate_reward_tensor(split["train"], reward_config)
    policies = [
        AutonomousPolicy(),
        AlwaysHumanPolicy(),
        RandomBudgetPolicy(budget, seed),
        ThresholdPolicy("uncertainty", validation_threshold(split["validation"], "uncertainty", budget), True, "confidence_threshold"),
        ThresholdPolicy("drift_risk", validation_threshold(split["validation"], "drift_risk", budget), True, "risk_rule"),
        SupervisedBenefitPolicy(encoder, budget, seed).fit(split["train"], split["validation"]),
        ContextualBanditPolicy(encoder, budget).fit(split["train"], rewards, split["validation"]),
    ]
    if llm_assessments is not None:
        logged = LoggedAssessmentPolicy.from_jsonl(llm_assessments)
        episode_scores = [max(logged.escalation_score(step.state) for step in episode.steps) for episode in split["validation"]]
        logged.threshold = validation_score_threshold(episode_scores, budget)
        policies.append(logged)
    common = {"training_epochs": epochs, "batch_size": batch_size, "human_budget": budget, "unsafe_budget": unsafe_budget, "seed": seed, "constraint_confidence_z": constraint_confidence_z, **(dqn_options or {})}
    configs = [
        ("vanilla_dqn", DQNConfig(**common, constrained=False, double_dqn=False)),
        ("unconstrained_double_dqn", DQNConfig(**common, constrained=False, double_dqn=True)),
        ("share_controller", DQNConfig(**common, constrained=True, double_dqn=True)),
    ]
    if sanity:
        configs = configs[-1:]
        policies = policies[:3]
    for name, config in configs:
        policy = DQNPolicy(encoder, config, reward_config)
        policy.name = name
        policy.fit(split["train"], split["validation"])
        if config.constrained:
            policy.calibrate_constraints(split["validation"], budget, unsafe_budget)
        else:
            policy.calibrate_human_budget(split["validation"], budget)
        policies.append(policy)
    return policies


def validation_threshold(episodes: Sequence[RepairEpisode], feature: str, budget: float) -> float:
    values = sorted(max(float(getattr(step.state, feature)) for step in episode.steps) for episode in episodes)
    return validation_score_threshold(values, budget)


def validation_score_threshold(values: Sequence[float], budget: float) -> float:
    values = sorted(values)
    if budget <= 0:
        return float("inf")
    if budget >= 1:
        return float("-inf")
    return values[min(len(values) - 1, int((1.0 - budget) * len(values)))]


def write_outputs(out: Path, rows: List[Dict], split: Dict[str, Sequence[RepairEpisode]], encoder: FeatureEncoder, episode_path: Path | None = None, run_config: Dict | None = None) -> None:
    (out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    source_files = [
        Path(__file__).resolve(),
        ROOT / "share" / "data.py",
        ROOT / "share" / "dqn.py",
        ROOT / "share" / "environment.py",
        ROOT / "share" / "evaluation.py",
        ROOT / "share" / "features.py",
        ROOT / "share" / "policies.py",
        ROOT / "share" / "replay.py",
        ROOT / "share" / "types.py",
    ]
    manifest = {
        "evidence_warning": "Results are synthetic when synthetic_evidence=true and must not be reported as practitioner outcomes.",
        "splits": {key: {"episodes": len(value), "states": sum(len(item.steps) for item in value)} for key, value in split.items()},
        "feature_dimension": encoder.dimension,
        "episode_sha256": sha256_file(episode_path) if episode_path and episode_path.is_file() else None,
        "run_config": {key: str(value) if isinstance(value, Path) else value for key, value in (run_config or {}).items()},
        "python": platform.python_version(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_files
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
