from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.environment import RepairAuthorityEnv
from share.evaluation import evaluate_policy, immediate_reward_tensor
from share.features import FeatureEncoder
from share.policies import ContextualBanditPolicy
from share.types import AuthorityAction, RepairEpisode, RewardConfig


PROTOCOL_VERSION = "share-controller-rescue-20pct-v1"
FIXED_SEEDS = (17, 29, 43, 71, 101)
POLICIES = ("share_bandit", "share_ddqn", "share_h", "full_share")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluation_action_proportions(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    counts = {action: 0 for action in AuthorityAction}
    for record in records:
        for raw_action in record["actions"]:
            counts[AuthorityAction(int(raw_action))] += 1
    total = sum(counts.values())
    return {
        "decision_states": total,
        "p_ai": counts[AuthorityAction.AI] / max(1, total),
        "p_human": counts[AuthorityAction.HUMAN] / max(1, total),
        "p_stop": counts[AuthorityAction.STOP] / max(1, total),
    }


def bandit_diagnostics(
    policy: ContextualBanditPolicy,
    episodes: Sequence[RepairEpisode],
) -> dict[str, float | int | str | None]:
    raw_values: list[np.ndarray] = []
    adjusted_values: list[np.ndarray] = []
    counts = {action: 0 for action in AuthorityAction}
    for episode in episodes:
        env = RepairAuthorityEnv(episode)
        state = env.reset()
        while not env.done:
            raw = policy._raw_values(state)
            adjusted = policy.values(state)
            action = policy.select(state, env.feasible_actions())
            raw_values.append(raw)
            adjusted_values.append(adjusted)
            counts[action] += 1
            result = env.step(action)
            if result.next_state is not None:
                state = result.next_state
    raw_mean = np.mean(np.stack(raw_values), axis=0)
    adjusted_mean = np.mean(np.stack(adjusted_values), axis=0)
    total = sum(counts.values())
    return {
        "epoch": 0,
        "phase": "fitted_validation",
        "value_semantics": "predicted_one_step_base_reward",
        "episodes": len(episodes),
        "decision_states": total,
        "p_ai": counts[AuthorityAction.AI] / max(1, total),
        "p_human": counts[AuthorityAction.HUMAN] / max(1, total),
        "p_stop": counts[AuthorityAction.STOP] / max(1, total),
        "q_ai_raw_mean": float(raw_mean[AuthorityAction.AI]),
        "q_human_raw_mean": float(raw_mean[AuthorityAction.HUMAN]),
        "q_stop_raw_mean": float(raw_mean[AuthorityAction.STOP]),
        "q_ai_adjusted_mean": float(adjusted_mean[AuthorityAction.AI]),
        "q_human_adjusted_mean": float(adjusted_mean[AuthorityAction.HUMAN]),
        "q_stop_adjusted_mean": float(adjusted_mean[AuthorityAction.STOP]),
        "beta_h": None,
        "beta_u": None,
        "human_calibration_penalty": float(policy.human_penalty),
        "ai_calibration_penalty": 0.0,
    }


def dqn_config(
    policy_name: str,
    *,
    seed: int,
    epochs: int,
    human_budget: float,
    unsafe_budget: float,
    device: str,
) -> DQNConfig:
    common = dict(
        training_epochs=epochs,
        human_budget=human_budget,
        unsafe_budget=unsafe_budget,
        seed=seed,
        device=device,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.95,
        replay_capacity=50_000,
        target_update=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        dual_learning_rate_human=5e-3,
        dual_learning_rate_unsafe=5e-3,
        updates_per_epoch=1_000,
        paired_replay=False,
        constraint_confidence_z=0.0,
        double_dqn=True,
    )
    if policy_name == "share_ddqn":
        return DQNConfig(**common, constrained=False, human_constraint=False, unsafe_constraint=False)
    if policy_name == "share_h":
        return DQNConfig(**common, constrained=True, human_constraint=True, unsafe_constraint=False)
    if policy_name == "full_share":
        return DQNConfig(**common, constrained=True, human_constraint=True, unsafe_constraint=True)
    raise ValueError(f"unknown DQN policy {policy_name}")


def run_seed(
    seed: int,
    split: Mapping[str, Sequence[RepairEpisode]],
    encoder: FeatureEncoder,
    reward: RewardConfig,
    *,
    epochs: int,
    human_budget: float,
    unsafe_budget: float,
    device: str,
    out: Path,
) -> dict[str, Any]:
    results = []
    diagnostics: dict[str, list[Mapping[str, Any]]] = {}
    rewards = immediate_reward_tensor(split["train"], reward)

    bandit = ContextualBanditPolicy(encoder, human_budget).fit(
        split["train"], rewards, split["validation"]
    )
    bandit.name = "share_bandit"
    bandit_result = evaluate_policy(bandit, split["test"], reward, human_budget, unsafe_budget)
    bandit_result.update(evaluation_action_proportions(bandit_result["records"]))
    bandit_result.update({"seed": seed, "target_budget": human_budget, "synthetic_evidence": True})
    results.append(bandit_result)
    diagnostics[bandit.name] = [bandit_diagnostics(bandit, split["validation"])]

    for name in POLICIES[1:]:
        config = dqn_config(
            name,
            seed=seed,
            epochs=epochs,
            human_budget=human_budget,
            unsafe_budget=unsafe_budget,
            device=device,
        )
        policy = DQNPolicy(encoder, config, reward)
        policy.name = name
        policy.fit(split["train"], split["validation"])
        if name == "full_share":
            policy.calibrate_constraints(split["validation"], human_budget, unsafe_budget)
        elif name == "share_h":
            policy.calibrate_constraints(split["validation"], human_budget, unsafe_budget)
        else:
            policy.calibrate_human_budget(split["validation"], human_budget)
        post_calibration = policy.policy_diagnostics(
            split["validation"], epoch=epochs, phase="post_calibration"
        )
        history = list(policy.training_history)
        if policy.selected_checkpoint_diagnostics is not None:
            history.append(policy.selected_checkpoint_diagnostics)
        history.append(post_calibration)
        diagnostics[name] = history
        result = evaluate_policy(policy, split["test"], reward, human_budget, unsafe_budget)
        result.update(evaluation_action_proportions(result["records"]))
        result.update(
            {
                "seed": seed,
                "target_budget": human_budget,
                "synthetic_evidence": True,
                "beta_h": float(policy.beta_h),
                "beta_u": float(policy.beta_u),
                "human_calibration_penalty": float(policy.human_calibration_penalty),
                "ai_calibration_penalty": float(policy.ai_calibration_penalty),
                "feasible_validation_checkpoint_found": bool(policy.feasible_checkpoint_found),
                "calibration_validation_hir": float(policy.calibration_validation_hir),
                "calibration_validation_unsafe": float(policy.calibration_validation_unsafe),
            }
        )
        results.append(result)
        policy.save(str(out / f"{name}_seed_{seed}.pt"))
    return {"seed": seed, "results": results, "diagnostics": diagnostics}


def protocol_payload(args: argparse.Namespace, episode_sha256: str) -> dict[str, Any]:
    reward = RewardConfig()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_class": "synthetic_development_diagnostic",
        "development_evaluation_already_inspected": True,
        "paper_results_frozen_during_rescue_run": True,
        "policies": list(POLICIES),
        "seeds": [int(value) for value in args.seeds.split(",")],
        "human_budget": args.human_budget,
        "unsafe_budget_for_reporting_and_full_share": args.unsafe_budget,
        "epochs": args.epochs,
        "device": args.device,
        "episode_sha256": episode_sha256,
        "reward_config": asdict(reward),
        "share_h_definition": "Double DQN with human dual/calibration only; unsafe penalty remains in the unchanged base reward",
        "diagnostic_definition": "Per-epoch greedy validation action proportions and mean raw/adjusted action values; final evaluation action proportions are reported separately",
        "acceptance_gate": {
            "repair_sr_min": 0.92,
            "hir_min": 0.18,
            "hir_max": 0.22,
            "uar_strict_max": 0.12,
            "stopped_rate_strict_max": 0.05,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen SHARE stop-collapse rescue diagnostic.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, FIXED_SEEDS)))
    parser.add_argument("--run-seeds", default=None, help="Optional subset to execute now while keeping the frozen five-seed protocol.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    run_seeds = tuple(int(value) for value in args.run_seeds.split(",")) if args.run_seeds else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be a non-empty unique list")
    if not run_seeds or len(set(run_seeds)) != len(run_seeds) or not set(run_seeds).issubset(seeds):
        raise SystemExit("run-seeds must be a non-empty unique subset of seeds")
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    if not 0.0 <= args.human_budget <= 1.0 or not 0.0 <= args.unsafe_budget <= 1.0:
        raise SystemExit("budgets must lie in [0,1]")
    episodes = load_episodes(args.episodes)
    split = {
        name: [episode for episode in episodes if episode.split == name]
        for name in ("train", "validation", "test")
    }
    if any(not values for values in split.values()):
        raise SystemExit("train, validation, and test splits must all be non-empty")
    encoder = FeatureEncoder.fit(
        [step.state for episode in split["train"] for step in episode.steps]
    )
    reward = RewardConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    protocol = protocol_payload(args, sha256_file(args.episodes))
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_paths = [
        Path(__file__),
        ROOT / "share" / "dqn.py",
        ROOT / "share" / "evaluation.py",
        ROOT / "share" / "policies.py",
        ROOT / "share" / "types.py",
    ]
    manifest = {
        **protocol,
        "protocol_sha256": protocol_sha256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_sha256:
            raise SystemExit("existing output uses a different frozen protocol")
        if existing.get("source_sha256") != manifest["source_sha256"]:
            raise SystemExit("existing output was created from different source hashes")
    else:
        atomic_json(manifest_path, manifest)
        encoder.save(args.out / "feature_encoder.json")

    for seed in run_seeds:
        seed_path = args.out / f"seed_{seed}.json"
        if seed_path.exists():
            continue
        payload = run_seed(
            seed,
            split,
            encoder,
            reward,
            epochs=args.epochs,
            human_budget=args.human_budget,
            unsafe_budget=args.unsafe_budget,
            device=args.device,
            out=args.out,
        )
        atomic_json(seed_path, payload)
        print(json.dumps({"completed_seed": seed, "out": str(seed_path)}), flush=True)

    missing = [seed for seed in seeds if not (args.out / f"seed_{seed}.json").is_file()]
    if missing:
        print(json.dumps({"out": str(args.out), "complete": False, "missing_seeds": missing}, indent=2))
        return
    payloads = [
        json.loads((args.out / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    summary = [
        {key: value for key, value in result.items() if key != "records"}
        for payload in payloads
        for result in payload["results"]
    ]
    atomic_json(args.out / "summary.json", summary)
    aggregate = []
    for policy in POLICIES:
        rows = [row for row in summary if row["policy"] == policy]
        record: dict[str, Any] = {"policy": policy, "seeds": len(rows)}
        for metric in (
            "repair_sr", "hir", "uar", "unsafe_autonomy_cost_rate", "stopped_rate",
            "normalized_return", "p_ai", "p_human", "p_stop",
        ):
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        gate = manifest["acceptance_gate"]
        record["passes_absolute_gate_on_means"] = bool(
            record["repair_sr_mean"] >= gate["repair_sr_min"]
            and gate["hir_min"] <= record["hir_mean"] <= gate["hir_max"]
            and record["uar_mean"] < gate["uar_strict_max"]
            and record["stopped_rate_mean"] < gate["stopped_rate_strict_max"]
        )
        aggregate.append(record)
    atomic_json(args.out / "aggregate.json", aggregate)
    print(json.dumps({"out": str(args.out), "rows": len(summary), "complete": True}, indent=2))


if __name__ == "__main__":
    main()
