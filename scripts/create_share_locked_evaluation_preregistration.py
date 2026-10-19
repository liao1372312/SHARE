from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.dqn import DQNPolicy
from share.features import FeatureEncoder
from share.types import RewardConfig


SEEDS = (17, 29, 43, 71, 101)
POINTS = (0.10, 0.20, 0.30)
POLICIES = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share")


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_state() -> dict[str, Any]:
    command = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    if command.returncode:
        return {"git_commit": None, "available": False, "reason": command.stderr.strip()}
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True)
    return {"git_commit": command.stdout.strip(), "available": True, "dirty": bool(dirty.stdout.strip())}


def file_item(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"required frozen file missing: {path}")
    return {"path": str(resolved), "role": role, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def comparator_parameters(path: Path) -> dict[str, Any]:
    pack = joblib.load(path)
    conf, static, bandit = (pack[name] for name in ("share_conf", "share_static", "share_bandit"))
    return {
        "share_conf": {"feature": conf.feature, "threshold": float(conf.threshold), "high_means_human": bool(conf.high_means_human)},
        "share_static": {
            "threshold": float(static.threshold), "intervention_rate": float(static.intervention_rate),
            "estimator_class": type(static.model).__name__, "estimator_params": static.model.get_params(),
        },
        "share_bandit": {
            "human_penalty": float(bandit.human_penalty), "intervention_rate": float(bandit.intervention_rate),
            "estimator_classes": [type(model).__name__ for model in bandit.models],
        },
    }


def checkpoint_parameters(path: Path, encoder: FeatureEncoder, reward: RewardConfig) -> dict[str, Any]:
    policy = DQNPolicy.load(str(path), encoder, reward, require_reward_config=True)
    return {
        "config": vars(policy.config),
        "beta_h": float(policy.beta_h), "beta_u": float(policy.beta_u),
        "human_calibration_penalty": float(policy.human_calibration_penalty),
        "ai_calibration_penalty": float(policy.ai_calibration_penalty),
        "calibration_human_target": float(policy.calibration_human_target),
        "calibration_unsafe_target": float(policy.calibration_unsafe_target),
        "calibration_validation_hir": float(policy.calibration_validation_hir),
        "calibration_validation_unsafe": float(policy.calibration_validation_unsafe),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze SHARE artifacts and seal the locked-holdout pre-analysis plan without reading outcomes.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locked-holdout-manifest", type=Path, default=None, help="Metadata-only locked manifest to bind; the outcome JSONL is never accepted by this command.")
    parser.add_argument("--primary-comparators", type=Path, default=Path("results/share_budget_matched_frozen_artifacts_20260825"))
    parser.add_argument("--primary-calibration", type=Path, default=Path("results/share_budget_matched_20pct_5seed_20260825"))
    parser.add_argument("--primary-checkpoints", type=Path, default=Path("results/share_controller_rescue_20pct_5seed_20260825"))
    parser.add_argument("--curve-artifacts", type=Path, default=Path("results/share_hard_cap_curve_artifacts_20260825"))
    parser.add_argument("--development", type=Path, default=Path("datasets/share_synthetic/episodes.jsonl"))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to overwrite an existing preregistration directory")
    args.out.mkdir(parents=True, exist_ok=True)
    encoder_path = args.primary_checkpoints / "feature_encoder.json"
    encoder = FeatureEncoder.load(encoder_path)
    primary_manifest = json.loads((args.primary_checkpoints / "manifest.json").read_text(encoding="utf-8"))
    reward = RewardConfig(**primary_manifest["reward_config"])
    files: list[dict[str, Any]] = []
    operating_points: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    for budget in POINTS:
        key = f"{budget:.1f}"
        operating_points[key] = {}
        parameters[key] = {}
        if budget == 0.20:
            manifest_paths = [
                args.primary_comparators / "manifest.json", args.primary_calibration / "manifest.json",
                args.primary_checkpoints / "manifest.json",
            ]
        else:
            budget_dir = args.curve_artifacts / ("budget_0p1" if budget == 0.10 else "budget_0p3")
            manifest_paths = [budget_dir / "manifest.json"]
        for manifest_path in manifest_paths:
            files.append(file_item(manifest_path, f"artifact_manifest_budget_{key}"))
        for seed in SEEDS:
            if budget == 0.20:
                comparator = args.primary_comparators / f"comparators_seed_{seed}.joblib"
                ddqn = args.primary_checkpoints / f"share_ddqn_seed_{seed}.pt"
                share = args.primary_checkpoints / f"share_h_seed_{seed}.pt"
                calibration = args.primary_calibration / f"seed_{seed}.json"
            else:
                budget_dir = args.curve_artifacts / ("budget_0p1" if budget == 0.10 else "budget_0p3")
                comparator = budget_dir / f"comparators_seed_{seed}.joblib"
                ddqn = budget_dir / f"share_ddqn_seed_{seed}.pt"
                share = budget_dir / f"share_h_seed_{seed}.pt"
                calibration = None
            entries = {
                "comparators": file_item(comparator, f"comparators_budget_{key}_seed_{seed}"),
                "share_ddqn": file_item(ddqn, f"ddqn_budget_{key}_seed_{seed}"),
                "share": file_item(share, f"share_budget_{key}_seed_{seed}"),
            }
            files.extend(entries.values())
            if calibration:
                entries["calibration"] = file_item(calibration, f"calibration_budget_{key}_seed_{seed}")
                files.append(entries["calibration"])
            operating_points[key][str(seed)] = entries
            parameters[key][str(seed)] = json_safe({
                "comparators": comparator_parameters(comparator),
                "share_ddqn": checkpoint_parameters(ddqn, encoder, reward),
                "share": checkpoint_parameters(share, encoder, reward),
            })
    source_paths = [
        ROOT / "scripts/run_share_locked_holdout_once.py", ROOT / "scripts/run_share_online_hard_cap.py",
        ROOT / "scripts/run_share_guard_experiments.py", ROOT / "share/data.py", ROOT / "share/dqn.py",
        ROOT / "share/environment.py", ROOT / "share/evaluation.py", ROOT / "share/features.py",
        ROOT / "share/policies.py", ROOT / "share/types.py",
    ]
    for path in source_paths:
        files.append(file_item(path, "frozen_evaluation_source"))
    files.extend([
        file_item(encoder_path, "normalization_and_feature_encoder"),
        file_item(args.development, "development_dataset_binding"),
        file_item(Path("results/share_final_development_ablation_20pct_20260826/method_freeze.json"), "method_freeze_manifest"),
        file_item(Path("results/share_final_development_ablation_20pct_20260826/manifest.json"), "final_development_ablation_manifest"),
    ])
    deduplicated = {item["path"]: item for item in files}
    files = [deduplicated[path] for path in sorted(deduplicated)]
    source_digest = sha256_json({item["path"]: item["sha256"] for item in files if item["role"] == "frozen_evaluation_source"})
    artifact_lock = {
        "schema_version": 1, "created_before_holdout_access": True,
        "files": files, "operating_points": operating_points,
        "serialized_parameters": parameters,
        "encoder": file_item(encoder_path, "normalization_and_feature_encoder"),
        "encoder_statistics": {
            "gap_types": encoder.gap_types, "root_types": encoder.root_types, "domains": encoder.domains,
            "mean": encoder.mean.tolist(), "scale": encoder.scale.tolist(), "disabled_groups": list(encoder.disabled_groups),
        },
        "reward_config": vars(reward), "seeds": list(SEEDS), "source_tree_digest": source_digest,
        "git": git_state(),
    }
    artifact_lock_path = args.out / "artifact_lock.json"
    atomic_json(artifact_lock_path, artifact_lock)
    holdout_manifest_sha = None
    if args.locked_holdout_manifest:
        if not args.locked_holdout_manifest.is_file():
            raise SystemExit("--locked-holdout-manifest does not exist")
        holdout_manifest_sha = sha256_file(args.locked_holdout_manifest)
    preregistration = {
        "schema_version": 1,
        "title": "SHARE one-shot locked-holdout evaluation pre-analysis plan",
        "status": "sealed_and_dataset_bound" if holdout_manifest_sha else "sealed_protocol_awaiting_locked_manifest_binding",
        "created_before_holdout_outcome_access": True,
        "design": "fixed-model evaluation on externally collected paired workflow outcomes; not an RCT",
        "registration": "local immutable pre-analysis manifest; AEA RCT registration not applicable",
        "sample_rule": "include every valid episode in the supplied, provenance-validated locked manifest; no outcome-based exclusions",
        "power_rule": "fixed holdout size; no post-hoc power or MDE gate; report prespecified confidence intervals",
        "locked_holdout_manifest_sha256": holdout_manifest_sha,
        "primary_operating_point": 0.20,
        "secondary_operating_points": [0.10, 0.30],
        "budget_rule": "cap=floor(B*N) episodes with at least one executed Human action; one global online ledger per method-seed run",
        "episode_order": "lexicographic episode_id; identical across every method, seed, and budget",
        "hard_cap_evaluator": "exact OnlineHardHumanCapPolicy frozen in artifact_lock; binary Conf/Static fall back to AI then Stop; value policies mask Human and replan over AI/Stop",
        "primary_methods": list(POLICIES),
        "primary_metrics": {
            "repair_sr": "mean final success over all episodes; higher is better",
            "hir": "fraction of episodes with at least one executed Human action; lower is better",
            "uar": "unsafe-or-failed fraction among no-Human episodes ending in a non-Stop action; lower is better; report denominator",
            "stopped_rate": "fraction of episodes whose final action is Stop; lower is better",
        },
        "secondary_metrics": {
            "normalized_return": "mean undiscounted episode reward sum; higher is better",
            "collaboration_regret": "mean visited-state oracle return minus selected return using gamma=.95; lower is better",
        },
        "primary_comparisons": "each of Conf, Static, Bandit, and DDQN minus SHARE for each primary metric (16-test family)",
        "inference": {
            "point_estimate": "mean of five paired seed differences",
            "confidence_interval": "95% percentile workflow-group cluster bootstrap; resample groups with replacement, retain all episodes and seed pairing",
            "bootstrap_samples": 10000, "bootstrap_seed": 20260826,
            "p_value": "two-sided exact sign-flip test over five paired seed differences",
            "multiplicity": "Holm adjustment across all 16 primary comparator-metric tests",
            "minimum_two_sided_seed_sign_flip_p": 0.0625,
            "pareto_rule": "maximize Repair.SR and minimize realized HIR, UAR, and Stop; nondominance uses aggregate point estimates without rounding",
        },
        "secondary_analyses": {
            "budgets": "repeat frozen five-method evaluation at 10% and 30%; descriptive mean/SD, cap audit, and Pareto status",
            "voh": "at 20%, within each seed rank raw Q(H)-Q(AI) into five bins using predictions only; relate to terminal Human reward minus recursively continued AI return at gamma=.95; report Spearman, bin slope, group-bootstrap CI, and one-sided exact seed sign-flip p",
        },
        "exploratory_only": {
            "complexity": ["C_low: x<.5", "C_mid: .5<=x<.8", "C_high: x>=.8"],
            "initial_horizon": ["H1", "H2", "H3", "other observed lengths retained and labeled"],
            "rule": "all interactions/subgroups are exploratory and cannot replace primary conclusions",
        },
        "missing_invalid_data": "abort before policy evaluation if provenance, schema, synthetic=false, hash, or historical-disjointness validation fails; otherwise no exclusions or imputation",
        "one_shot_rule": "write an unblinding-attempt ledger before reading the outcome JSONL and refuse any second attempt",
        "prohibited_actions": ["training", "recalibration", "parameter changes", "threshold selection", "checkpoint selection", "outcome-based exclusions", "rerunning after results"],
        "reporting_rule": "report preregistered primary results first even if null or unfavorable; disclose every deviation",
        "artifact_lock_sha256": sha256_file(artifact_lock_path),
        "git_commit": artifact_lock["git"]["git_commit"],
        "git_unavailable_substitute": source_digest if not artifact_lock["git"]["available"] else None,
    }
    prereg_path = args.out / "preregistration.json"
    atomic_json(prereg_path, preregistration)
    seal = {
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "holdout_outcomes_accessed": False,
        "dataset_bound": holdout_manifest_sha is not None,
        "artifact_lock_sha256": sha256_file(artifact_lock_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "combined_seal_sha256": sha256_json({"artifact_lock": sha256_file(artifact_lock_path), "preregistration": sha256_file(prereg_path)}),
    }
    atomic_json(args.out / "seal.json", seal)
    print(json.dumps({"out": str(args.out), "dataset_bound": seal["dataset_bound"], "files": len(files), "git": artifact_lock["git"]}))


if __name__ == "__main__":
    main()
