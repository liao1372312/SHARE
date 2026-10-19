from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import load_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.protocol import GuardPartitions, guard_development_partitions
from share.replay import FrozenLoggedReplay, collect_frozen_logged_replay
from share.safety import certify_group_mean, certify_unsafe_rate
from share.selection import StableSelectionConfig, select_stable_candidate
from share.statistics import cluster_bootstrap_ci, hierarchical_bootstrap_ci, paired_differences
from share.support import JointCalibratedPolicy, SupportCheckedPolicy, SupportMaskedPolicy, SupportRoutedPolicy
from share.types import RepairEpisode, RewardConfig
from share.vector_dqn import VectorDQNConfig, VectorDQNPolicy


PRIMARY_DUALS = "0:0,0:2,0:4,1:0,1:2,1:4,2:2,2:4"
PRIMARY_SELECTION_HUMAN_TARGET = 0.15
PRIMARY_SELECTION_UNSAFE_TARGET = 0.03
PRIMARY_SELECTION_FOLDS = 3
PRIMARY_SELECTION_FRACTION = 0.2
PRIMARY_UPDATES_PER_TRANSITION = 30
PRIMARY_SEEDS = "17,29,43,71,101"
LOCKED_HOLDOUT_PROTOCOL_VERSION = "share-guard-frozen-replay-v2"
LOCKED_COMPARISON_RULES = {
    "repair_sr": {"direction": "higher_is_better", "noninferiority_margin": -0.02},
    "normalized_return": {"direction": "higher_is_better", "noninferiority_margin": -0.05},
    "hir": {"direction": "lower_is_better", "noninferiority_margin": 0.0},
    "unsafe_autonomy_cost_rate": {"direction": "lower_is_better", "noninferiority_margin": 0.0},
    "stopped_rate": {"direction": "lower_is_better", "noninferiority_margin": 0.0},
}


def parse_duals(value: str) -> tuple[tuple[float, float], ...]:
    pairs = []
    for item in value.split(","):
        left, separator, right = item.strip().partition(":")
        if not separator:
            raise ValueError("candidate duals must use human:unsafe pairs")
        human, unsafe = float(left), float(right)
        if human < 0.0 or unsafe < 0.0:
            raise ValueError("candidate duals must be non-negative")
        pairs.append((human, unsafe))
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("candidate duals must be non-empty and unique")
    return tuple(pairs)


def candidate_name(human_dual: float, unsafe_dual: float) -> str:
    return f"h{human_dual:g}_u{unsafe_dual:g}"


def build_scalar_candidates(
    encoder: FeatureEncoder,
    replay: FrozenLoggedReplay,
    reward: RewardConfig,
    seed: int,
    updates: int,
    duals: Sequence[tuple[float, float]],
    support_mode: str = "posthoc_stop",
    minimum_ai_support: int = 3,
    router_fitting: Sequence[RepairEpisode] = (),
    router_human_budget: float = 0.15,
    router_unsafe_budget: float = 0.03,
    router_stop_penalty: float = 0.25,
    router_threshold_grid: int = 33,
    joint_grid_size: int = 9,
    joint_human_budget: float = 0.15,
    joint_unsafe_budget: float = 0.03,
    joint_max_stopped_rate: float = 0.5,
    joint_stop_penalty: float = 0.25,
    joint_human_reserve_rate: float = 0.0,
    device: str = "cpu",
    batch_size: int = 64,
    joint_human_reserve_mode: str = "hash",
    joint_calibration_folds: int = 1,
    joint_human_reserve_grid_size: int = 1,
) -> dict[str, SupportCheckedPolicy | SupportMaskedPolicy | SupportRoutedPolicy | JointCalibratedPolicy]:
    candidates = {}
    for human_dual, unsafe_dual in duals:
        name = candidate_name(human_dual, unsafe_dual)
        config = DQNConfig(
            training_epochs=0,
            seed=seed,
            constrained=False,
            checkpoint_selection=False,
            fixed_human_dual=human_dual,
            fixed_unsafe_dual=unsafe_dual,
            support_mode=support_mode,
            minimum_ai_support=minimum_ai_support,
            device=device,
            batch_size=batch_size,
        )
        policy = DQNPolicy(encoder, config, reward).fit_replay(replay, updates)
        policy.name = f"scalar_discounted_{name}"
        candidates[name] = wrap_supported_policy(
            policy, replay, support_mode, minimum_ai_support, encoder, router_fitting,
            router_human_budget, router_unsafe_budget, router_stop_penalty,
            router_threshold_grid, seed, joint_grid_size, joint_human_budget,
            joint_unsafe_budget, joint_max_stopped_rate, joint_stop_penalty,
            joint_human_reserve_rate,
            joint_human_reserve_mode,
            joint_calibration_folds,
            joint_human_reserve_grid_size,
        )
    return candidates


def build_vector_candidates(
    encoder: FeatureEncoder,
    replay: FrozenLoggedReplay,
    reward: RewardConfig,
    seed: int,
    updates: int,
    duals: Sequence[tuple[float, float]],
    cost_discount: float,
    method_name: str,
    support_mode: str = "posthoc_stop",
    minimum_ai_support: int = 3,
    router_fitting: Sequence[RepairEpisode] = (),
    router_human_budget: float = 0.15,
    router_unsafe_budget: float = 0.03,
    router_stop_penalty: float = 0.25,
    router_threshold_grid: int = 33,
    joint_grid_size: int = 9,
    joint_human_budget: float = 0.15,
    joint_unsafe_budget: float = 0.03,
    joint_max_stopped_rate: float = 0.5,
    joint_stop_penalty: float = 0.25,
    joint_human_reserve_rate: float = 0.0,
    device: str = "cpu",
    batch_size: int = 64,
    joint_human_reserve_mode: str = "hash",
    joint_calibration_folds: int = 1,
    joint_human_reserve_grid_size: int = 1,
) -> dict[str, SupportCheckedPolicy | SupportMaskedPolicy | SupportRoutedPolicy | JointCalibratedPolicy]:
    candidates = {}
    for human_dual, unsafe_dual in duals:
        name = candidate_name(human_dual, unsafe_dual)
        dqn = DQNConfig(training_epochs=0, seed=seed, support_mode=support_mode, minimum_ai_support=minimum_ai_support, device=device, batch_size=batch_size)
        vector_config = VectorDQNConfig(dqn, human_dual, unsafe_dual, cost_discount, cost_discount)
        policy = VectorDQNPolicy(encoder, vector_config, reward).fit_replay(replay, updates)
        policy.name = f"{method_name}_{name}"
        candidates[name] = wrap_supported_policy(
            policy, replay, support_mode, minimum_ai_support, encoder, router_fitting,
            router_human_budget, router_unsafe_budget, router_stop_penalty,
            router_threshold_grid, seed, joint_grid_size, joint_human_budget,
            joint_unsafe_budget, joint_max_stopped_rate, joint_stop_penalty,
            joint_human_reserve_rate,
            joint_human_reserve_mode,
            joint_calibration_folds,
            joint_human_reserve_grid_size,
        )
    return candidates


def wrap_supported_policy(
    policy,
    replay: FrozenLoggedReplay,
    support_mode: str,
    minimum_ai_support: int,
    encoder: FeatureEncoder,
    router_fitting: Sequence[RepairEpisode],
    router_human_budget: float,
    router_unsafe_budget: float,
    router_stop_penalty: float,
    router_threshold_grid: int,
    seed: int,
    joint_grid_size: int,
    joint_human_budget: float,
    joint_unsafe_budget: float,
    joint_max_stopped_rate: float,
    joint_stop_penalty: float,
    joint_human_reserve_rate: float,
    joint_human_reserve_mode: str,
    joint_calibration_folds: int,
    joint_human_reserve_grid_size: int,
) -> SupportCheckedPolicy | SupportMaskedPolicy | SupportRoutedPolicy | JointCalibratedPolicy:
    if support_mode == "posthoc_stop":
        return SupportCheckedPolicy(policy, replay.support_counts, minimum_count=minimum_ai_support)
    if support_mode == "predecision_mask":
        return SupportMaskedPolicy(policy, replay.support_counts, minimum_count=minimum_ai_support)
    if support_mode == "calibrated_router":
        return SupportRoutedPolicy(
            policy,
            replay.support_counts,
            encoder,
            human_budget=router_human_budget,
            unsafe_budget=router_unsafe_budget,
            stop_penalty=router_stop_penalty,
            threshold_grid=router_threshold_grid,
            minimum_count=minimum_ai_support,
            seed=seed,
        ).fit(router_fitting)
    if support_mode == "joint_calibrated":
        return JointCalibratedPolicy(
            policy,
            replay.support_counts,
            human_budget=joint_human_budget,
            unsafe_budget=joint_unsafe_budget,
            max_stopped_rate=joint_max_stopped_rate,
            stop_penalty=joint_stop_penalty,
            grid_size=joint_grid_size,
            minimum_count=minimum_ai_support,
            human_reserve_rate=joint_human_reserve_rate,
            reserve_seed=seed,
            human_reserve_mode=joint_human_reserve_mode,
            calibration_folds=joint_calibration_folds,
            human_reserve_grid_size=joint_human_reserve_grid_size,
        ).fit(router_fitting)
    raise ValueError("support_mode must be posthoc_stop, predecision_mask, calibrated_router, or joint_calibrated")


def summarize(result: Mapping) -> dict:
    return {key: value for key, value in result.items() if key != "records"}


def evaluate_with_diagnostics(
    policy: SupportCheckedPolicy | SupportMaskedPolicy | SupportRoutedPolicy | JointCalibratedPolicy,
    episodes: Sequence[RepairEpisode],
    reward: RewardConfig,
    unsafe_budget: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    policy.reset_diagnostics()
    result = evaluate_policy(policy, episodes, reward, human_budget=0.20, unsafe_budget=unsafe_budget)
    decisions = max(1, int(result["decision_states"]))
    result["unsupported_action_fallback_actions"] = policy.fallback_actions
    result["unsupported_action_fallback_rate"] = policy.fallback_actions / decisions
    result["unsupported_ai_mask_actions"] = policy.masked_ai_actions
    result["unsupported_ai_mask_rate"] = policy.masked_ai_actions / decisions
    if isinstance(policy, SupportRoutedPolicy):
        result["unsupported_ai_route_actions"] = policy.routed_ai_actions
        result["unsupported_ai_route_rate"] = policy.routed_ai_actions / decisions
        result["unsupported_ai_route_human_actions"] = policy.routed_human_actions
        result["unsupported_ai_route_human_rate"] = policy.routed_human_actions / decisions
        result["unsupported_ai_route_stop_actions"] = policy.routed_stop_actions
        result["unsupported_ai_route_stop_rate"] = policy.routed_stop_actions / decisions
        result["support_router_calibration"] = policy.calibration
    if isinstance(policy, JointCalibratedPolicy):
        result["joint_calibration"] = policy.calibration
        result["joint_calibrated_action_rate"] = policy.joint_calibrated_actions / decisions
        result["joint_calibrated_ai_rate"] = policy.joint_action_counts.get(0, 0) / decisions
        result["joint_calibrated_human_rate"] = policy.joint_action_counts.get(1, 0) / decisions
        result["joint_calibrated_stop_rate"] = policy.joint_action_counts.get(2, 0) / decisions
    groups = {episode.episode_id: episode.group_id for episode in episodes}
    result["cluster_bootstrap"] = {
        "unit": "workflow_group",
        "samples": bootstrap_samples,
        "seed": bootstrap_seed,
        "intervals": {
            metric: cluster_bootstrap_ci(result["records"], groups, metric, bootstrap_samples, bootstrap_seed)
            for metric in ("repair_sr", "hir", "unsafe_autonomy_cost_rate", "stopped_rate", "normalized_return")
        },
    }
    return result


def group_assurance_diagnostics(result: Mapping, episodes: Sequence[RepairEpisode], unsafe_budget: float, confidence: float) -> dict:
    records = {str(record["episode_id"]): record for record in result["records"]}
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for episode in episodes:
        groups[episode.group_id].append(records[episode.episode_id])
    unsafe_horizon = max(len(episode.steps) for episode in episodes)
    group_j_u = [float(np.mean([record["unsafe_actions"] for record in rows])) for rows in groups.values()]
    group_hir = [float(np.mean([bool(record["human_used"]) for record in rows])) for rows in groups.values()]
    incidents = sum(any(int(record["unsafe_actions"]) > 0 for record in rows) for rows in groups.values())
    cluster_bound = certify_group_mean(float(np.mean(group_j_u)), len(groups), unsafe_budget, confidence, unsafe_horizon)
    incident_bound = certify_unsafe_rate(incidents, len(groups), unsafe_budget, confidence)
    return {
        "independent_unit": "workflow_group",
        "primary_endpoint_warning": "Episode-weighted cumulative J_U is primary. These group summaries are separate diagnostics and do not certify J_U.",
        "episode_level": summarize(result),
        "group_mean_hir": float(np.mean(group_hir)),
        "group_mean_cumulative_j_u": float(np.mean(group_j_u)),
        "group_mean_cumulative_j_u_bound": asdict(cluster_bound),
        "group_incident_unsafe": {
            "incidents": int(incidents),
            "groups": len(groups),
            "bound": asdict(incident_bound),
            "warning": "GIR_U is distinct from cumulative episode-level J_U.",
        },
    }


def support_summary(replay: FrozenLoggedReplay, minimum_ai_support: int = 3) -> dict:
    action_counts: Counter[str] = Counter()
    bucket_counts: Counter[int] = Counter()
    for (_, action), count in replay.support_counts.items():
        action_counts[str(action)] += count
        bucket_counts[count] += 1
    return {
        "behavior_policy": replay.behavior_policy,
        "behavior_seed": replay.behavior_seed,
        "content_sha256": replay.content_sha256,
        "transitions": len(replay.transitions),
        "unique_support_observations": replay.unique_support_observations,
        "support_by_action": dict(sorted(action_counts.items())),
        "bucket_support_histogram": {str(count): buckets for count, buckets in sorted(bucket_counts.items())},
        "minimum_ai_support": minimum_ai_support,
    }


def partition_summary(partitions: GuardPartitions) -> dict:
    return {
        name: {"episodes": len(items), "groups": len({episode.group_id for episode in items})}
        for name, items in (
            ("fitting", partitions.fitting),
            ("selection", partitions.selection),
            ("development_validation", partitions.certification),
            ("development_test", partitions.test),
        )
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def validate_locked_holdout(
    holdout_path: Path,
    manifest_path: Path,
    holdout: Sequence[RepairEpisode],
    development: Sequence[RepairEpisode],
    historical_identifiers: Mapping | None = None,
) -> dict:
    if not holdout:
        raise ValueError("locked holdout is empty")
    if any(episode.synthetic for episode in holdout):
        raise ValueError("locked holdout requires externally collected paired outcomes with synthetic=false")
    if not holdout_path.is_file():
        raise ValueError("locked holdout must be one JSONL file so its source digest is unambiguous")
    if not manifest_path.is_file():
        raise ValueError(f"locked holdout manifest does not exist: {manifest_path}")
    if historical_identifiers is None:
        raise ValueError("locked holdout validation requires a historical identifier index")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "protocol_version": LOCKED_HOLDOUT_PROTOCOL_VERSION,
        "dataset_sha256": sha256_file(holdout_path),
        "episodes": len(holdout),
        "workflow_groups": len({episode.group_id for episode in holdout}),
        "episode_ids_sha256": identifier_digest([episode.episode_id for episode in holdout]),
        "group_ids_sha256": identifier_digest([episode.group_id for episode in holdout]),
    }
    expected.update({
        "historical_identifier_index_sha256": historical_identifiers["index_sha256"],
        "historical_source_sha256": historical_identifiers["source_sha256"],
        "historical_source_episodes": historical_identifiers["source_episodes"],
        "historical_source_groups": historical_identifiers["source_groups"],
    })
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"locked holdout manifest mismatch for {field}")
    frozen_at = manifest.get("frozen_at")
    if not isinstance(frozen_at, str):
        raise ValueError("locked holdout manifest requires ISO-8601 frozen_at")
    try:
        parsed_frozen_at = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("locked holdout manifest frozen_at is not ISO-8601") from error
    if parsed_frozen_at.tzinfo is None:
        raise ValueError("locked holdout manifest frozen_at must include a UTC offset or Z")
    if not isinstance(manifest.get("prior_unseen_attestation"), str) or not manifest["prior_unseen_attestation"].strip():
        raise ValueError("locked holdout manifest requires a non-empty prior_unseen_attestation")
    if manifest.get("evidence_kind") != "externally_collected_real_paired_outcomes":
        raise ValueError("locked holdout manifest requires externally collected real paired evidence")
    if not isinstance(manifest.get("import_report_sha256"), str) or len(manifest["import_report_sha256"]) != 64:
        raise ValueError("locked holdout manifest requires a bound real paired-outcome import report")
    known_groups = {episode.group_id for episode in development}
    known_episode_ids = {episode.episode_id for episode in development}
    known_groups.update(historical_identifiers["historical_group_ids"])
    known_episode_ids.update(historical_identifiers["historical_episode_ids"])
    overlap = {episode.group_id for episode in holdout} & known_groups
    if overlap:
        raise ValueError(f"locked holdout group leakage with development data: {sorted(overlap)[:5]}")
    episode_overlap = {episode.episode_id for episode in holdout} & known_episode_ids
    if episode_overlap:
        raise ValueError(f"locked holdout episode leakage with development data: {sorted(episode_overlap)[:5]}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "holdout_path": str(holdout_path.resolve()),
        "manifest": manifest,
        "group_overlap": 0,
        "episode_overlap": 0,
        "verification_warning": "The manifest makes the selected file auditable. Its prior-unseen status still requires an external time-stamped attestation.",
    }


def load_historical_identifier_index(path: Path, development_path: Path, development: Sequence[RepairEpisode]) -> dict:
    """Load identifier-only history for holdout leakage checks without outcomes."""
    if not path.is_file():
        raise ValueError(f"historical identifier index does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "purpose", "source_sha256", "source_episodes", "source_groups",
        "development_sha256", "development_episodes", "development_groups",
        "development_episode_ids_sha256", "development_group_ids_sha256", "historical_episode_ids", "historical_group_ids",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"historical identifier index missing fields: {missing}")
    if value["schema_version"] != 1 or value["purpose"] != "SHARE-Guard metadata-only historical identifier index":
        raise ValueError("unsupported historical identifier index schema")
    expected = {
        "development_sha256": sha256_file(development_path),
        "development_episodes": len(development),
        "development_groups": len({episode.group_id for episode in development}),
        "development_episode_ids_sha256": identifier_digest([episode.episode_id for episode in development]),
        "development_group_ids_sha256": identifier_digest([episode.group_id for episode in development]),
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"historical identifier index mismatch for {field}")
    if not all(isinstance(value[field], list) and all(isinstance(item, str) and item for item in value[field]) for field in ("historical_episode_ids", "historical_group_ids")):
        raise ValueError("historical identifier index must contain non-empty string identifiers")
    historical_episode_ids = value["historical_episode_ids"]
    historical_group_ids = value["historical_group_ids"]
    if len(historical_episode_ids) != len(set(historical_episode_ids)) or len(historical_group_ids) != len(set(historical_group_ids)):
        raise ValueError("historical identifier index contains duplicate identifiers")
    if (
        not isinstance(value["source_sha256"], str)
        or len(value["source_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["source_sha256"])
        or value["source_episodes"] != len(historical_episode_ids)
        or value["source_groups"] != len(historical_group_ids)
    ):
        raise ValueError("historical identifier index source metadata is inconsistent")
    development_episode_ids = {episode.episode_id for episode in development}
    development_group_ids = {episode.group_id for episode in development}
    if not development_episode_ids <= set(historical_episode_ids) or not development_group_ids <= set(historical_group_ids):
        raise ValueError("historical identifier index must contain all development identifiers")
    return {
        "index_path": str(path.resolve()),
        "index_sha256": sha256_file(path),
        "source_sha256": value["source_sha256"],
        "source_episodes": value["source_episodes"],
        "source_groups": value["source_groups"],
        "historical_episode_ids": tuple(historical_episode_ids),
        "historical_group_ids": tuple(historical_group_ids),
    }


def diagnostic_partitions(episodes: Sequence[RepairEpisode]) -> GuardPartitions:
    """Validate separately supplied historical diagnostics only when requested."""
    certification = tuple(episode for episode in episodes if episode.split == "validation")
    test = tuple(episode for episode in episodes if episode.split == "test")
    unknown = sorted({episode.split for episode in episodes} - {"validation", "test"})
    if unknown or not certification or not test:
        raise ValueError("diagnostic episodes must contain non-empty validation and test splits only")
    if {episode.group_id for episode in certification} & {episode.group_id for episode in test}:
        raise ValueError("group leakage across diagnostic validation and test partitions")
    return GuardPartitions((), (), certification, test)


def primary_development_protocol(args: argparse.Namespace) -> bool:
    return (
        args.seeds == PRIMARY_SEEDS
        and args.candidate_duals == PRIMARY_DUALS
        and args.selection_fraction == PRIMARY_SELECTION_FRACTION
        and args.selection_folds == PRIMARY_SELECTION_FOLDS
        and args.selection_human_target == PRIMARY_SELECTION_HUMAN_TARGET
        and args.selection_unsafe_target == PRIMARY_SELECTION_UNSAFE_TARGET
        and args.deployment_seed == 17
        and args.unsafe_budget == 0.05
        and args.group_incident_confidence == 0.95
        and args.updates_per_transition == PRIMARY_UPDATES_PER_TRANSITION
        and args.bootstrap_samples == 10_000
        and args.bootstrap_seed == 17
        and not args.paired_replay
        and args.support_mode == "posthoc_stop"
        and args.minimum_ai_support == 3
        and args.device == "cpu"
        and args.batch_size == 64
        and args.joint_human_reserve_mode == "hash"
        and args.joint_calibration_folds == 1
        and args.joint_human_reserve_grid_size == 1
        and args.router_human_budget == PRIMARY_SELECTION_HUMAN_TARGET
        and args.router_unsafe_budget == PRIMARY_SELECTION_UNSAFE_TARGET
        and args.router_stop_penalty == 0.25
        and args.router_threshold_grid == 33
        and args.development_diagnostics_episodes is None
        and args.locked_holdout is None
        and args.locked_holdout_manifest is None
        and args.historical_identifier_index is None
        and not args.sanity
    )


def episode_source_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.jsonl")):
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def protocol_signature(
    args: argparse.Namespace,
    duals: Sequence[tuple[float, float]],
    seeds: Sequence[int],
    holdout_provenance: Mapping | None,
    identifier_provenance: Mapping | None,
) -> dict:
    """Fields that must agree before an interrupted run can be resumed."""
    signature = {
        "development_episodes": str(args.episodes.resolve()),
        "development_episodes_sha256": episode_source_sha256(args.episodes),
        "locked_holdout_manifest_sha256": holdout_provenance["manifest_sha256"] if holdout_provenance else None,
        "historical_identifier_index_sha256": identifier_provenance["index_sha256"] if identifier_provenance else None,
        "development_diagnostics": args.development_diagnostics,
        "development_diagnostics_episodes_sha256": episode_source_sha256(args.development_diagnostics_episodes) if args.development_diagnostics_episodes else None,
        "seeds": list(seeds),
        "deployment_seed": args.deployment_seed,
        "candidate_duals": [list(pair) for pair in duals],
        "selection_fraction": args.selection_fraction,
        "selection_folds": args.selection_folds,
        "selection_human_target": args.selection_human_target,
        "selection_unsafe_target": args.selection_unsafe_target,
        "unsafe_budget": args.unsafe_budget,
        "group_incident_confidence": args.group_incident_confidence,
        "updates_per_transition": args.updates_per_transition,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "paired_replay": args.paired_replay,
        "support_mode": args.support_mode,
        "minimum_ai_support": args.minimum_ai_support,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.support_mode == "calibrated_router":
        signature.update({
            "router_human_budget": args.router_human_budget,
            "router_unsafe_budget": args.router_unsafe_budget,
            "router_stop_penalty": args.router_stop_penalty,
            "router_threshold_grid": args.router_threshold_grid,
        })
    if args.support_mode == "joint_calibrated":
        signature.update({
            "joint_human_budget": args.joint_human_budget,
            "joint_unsafe_budget": args.joint_unsafe_budget,
            "joint_max_stopped_rate": args.joint_max_stopped_rate,
            "joint_stop_penalty": args.joint_stop_penalty,
            "joint_grid_size": args.joint_grid_size,
            "joint_human_reserve_rate": args.joint_human_reserve_rate,
            "joint_human_reserve_mode": args.joint_human_reserve_mode,
            "joint_calibration_folds": args.joint_calibration_folds,
            "joint_human_reserve_grid_size": args.joint_human_reserve_grid_size,
        })
    return signature


def compatible_protocol_signature(prior: Mapping, expected: Mapping) -> bool:
    """Permit resuming a legacy post-hoc ledger without router-only fields."""
    normalized = dict(prior)
    if expected.get("support_mode") == "posthoc_stop":
        normalized.setdefault("support_mode", "posthoc_stop")
        normalized.setdefault("minimum_ai_support", 3)
    return normalized == dict(expected)


def write_payload(path: Path, payload: Mapping) -> None:
    """Atomically replace the progress file after every completed run unit."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def paired_comparison(candidate_records: Sequence[Mapping], reference_records: Sequence[Mapping], episodes: Sequence[RepairEpisode], bootstrap_samples: int, bootstrap_seed: int) -> dict:
    groups = {episode.episode_id: episode.group_id for episode in episodes}
    output = {}
    for metric, rule in LOCKED_COMPARISON_RULES.items():
        differences = paired_differences(candidate_records, reference_records, metric)
        point_estimate = float(np.mean(list(differences.values())))
        low, high = hierarchical_bootstrap_ci(differences, groups, bootstrap_samples, bootstrap_seed)
        if rule["direction"] == "higher_is_better":
            passes_noninferiority = low >= rule["noninferiority_margin"]
        else:
            passes_noninferiority = high <= rule["noninferiority_margin"]
        output[metric] = {
            "point_difference": point_estimate,
            "ci95": [low, high],
            "rule": rule,
            "passes_noninferiority": passes_noninferiority,
        }
    return output


def locked_holdout_comparisons(records: Mapping[str, Sequence[Mapping]], episodes: Sequence[RepairEpisode], bootstrap_samples: int, bootstrap_seed: int) -> dict:
    pairs = (
        ("vector_matched_discount", "scalar_discounted", "head_decomposition"),
        ("share_guard_vector_undiscounted_cost", "vector_matched_discount", "cost_discount_semantics"),
        ("share_guard_vector_undiscounted_cost", "scalar_discounted", "overall_system_difference"),
    )
    output = {}
    for candidate, reference, label in pairs:
        if candidate in records and reference in records:
            output[label] = {
                "candidate": candidate,
                "reference": reference,
                "metrics": paired_comparison(records[candidate], records[reference], episodes, bootstrap_samples, bootstrap_seed),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-replay SHARE-Guard representation diagnostics.")
    parser.add_argument("--episodes", type=Path, default=Path("datasets/share_guard_development/episodes.jsonl"), help="Train-only development JSONL created by create_share_guard_development_data.py.")
    parser.add_argument("--out", type=Path, default=Path("results/share_guard"))
    parser.add_argument("--locked-holdout", type=Path, default=None, help="Previously unseen workflow groups for post-freeze generalization evaluation.")
    parser.add_argument("--locked-holdout-manifest", type=Path, default=None, help="Immutable manifest accompanying --locked-holdout.")
    parser.add_argument("--historical-identifier-index", type=Path, default=None, help="Metadata-only historical identifier index required with a locked holdout.")
    parser.add_argument("--development-diagnostics", action="store_true", help="Allow separately supplied historical validation/test diagnostics; never held-out evidence.")
    parser.add_argument("--development-diagnostics-episodes", type=Path, default=None, help="Separate validation/test-only JSONL, read only with --development-diagnostics.")
    parser.add_argument("--seeds", default=PRIMARY_SEEDS)
    parser.add_argument("--deployment-seed", type=int, default=17)
    parser.add_argument("--shard-only", action="store_true", help="Scheduling-only shard: permit --deployment-seed outside this seed subset; locked-holdout evaluation is forbidden and shards must be merged before interpretation.")
    parser.add_argument("--candidate-duals", default=PRIMARY_DUALS)
    parser.add_argument("--selection-fraction", type=float, default=PRIMARY_SELECTION_FRACTION)
    parser.add_argument("--selection-folds", type=int, default=PRIMARY_SELECTION_FOLDS)
    parser.add_argument("--selection-human-target", type=float, default=PRIMARY_SELECTION_HUMAN_TARGET)
    parser.add_argument("--selection-unsafe-target", type=float, default=PRIMARY_SELECTION_UNSAFE_TARGET)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--group-incident-confidence", type=float, default=0.95)
    parser.add_argument("--updates-per-transition", type=int, default=PRIMARY_UPDATES_PER_TRANSITION)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--paired-replay", action="store_true", help="Counterfactual replay ablation; default is candidate-independent uniform logged behavior.")
    parser.add_argument("--support-mode", choices=("posthoc_stop", "predecision_mask", "calibrated_router", "joint_calibrated"), default="posthoc_stop", help="posthoc_stop preserves G002; other choices are fitting-only development redesigns.")
    parser.add_argument("--minimum-ai-support", type=int, default=3)
    parser.add_argument("--device", default="cpu", help="PyTorch training device: cpu, cuda, or cuda:<index>.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--router-human-budget", type=float, default=PRIMARY_SELECTION_HUMAN_TARGET)
    parser.add_argument("--router-unsafe-budget", type=float, default=PRIMARY_SELECTION_UNSAFE_TARGET)
    parser.add_argument("--router-stop-penalty", type=float, default=0.25)
    parser.add_argument("--router-threshold-grid", type=int, default=33)
    parser.add_argument("--joint-human-budget", type=float, default=PRIMARY_SELECTION_HUMAN_TARGET)
    parser.add_argument("--joint-unsafe-budget", type=float, default=PRIMARY_SELECTION_UNSAFE_TARGET)
    parser.add_argument("--joint-max-stopped-rate", type=float, default=0.50)
    parser.add_argument("--joint-stop-penalty", type=float, default=0.25)
    parser.add_argument("--joint-grid-size", type=int, default=9)
    parser.add_argument("--joint-human-reserve-rate", type=float, default=0.0, help="Deterministic episode-level Human reserve used only by joint calibration.")
    parser.add_argument("--joint-human-reserve-mode", choices=("hash", "risk"), default="hash")
    parser.add_argument("--joint-calibration-folds", type=int, default=1)
    parser.add_argument("--joint-human-reserve-grid-size", type=int, default=1, help="Fit-only grid from zero to --joint-human-reserve-rate; one preserves a fixed rate.")
    parser.add_argument("--sanity", action="store_true", help="Run one seed, two duals, two updates per transition, and no evaluation outside selection.")
    parser.add_argument("--resume", action="store_true", help="Continue a compatible interrupted run from results.json.")
    args = parser.parse_args()
    if args.updates_per_transition < 0:
        raise SystemExit("updates per transition must be non-negative")
    if args.minimum_ai_support <= 0:
        raise SystemExit("minimum AI support must be positive")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.device.startswith("cuda") and not __import__("torch").cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not 0.0 <= args.router_human_budget <= 1.0 or not 0.0 <= args.router_unsafe_budget <= 1.0:
        raise SystemExit("router budgets must be in [0, 1]")
    if args.router_stop_penalty < 0.0 or args.router_threshold_grid < 2:
        raise SystemExit("router stop penalty must be non-negative and threshold grid at least two")
    if not 0.0 <= args.joint_human_budget <= 1.0 or not 0.0 <= args.joint_unsafe_budget <= 1.0:
        raise SystemExit("joint calibration budgets must be in [0, 1]")
    if not 0.0 <= args.joint_max_stopped_rate <= 1.0:
        raise SystemExit("joint maximum stopped rate must be in [0, 1]")
    if args.joint_stop_penalty < 0.0 or args.joint_grid_size < 2:
        raise SystemExit("joint stop penalty must be non-negative and grid size at least two")
    if not 0.0 <= args.joint_human_reserve_rate <= 1.0:
        raise SystemExit("joint human reserve rate must be in [0, 1]")
    if args.joint_calibration_folds < 1:
        raise SystemExit("joint calibration folds must be positive")
    if args.joint_human_reserve_grid_size < 1:
        raise SystemExit("joint human reserve grid size must be positive")
    episodes = load_episodes(args.episodes)
    partitions = guard_development_partitions(episodes, args.deployment_seed, args.selection_fraction)
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    if args.deployment_seed not in seeds and not args.shard_only:
        raise SystemExit("deployment seed must be included in --seeds")
    duals = parse_duals(args.candidate_duals)
    if args.sanity:
        seeds, duals, args.updates_per_transition = (seeds[0],), duals[:2], 2
    selection_config = StableSelectionConfig(
        human_operating_target=args.selection_human_target,
        unsafe_operating_target=args.selection_unsafe_target,
        folds=args.selection_folds,
        seed=args.deployment_seed,
        max_stopped_rate=args.joint_max_stopped_rate if args.support_mode == "joint_calibrated" else None,
    )
    encoder = FeatureEncoder.fit([step.state for episode in partitions.fitting for step in episode.steps])
    reward = RewardConfig()
    if bool(args.locked_holdout) != bool(args.locked_holdout_manifest) or bool(args.locked_holdout) != bool(args.historical_identifier_index):
        raise SystemExit("--locked-holdout, --locked-holdout-manifest, and --historical-identifier-index must be supplied together")
    if args.development_diagnostics != bool(args.development_diagnostics_episodes):
        raise SystemExit("--development-diagnostics and --development-diagnostics-episodes must be supplied together")
    locked_holdout = load_episodes(args.locked_holdout) if args.locked_holdout else ()
    if args.shard_only and locked_holdout:
        raise SystemExit("--shard-only cannot evaluate a locked holdout")
    diagnostics = diagnostic_partitions(load_episodes(args.development_diagnostics_episodes)) if args.development_diagnostics else None
    if diagnostics and ({episode.group_id for episode in episodes} & ({episode.group_id for episode in diagnostics.certification} | {episode.group_id for episode in diagnostics.test})):
        raise ValueError("development diagnostics group leakage with train-only Guard development data")
    identifier_provenance = load_historical_identifier_index(args.historical_identifier_index, args.episodes, episodes) if args.historical_identifier_index else None
    holdout_provenance = None
    if locked_holdout:
        holdout_provenance = validate_locked_holdout(args.locked_holdout, args.locked_holdout_manifest, locked_holdout, episodes, identifier_provenance)
    signature = protocol_signature(args, duals, seeds, holdout_provenance, identifier_provenance)
    methods = (
        ("scalar_discounted", lambda replay, seed: build_scalar_candidates(encoder, replay, reward, seed, args.updates_per_transition * len(replay.transitions), duals, args.support_mode, args.minimum_ai_support, partitions.fitting, args.router_human_budget, args.router_unsafe_budget, args.router_stop_penalty, args.router_threshold_grid, args.joint_grid_size, args.joint_human_budget, args.joint_unsafe_budget, args.joint_max_stopped_rate, args.joint_stop_penalty, args.joint_human_reserve_rate, args.device, args.batch_size, args.joint_human_reserve_mode, args.joint_calibration_folds, args.joint_human_reserve_grid_size)),
        ("vector_matched_discount", lambda replay, seed: build_vector_candidates(encoder, replay, reward, seed, args.updates_per_transition * len(replay.transitions), duals, 0.95, "vector_matched_discount", args.support_mode, args.minimum_ai_support, partitions.fitting, args.router_human_budget, args.router_unsafe_budget, args.router_stop_penalty, args.router_threshold_grid, args.joint_grid_size, args.joint_human_budget, args.joint_unsafe_budget, args.joint_max_stopped_rate, args.joint_stop_penalty, args.joint_human_reserve_rate, args.device, args.batch_size, args.joint_human_reserve_mode, args.joint_calibration_folds, args.joint_human_reserve_grid_size)),
        ("share_guard_vector_undiscounted_cost", lambda replay, seed: build_vector_candidates(encoder, replay, reward, seed, args.updates_per_transition * len(replay.transitions), duals, 1.0, "share_guard_vector_undiscounted_cost", args.support_mode, args.minimum_ai_support, partitions.fitting, args.router_human_budget, args.router_unsafe_budget, args.router_stop_penalty, args.router_threshold_grid, args.joint_grid_size, args.joint_human_budget, args.joint_unsafe_budget, args.joint_max_stopped_rate, args.joint_stop_penalty, args.joint_human_reserve_rate, args.device, args.batch_size, args.joint_human_reserve_mode, args.joint_calibration_folds, args.joint_human_reserve_grid_size)),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "results.json"
    existing = {}
    development = {}
    locked_evaluation: dict = {}
    locked_records: dict[str, list[Mapping]] = {}
    if args.resume:
        if not result_path.is_file():
            raise SystemExit(f"cannot resume: {result_path} does not exist")
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if not compatible_protocol_signature(prior.get("protocol_signature", {}), signature):
            raise SystemExit("cannot resume: results.json protocol signature differs from this invocation")
        existing = {(str(run["method"]), int(run["seed"])): run for run in prior.get("runs", [])}
        development = dict(prior.get("development_diagnostics", {}))
        locked_evaluation = dict(prior.get("locked_holdout_evaluation", {}))
        locked_records = {key: list(value) for key, value in prior.get("locked_holdout_records", {}).items()}

    def payload_for(runs: Sequence[Mapping]) -> dict:
        return {
            "evidence_warning": "All current SHARE episodes are synthetic. Outputs are engineering diagnostics, not deployment-safety or practitioner evidence.",
            "primary_development_protocol": primary_development_protocol(args),
            "locked_holdout_protocol": {
                "protocol_version": LOCKED_HOLDOUT_PROTOCOL_VERSION,
                "estimand": "only the predesignated deployment seed is evaluated; other seeds report selection stability only",
                "deployment_seed": args.deployment_seed,
                "comparison_rules": LOCKED_COMPARISON_RULES,
                "difference_direction": "candidate_minus_reference",
                "provenance": holdout_provenance,
                "historical_identifier_index": identifier_provenance,
            } if holdout_provenance else None,
            "protocol_signature": signature,
            "protocol": {
                "environment": "common unshielded original AI/Human/Stop transition kernel",
                "primary_estimand": "episode-weighted cumulative unsafe autonomous actions J_U",
                "replay": "paired_counterfactual_ablation" if args.paired_replay else "candidate-independent_uniform_feasible_logged_behavior",
                "comparison": {
                    "scalar_discounted": "one head; reward, human cost, and unsafe cost all have gamma=0.95",
                    "vector_matched_discount": "three heads; all heads have gamma=0.95",
                    "share_guard_vector_undiscounted_cost": "three heads; reward gamma=0.95 and both cost heads gamma=1",
                },
                "selection": asdict(selection_config),
                "action_tie_break": "Stop, then Human, then AI for exactly equal scalar values in learning and execution",
                "support_rule": "posthoc_stop: replace an unsupported proposed AI action with Stop; predecision_mask: remove unsupported AI before policy comparison so Human and Stop remain comparable; calibrated_router: only unsupported proposed AI actions are routed to a fitting-only calibrated Human/Stop classifier; joint_calibrated: fitting-only joint AI/Human action-penalty calibration with unsupported AI infeasible and Stop explicitly penalized in the objective. The same fixed fitting replay table is used for every method and dual at a seed.",
                "support_mode": args.support_mode,
                "minimum_ai_support": args.minimum_ai_support,
                "support_router": {
                    "fitting_only": True,
                    "human_budget": args.router_human_budget,
                    "unsafe_budget": args.router_unsafe_budget,
                    "stop_penalty": args.router_stop_penalty,
                    "threshold_grid": args.router_threshold_grid,
                },
                "joint_calibration": {
                    "fitting_only": True,
                    "human_budget": args.joint_human_budget,
                    "unsafe_budget": args.joint_unsafe_budget,
                    "max_stopped_rate": args.joint_max_stopped_rate,
                    "stop_penalty": args.joint_stop_penalty,
                    "grid_size": args.joint_grid_size,
                    "human_reserve_rate": args.joint_human_reserve_rate,
                    "human_reserve_mode": args.joint_human_reserve_mode,
                    "calibration_folds": args.joint_calibration_folds,
                    "human_reserve_grid_size": args.joint_human_reserve_grid_size,
                },
                "deployment_seed": args.deployment_seed,
                "development_role": "not evaluated unless --development-diagnostics is explicit; never labeled held-out evidence",
                "locked_holdout_role": "evaluated only after selection when a non-overlapping --locked-holdout is supplied",
            },
            "partitions": {
                **partition_summary(partitions),
                "development_diagnostics": partition_summary(diagnostics) if diagnostics else None,
            },
            "candidate_duals": [{"human_dual": human, "unsafe_dual": unsafe} for human, unsafe in duals],
            "runs": sorted(runs, key=lambda run: (int(run["seed"]), str(run["method"]))),
            "development_diagnostics": development,
            "locked_holdout_evaluation": locked_evaluation,
            "locked_holdout_paired_comparisons": locked_holdout_comparisons(locked_records, locked_holdout, args.bootstrap_samples, args.bootstrap_seed) if locked_holdout else {},
            "locked_holdout_records": locked_records,
            "python": platform.python_version(),
        }

    encoder.save(args.out / "feature_encoder.json")
    for seed in seeds:
        replay = collect_frozen_logged_replay(partitions.fitting, encoder, reward, behavior_seed=seed, paired_counterfactual=args.paired_replay)
        for method, builder in methods:
            if (method, seed) in existing:
                continue
            candidates = builder(replay, seed)
            selection = select_stable_candidate(candidates, partitions.selection, selection_config, reward)
            run = {
                "method": method,
                "seed": seed,
                "selected_candidate": selection.selected,
                "selection": selection.to_dict(),
                "replay": support_summary(replay, args.minimum_ai_support),
                "status": "no_stable_candidate" if selection.selected is None else "selected",
            }
            if args.support_mode == "calibrated_router":
                run["support_router_calibration"] = {
                    name: candidate.calibration
                    for name, candidate in candidates.items()
                    if isinstance(candidate, SupportRoutedPolicy)
                }
            if args.support_mode == "joint_calibrated":
                run["joint_calibration"] = {
                    name: candidate.calibration
                    for name, candidate in candidates.items()
                    if isinstance(candidate, JointCalibratedPolicy)
                }
            if selection.selected is not None and diagnostics:
                validation_result = evaluate_with_diagnostics(candidates[selection.selected], diagnostics.certification, reward, args.unsafe_budget, args.bootstrap_samples, args.bootstrap_seed)
                test_result = evaluate_with_diagnostics(candidates[selection.selected], diagnostics.test, reward, args.unsafe_budget, args.bootstrap_samples, args.bootstrap_seed)
                development[f"{method}:seed-{seed}"] = {
                    "warning": "Iterative development diagnostics: these data informed method redesign and are not new held-out evidence.",
                    "validation": summarize(validation_result),
                    "test": summarize(test_result),
                }
                if seed == args.deployment_seed:
                    development[f"{method}:seed-{seed}"]["group_assurance_diagnostics"] = group_assurance_diagnostics(validation_result, diagnostics.certification, args.unsafe_budget, args.group_incident_confidence)
            existing[(method, seed)] = run
            write_payload(result_path, payload_for(list(existing.values())))

    # The locked holdout is a single predeclared comparison: either all three
    # selected deployment policies are evaluated together, or none is exposed.
    if locked_holdout and not locked_evaluation:
        deployment_runs = {method: existing.get((method, args.deployment_seed)) for method, _ in methods}
        missing = [method for method, run in deployment_runs.items() if run is None or run["selected_candidate"] is None]
        if missing:
            locked_evaluation = {
                "status": "not_evaluated",
                "reason": "all_predeclared_methods_must_select_a_stable_deployment_candidate",
                "missing_methods": missing,
                "deployment_seed": args.deployment_seed,
                "methods": {},
            }
        else:
            replay = collect_frozen_logged_replay(partitions.fitting, encoder, reward, behavior_seed=args.deployment_seed, paired_counterfactual=args.paired_replay)
            locked_methods = {}
            for method, builder in methods:
                candidates = builder(replay, args.deployment_seed)
                selected = str(deployment_runs[method]["selected_candidate"])
                result = evaluate_with_diagnostics(candidates[selected], locked_holdout, reward, args.unsafe_budget, args.bootstrap_samples, args.bootstrap_seed)
                locked_methods[method] = summarize(result)
                locked_records[method] = result["records"]
            locked_evaluation = {
                "status": "evaluated",
                "deployment_seed": args.deployment_seed,
                "methods": locked_methods,
            }
    payload = payload_for(list(existing.values()))
    write_payload(result_path, payload)
    print(json.dumps({"out": str(args.out), "runs": len(existing), "selected": sum(run["selected_candidate"] is not None for run in existing.values()), "locked_evaluated": len(locked_evaluation)}, indent=2))


if __name__ == "__main__":
    main()
