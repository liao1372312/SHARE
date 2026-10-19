from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.deployment import (
    LONGITUDINAL_PROTOCOL_VERSION,
    DeploymentEvent,
    chronological_event_windows,
    load_deployment_events,
    sha256_file,
    summarize_observed_events,
    validate_deployment_manifest,
)
from share.dqn import DQNPolicy, feasible_mask
from share.environment import base_reward
from share.evidence_store import ValidatedCorrection, ValidatedEvidenceStore, parse_timestamp
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


def bounded_deployment_updates(policy: DQNPolicy, updates: int) -> int:
    """Run at most the predeclared number of local replay updates."""
    if updates <= 0 or not policy.replay:
        return 0
    original_batch_size = policy.config.batch_size
    policy.config.batch_size = min(original_batch_size, len(policy.replay))
    before = policy.gradient_steps
    try:
        for _ in range(updates):
            policy._optimize()
    finally:
        policy.config.batch_size = original_batch_size
    return policy.gradient_steps - before


def validated_transition(
    policy: DQNPolicy,
    store: ValidatedEvidenceStore,
    visible: Sequence[ValidatedCorrection],
    correction: ValidatedCorrection,
    reward: RewardConfig,
) -> tuple[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray, float, float], dict]:
    """Build a TD transition from one immutable observed deployment event.

    Only Adaptive-arm events are admitted by the caller. No paired outcome or
    policy re-execution is read here: the selected action originates entirely
    from the manifest-bound deployment event stream. Independently adjudicated
    correction fields may supersede the preliminary outcome only in this local
    replay audit; observed-arm summaries retain their original outcomes.
    """
    event = store.observed_event_for(correction)
    if event.policy_mode != "adaptive":
        raise ValueError("only observed adaptive-arm events may train the adaptive replay audit")
    if event.selected_action != correction.observed_action:
        raise ValueError("correction action differs from immutable observed action")
    validated_outcome = correction.validated_outcome
    state = store.feature_adjustment(event.state, visible)
    next_state = store.feature_adjustment(event.next_state, visible) if event.next_state else None
    transition = (
        policy.encoder.transform_one(state),
        int(event.selected_action),
        base_reward(validated_outcome, event.selected_action, reward),
        policy.encoder.transform_one(next_state) if next_state else np.zeros(policy.encoder.dimension, dtype=np.float32),
        validated_outcome.terminal,
        feasible_mask(next_state) if next_state else np.zeros(len(AuthorityAction), dtype=bool),
        float(event.selected_action == AuthorityAction.HUMAN),
        float(validated_outcome.unsafe or (validated_outcome.terminal and event.selected_action == AuthorityAction.AI and not validated_outcome.success)),
    )
    audit = {
        "correction_id": correction.correction_id,
        "episode_id": event.episode_id,
        "step": event.state.step,
        "event_id": event.event_id,
        "selected_action": int(event.selected_action),
        "decision_at": event.decision_at,
        "executed_at": event.executed_at,
        "outcome_observed_at": event.outcome_observed_at,
        "execution_id": event.execution_id,
        "execution_artifact_sha256": event.execution_artifact_sha256,
        "observed_outcome_sha256": event.outcome_sha256,
        "validated_outcome_sha256": correction.validated_outcome_sha256,
        "validation_artifact_sha256": correction.validation_artifact_sha256,
        "correction_digest": correction.correction_digest,
        "validated_success": correction.validated_success,
        "validated_unsafe": correction.validated_unsafe,
        "policy_state_sha256": event.policy_state_sha256,
        "evidence_snapshot_sha256": event.evidence_snapshot_sha256,
    }
    return transition, audit


def append_validated_updates(
    policy: DQNPolicy,
    store: ValidatedEvidenceStore,
    update_cutoff,
    consumed: set[str],
    reward: RewardConfig,
    updates_per_correction: int,
) -> tuple[list[dict], list[dict], int]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    gradient_updates = 0
    visible = store.visible_before(update_cutoff)
    for correction in visible:
        if correction.correction_id in consumed:
            continue
        event = store.observed_event_for(correction)
        if parse_timestamp(event.outcome_observed_at) >= update_cutoff:
            rejected.append({"correction_id": correction.correction_id, "reason": "outcome_not_observed_before_update_cutoff"})
            continue
        if event.policy_mode != "adaptive":
            rejected.append({"correction_id": correction.correction_id, "reason": "frozen_arm_events_cannot_train_adaptive_replay"})
            consumed.add(correction.correction_id)
            continue
        try:
            transition, event_audit = validated_transition(policy, store, visible, correction, reward)
        except ValueError as error:
            rejected.append({"correction_id": correction.correction_id, "reason": str(error)})
            continue
        policy.replay.append(transition)
        gradient_updates += bounded_deployment_updates(policy, updates_per_correction)
        consumed.add(correction.correction_id)
        accepted.append(event_audit)
    return accepted, rejected, gradient_updates


def window_snapshot(store: ValidatedEvidenceStore, window: Sequence[DeploymentEvent]) -> tuple[tuple[ValidatedCorrection, ...], dict]:
    start = min(parse_timestamp(event.decision_at) for event in window)
    visible = store.visible_before(start)
    snapshot = store.snapshot(visible)
    unexpected = [event.event_id for event in window if event.evidence_snapshot_sha256 != snapshot["content_sha256"]]
    if unexpected:
        raise ValueError(
            "deployment event evidence snapshot differs from the pre-window immutable evidence ledger: "
            f"{unexpected[:5]}"
        )
    return visible, snapshot


def validate_observational_arms(window: Sequence[DeploymentEvent]) -> None:
    modes = {event.policy_mode for event in window}
    if modes != {"frozen", "adaptive"}:
        raise ValueError("each chronological window must contain both preassigned frozen and adaptive deployment arms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit observed Frozen/Adaptive SHARE deployment events without counterfactual replay.")
    parser.add_argument("--deployment-events", type=Path, required=True, help="Immutable JSONL of observed single-branch deployment events.")
    parser.add_argument("--deployment-manifest", type=Path, required=True, help="Externally attested manifest that binds events, budgets, policy versions, and provenance references.")
    parser.add_argument("--validated-corrections", type=Path, required=True, help="JSONL of independently validated post-termination corrections; it may be empty.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen checkpoint used only to initialize the local adaptive replay audit.")
    parser.add_argument("--encoder", type=Path, required=True, help="Feature encoder paired with --checkpoint.")
    parser.add_argument("--out", type=Path, default=Path("results/share_longitudinal"))
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--updates-per-correction", type=int, default=20)
    parser.add_argument("--human-budget", type=float, default=0.2)
    parser.add_argument("--unsafe-budget", type=float, default=0.05)
    parser.add_argument("--trusted-checkpoint", action="store_true", help="Allow legacy local checkpoints that require torch weights_only=False.")
    args = parser.parse_args()
    if args.updates_per_correction < 0:
        raise SystemExit("updates per correction must be non-negative")
    if not 0.0 <= args.human_budget <= 1.0 or not 0.0 <= args.unsafe_budget <= 1.0:
        raise SystemExit("human and unsafe budgets must be in [0, 1]")

    reward = RewardConfig()
    events = load_deployment_events(args.deployment_events)
    if {event.split for event in events} != {"test"}:
        raise ValueError("longitudinal deployment analysis accepts only a group-isolated test deployment stream")
    manifest = validate_deployment_manifest(
        args.deployment_manifest,
        events,
        reward,
        args.human_budget,
        args.unsafe_budget,
        args.deployment_events,
        args.encoder,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 not in manifest["allowed_policy_state_sha256"]["frozen"]:
        raise ValueError("frozen checkpoint hash is not declared by the deployment manifest")
    inconsistent_frozen = [event.event_id for event in events if event.policy_mode == "frozen" and event.policy_state_sha256 != checkpoint_sha256]
    if inconsistent_frozen:
        raise ValueError(f"Frozen-arm events must use the declared frozen checkpoint hash: {inconsistent_frozen[:5]}")

    store = ValidatedEvidenceStore.from_jsonl(args.validated_corrections)
    store.bind_deployment_events(events)
    encoder = FeatureEncoder.load(args.encoder)
    frozen_policy = DQNPolicy.load(
        str(args.checkpoint),
        encoder,
        reward,
        trusted_checkpoint=args.trusted_checkpoint,
        require_reward_config=True,
    )
    if frozen_policy.reward_config != reward:
        raise ValueError("frozen checkpoint reward config differs from the longitudinal protocol reward config")
    adaptive_replay_policy = copy.deepcopy(frozen_policy)
    adaptive_replay_policy.name = "adaptive_replay_audit"
    windows = chronological_event_windows(events, args.windows)

    args.out.mkdir(parents=True, exist_ok=True)
    consumed: set[str] = set()
    results = []
    episode_records = []
    audits = []
    for index, window in enumerate(windows, 1):
        validate_observational_arms(window)
        visible, snapshot = window_snapshot(store, window)
        window_start = min(parse_timestamp(event.decision_at) for event in window)
        window_end = max(parse_timestamp(event.outcome_observed_at) for event in window)
        for mode in ("frozen", "adaptive"):
            arm_events = tuple(event for event in window if event.policy_mode == mode)
            result = summarize_observed_events(arm_events, f"observed_{mode}_arm", reward, args.human_budget, args.unsafe_budget)
            results.append({
                **{key: value for key, value in result.items() if key != "records"},
                "round": index,
                "mode": mode,
                "window_start": window_start.isoformat().replace("+00:00", "Z"),
                "window_end": window_end.isoformat().replace("+00:00", "Z"),
                "evidence_snapshot": snapshot,
                "analysis_role": "external-attestation-referenced observational deployment summary",
                "causal_frozen_vs_adaptive_claim_supported": False,
                "provenance_verified_by_runner": False,
                "protocol_only": True,
            })
            episode_records.append({
                "round": index,
                "mode": mode,
                "evidence_snapshot_sha256": snapshot["content_sha256"],
                "records": result["records"],
            })

        next_cutoff = min(parse_timestamp(event.decision_at) for event in windows[index]) if index < len(windows) else None
        local_checkpoint_before = checkpoint_sha256 if index == 1 else sha256_file(args.out / f"adaptive_replay_round_{index - 1}.pt")
        accepted: list[dict] = []
        rejected: list[dict] = []
        gradient_updates = 0
        if next_cutoff is not None:
            accepted, rejected, gradient_updates = append_validated_updates(
                adaptive_replay_policy,
                store,
                next_cutoff,
                consumed,
                reward,
                args.updates_per_correction,
            )
        local_checkpoint_path = args.out / f"adaptive_replay_round_{index}.pt"
        adaptive_replay_policy.save(str(local_checkpoint_path))
        audits.append({
            "round": index,
            "window_event_ids": [event.event_id for event in window],
            "window_episode_ids": sorted({event.episode_id for event in window}),
            "evidence_visible_before_window": snapshot,
            "accepted_observed_adaptive_transitions_after_window": accepted,
            "rejected_corrections_after_window": rejected,
            "adaptive_gradient_updates_after_window": gradient_updates,
            "next_window_cutoff": next_cutoff.isoformat().replace("+00:00", "Z") if next_cutoff else None,
            "local_replay_checkpoint_sha256_before": local_checkpoint_before,
            "local_replay_checkpoint_sha256_after": sha256_file(local_checkpoint_path),
            "local_replay_checkpoint": local_checkpoint_path.name,
        })

    outputs = {
        "longitudinal.json": results,
        "episode_records.json": episode_records,
        "update_audit.json": audits,
    }
    for name, content in outputs.items():
        (args.out / name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    output_hashes = {name: sha256_file(args.out / name) for name in outputs}
    output_hashes.update({path.name: sha256_file(path) for path in sorted(args.out.glob("adaptive_replay_round_*.pt"))})
    audit_manifest = {
        "protocol_version": LONGITUDINAL_PROTOCOL_VERSION,
        "analysis_role": "external-attestation-referenced observational deployment audit",
        "claim_boundary": "The runner validates internal hashes and temporal consistency only. It cannot authenticate the external attestation, prove real-world provenance, or establish a causal Frozen-versus-Adaptive effect from non-randomized observations.",
        "deployment_events_sha256": sha256_file(args.deployment_events),
        "deployment_manifest_sha256": sha256_file(args.deployment_manifest),
        "validated_corrections_sha256": sha256_file(args.validated_corrections),
        "frozen_checkpoint_sha256": checkpoint_sha256,
        "encoder_sha256": sha256_file(args.encoder),
        "reward_config": asdict(reward),
        "human_budget": args.human_budget,
        "unsafe_budget": args.unsafe_budget,
        "windows": args.windows,
        "updates_per_correction": args.updates_per_correction,
        "assignment_protocol": manifest["assignment_protocol"],
        "external_provenance_attestation": manifest["external_provenance_attestation"],
        "prior_unseen_attestation": manifest["prior_unseen_attestation"],
        "code_revision": manifest["code_revision"],
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
        "output_sha256": output_hashes,
        "provenance_verified_by_runner": False,
        "causal_frozen_vs_adaptive_claim_supported": False,
        "protocol_only": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(audit_manifest, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
