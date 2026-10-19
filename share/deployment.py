from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .environment import base_reward
from .evidence_store import canonical_timestamp, parse_timestamp
from .types import AuthorityAction, BranchOutcome, RepairState, RewardConfig


DEPLOYMENT_EVENT_SCHEMA_VERSION = 2
DEPLOYMENT_EVENT_KIND = "observed_deployment"
DEPLOYMENT_MANIFEST_SCHEMA_VERSION = 1
DEPLOYMENT_MANIFEST_KIND = "share_longitudinal_deployment_manifest"
LONGITUDINAL_PROTOCOL_VERSION = "share-longitudinal-observed-events-v2"


def canonical_digest(value: Mapping, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    canonical = {key: value[key] for key in sorted(value) if key not in excluded}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def identifier_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class DeploymentEvent:
    event_id: str
    episode_id: str
    group_id: str
    split: str
    source: str
    decision_at: str
    executed_at: str
    outcome_observed_at: str
    execution_id: str
    execution_artifact_sha256: str
    policy_mode: str
    policy_state_sha256: str
    evidence_snapshot_sha256: str
    state: RepairState
    selected_action: AuthorityAction
    outcome: BranchOutcome
    next_state: RepairState | None
    outcome_sha256: str
    event_digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object], line_number: int | None = None) -> "DeploymentEvent":
        prefix = f" at line {line_number}" if line_number is not None else ""
        required = {
            "schema_version", "evidence_kind", "event_id", "episode_id", "group_id", "split", "source",
            "decision_at", "executed_at", "outcome_observed_at", "execution_id", "execution_artifact_sha256",
            "policy_mode", "policy_state_sha256", "evidence_snapshot_sha256", "state", "selected_action",
            "outcome", "next_state", "outcome_sha256", "event_digest",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"deployment event missing fields{prefix}: {missing}")
        if value["schema_version"] != DEPLOYMENT_EVENT_SCHEMA_VERSION or value["evidence_kind"] != DEPLOYMENT_EVENT_KIND:
            raise ValueError(f"unsupported deployment event schema{prefix}")
        for field in ("event_id", "episode_id", "group_id", "source", "execution_id"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"deployment event has empty {field}{prefix}")
        for field in ("execution_artifact_sha256", "policy_state_sha256", "evidence_snapshot_sha256", "outcome_sha256", "event_digest"):
            if not is_sha256(value[field]):
                raise ValueError(f"deployment event has invalid {field}{prefix}")
        if canonical_digest(value, {"event_digest"}) != value["event_digest"]:
            raise ValueError(f"deployment event digest mismatch{prefix}")
        if value["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"deployment event has invalid split{prefix}")
        if value["policy_mode"] not in {"frozen", "adaptive"}:
            raise ValueError(f"deployment event has invalid policy_mode{prefix}")
        decision_at = canonical_timestamp(value["decision_at"], "decision_at")
        executed_at = canonical_timestamp(value["executed_at"], "executed_at")
        outcome_observed_at = canonical_timestamp(value["outcome_observed_at"], "outcome_observed_at")
        if not (parse_timestamp(decision_at) <= parse_timestamp(executed_at) <= parse_timestamp(outcome_observed_at)):
            raise ValueError(f"deployment event timestamps are not ordered{prefix}")
        try:
            action = AuthorityAction(int(value["selected_action"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"deployment event has invalid selected_action{prefix}") from error
        if not isinstance(value["state"], Mapping) or not isinstance(value["outcome"], Mapping):
            raise ValueError(f"deployment event state and outcome must be objects{prefix}")
        state = RepairState(**value["state"])
        outcome = BranchOutcome(**value["outcome"])
        if canonical_digest(value["outcome"]) != value["outcome_sha256"]:
            raise ValueError(f"deployment event outcome digest mismatch{prefix}")
        if state.episode_id != value["episode_id"]:
            raise ValueError(f"deployment event state episode_id mismatch{prefix}")
        if action != AuthorityAction.AI and not outcome.terminal:
            raise ValueError(f"only AI deployment actions may be nonterminal{prefix}")
        next_state = None
        if outcome.terminal:
            if outcome.next_step is not None or value["next_state"] is not None:
                raise ValueError(f"terminal deployment event cannot carry next state{prefix}")
        else:
            if outcome.next_step is None or not isinstance(value["next_state"], Mapping):
                raise ValueError(f"nonterminal deployment event requires a next state{prefix}")
            next_state = RepairState(**value["next_state"])
            if next_state.episode_id != state.episode_id or next_state.step != outcome.next_step:
                raise ValueError(f"deployment event next state mismatch{prefix}")
        return cls(
            event_id=value["event_id"].strip(),
            episode_id=value["episode_id"].strip(),
            group_id=value["group_id"].strip(),
            split=value["split"],
            source=value["source"].strip(),
            decision_at=decision_at,
            executed_at=executed_at,
            outcome_observed_at=outcome_observed_at,
            execution_id=value["execution_id"].strip(),
            execution_artifact_sha256=value["execution_artifact_sha256"].strip(),
            policy_mode=value["policy_mode"],
            policy_state_sha256=value["policy_state_sha256"].strip(),
            evidence_snapshot_sha256=value["evidence_snapshot_sha256"].strip(),
            state=state,
            selected_action=action,
            outcome=outcome,
            next_state=next_state,
            outcome_sha256=value["outcome_sha256"].strip(),
            event_digest=value["event_digest"].strip(),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": DEPLOYMENT_EVENT_SCHEMA_VERSION,
            "evidence_kind": DEPLOYMENT_EVENT_KIND,
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "split": self.split,
            "source": self.source,
            "decision_at": self.decision_at,
            "executed_at": self.executed_at,
            "outcome_observed_at": self.outcome_observed_at,
            "execution_id": self.execution_id,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "policy_mode": self.policy_mode,
            "policy_state_sha256": self.policy_state_sha256,
            "evidence_snapshot_sha256": self.evidence_snapshot_sha256,
            "state": asdict(self.state),
            "selected_action": int(self.selected_action),
            "outcome": asdict(self.outcome),
            "next_state": asdict(self.next_state) if self.next_state else None,
            "outcome_sha256": self.outcome_sha256,
            "event_digest": self.event_digest,
        }


def load_deployment_events(path: Path) -> list[DeploymentEvent]:
    if not path.is_file():
        raise FileNotFoundError(path)
    events = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = DeploymentEvent.from_dict(json.loads(line), line_number)
            if event.event_id in seen_ids:
                raise ValueError(f"duplicate deployment event_id at line {line_number}: {event.event_id}")
            seen_ids.add(event.event_id)
            events.append(event)
    if not events:
        raise ValueError("deployment event log is empty")
    validate_deployment_events(events)
    return sorted(events, key=lambda item: (parse_timestamp(item.decision_at), item.event_id))


def episode_events(events: Sequence[DeploymentEvent]) -> dict[str, tuple[DeploymentEvent, ...]]:
    grouped: dict[str, list[DeploymentEvent]] = defaultdict(list)
    for event in events:
        grouped[event.episode_id].append(event)
    return {
        episode_id: tuple(sorted(rows, key=lambda item: (item.state.step, parse_timestamp(item.decision_at), item.event_id)))
        for episode_id, rows in grouped.items()
    }


def validate_deployment_events(events: Sequence[DeploymentEvent]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    execution_ids = set()
    for event in events:
        group_splits[event.group_id].add(event.split)
        if event.execution_id in execution_ids:
            raise ValueError(f"deployment event reuses immutable execution_id: {event.execution_id}")
        execution_ids.add(event.execution_id)
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"deployment workflow groups cross splits: {leaked[:5]}")
    for episode_id, rows in episode_events(events).items():
        first = rows[0]
        if any(
            (row.group_id, row.split, row.policy_mode, row.source, row.policy_state_sha256, row.evidence_snapshot_sha256)
            != (first.group_id, first.split, first.policy_mode, first.source, first.policy_state_sha256, first.evidence_snapshot_sha256)
            for row in rows
        ):
            raise ValueError(f"deployment episode has inconsistent metadata: {episode_id}")
        if rows[0].state.step != 0:
            raise ValueError(f"deployment episode does not start at step zero: {episode_id}")
        for index, row in enumerate(rows):
            if row.state.step != index:
                raise ValueError(f"deployment episode has non-contiguous steps: {episode_id}")
            if index and parse_timestamp(row.decision_at) < parse_timestamp(rows[index - 1].outcome_observed_at):
                raise ValueError(f"deployment episode decision precedes prior observed outcome: {episode_id}")
            if index < len(rows) - 1:
                if row.outcome.terminal or row.next_state is None:
                    raise ValueError(f"deployment episode terminates before its final row: {episode_id}")
                if row.next_state != rows[index + 1].state:
                    raise ValueError(f"deployment episode next state does not match subsequent row: {episode_id}")
            elif not row.outcome.terminal:
                raise ValueError(f"deployment episode is missing terminal outcome: {episode_id}")


def chronological_event_windows(events: Sequence[DeploymentEvent], windows: int) -> tuple[tuple[DeploymentEvent, ...], ...]:
    if windows < 2:
        raise ValueError("windows must be at least two")
    episodes = list(episode_events(events).values())
    if len(episodes) < windows:
        raise ValueError("deployment log must contain at least one episode per window")
    ordered = sorted(episodes, key=lambda rows: (parse_timestamp(rows[0].decision_at), rows[0].episode_id))
    quotient, remainder = divmod(len(ordered), windows)
    output = []
    index = 0
    for window in range(windows):
        size = quotient + int(window < remainder)
        output.append(tuple(event for rows in ordered[index:index + size] for event in rows))
        index += size
    for previous, current in zip(output, output[1:]):
        previous_end = max(parse_timestamp(event.outcome_observed_at) for event in previous)
        current_start = min(parse_timestamp(event.decision_at) for event in current)
        if previous_end >= current_start:
            raise ValueError("deployment episode windows overlap; use non-overlapping completed deployment windows")
    return tuple(output)


def summarize_observed_events(events: Sequence[DeploymentEvent], policy_name: str, reward: RewardConfig, human_budget: float, unsafe_budget: float) -> dict:
    grouped = episode_events(events)
    records = []
    for episode_id, rows in sorted(grouped.items()):
        actions = [int(row.selected_action) for row in rows]
        rewards = [base_reward(row.outcome, row.selected_action, reward) for row in rows]
        unsafe_actions = sum(row.outcome.unsafe or (row.outcome.terminal and row.selected_action == AuthorityAction.AI and not row.outcome.success) for row in rows)
        records.append({
            "episode_id": episode_id,
            "group_id": rows[0].group_id,
            "actions": actions,
            "rewards": rewards,
            "success": rows[-1].outcome.success,
            "unsafe": bool(unsafe_actions),
            "unsafe_actions": int(unsafe_actions),
            "human_used": any(row.selected_action == AuthorityAction.HUMAN for row in rows),
            "human_minutes": sum(row.outcome.human_minutes for row in rows),
            "latency": sum(row.outcome.latency for row in rows),
            "execution_cost": sum(row.outcome.execution_cost for row in rows),
            "event_ids": [row.event_id for row in rows],
        })
    count = len(records)
    autonomous_completed = [record for record in records if not record["human_used"] and record["actions"][-1] != int(AuthorityAction.STOP)]
    repair_sr = float(np.mean([record["success"] for record in records]))
    hir = float(np.mean([record["human_used"] for record in records]))
    unsafe_rate = float(sum(record["unsafe_actions"] for record in records) / count)
    return {
        "policy": policy_name,
        "episodes": count,
        "decision_states": sum(len(record["actions"]) for record in records),
        "repair_sr": repair_sr,
        "hir": hir,
        "uar": float(np.mean([record["unsafe"] or not record["success"] for record in autonomous_completed])) if autonomous_completed else 0.0,
        "unsafe_autonomy_cost_rate": unsafe_rate,
        "stopped_rate": float(np.mean([record["actions"][-1] == int(AuthorityAction.STOP) for record in records])),
        "normalized_return": float(np.mean([sum(record["rewards"]) for record in records])),
        "human_minutes_per_100": float(np.mean([record["human_minutes"] for record in records]) * 100.0),
        "latency": float(np.mean([record["latency"] for record in records])),
        "execution_cost": float(np.mean([record["execution_cost"] for record in records])),
        "human_budget_violation": max(0.0, hir - human_budget),
        "unsafe_budget_violation": max(0.0, unsafe_rate - unsafe_budget),
        "records": records,
    }


def deployment_manifest_payload(
    events: Sequence[DeploymentEvent],
    frozen_at: str,
    prior_unseen_attestation: str,
    external_provenance_attestation: str,
    assignment_protocol: str,
    allowed_policy_state_sha256: Mapping[str, Sequence[str]],
    reward: RewardConfig,
    human_budget: float,
    unsafe_budget: float,
    code_revision: str,
    deployment_events_sha256: str,
    encoder_sha256: str,
) -> dict:
    return {
        "schema_version": DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        "kind": DEPLOYMENT_MANIFEST_KIND,
        "protocol_version": LONGITUDINAL_PROTOCOL_VERSION,
        "deployment_events_sha256": deployment_events_sha256,
        "event_count": len(events),
        "event_ids_sha256": identifier_digest([event.event_id for event in events]),
        "episode_ids_sha256": identifier_digest(list(episode_events(events))),
        "frozen_at": canonical_timestamp(frozen_at, "frozen_at"),
        "prior_unseen_attestation": prior_unseen_attestation,
        "external_provenance_attestation": external_provenance_attestation,
        "assignment_protocol": assignment_protocol,
        "allowed_policy_state_sha256": {mode: sorted(values) for mode, values in sorted(allowed_policy_state_sha256.items())},
        "reward_config": asdict(reward),
        "human_budget": human_budget,
        "unsafe_budget": unsafe_budget,
        "code_revision": code_revision,
        "encoder_sha256": encoder_sha256,
    }


def validate_deployment_manifest(
    path: Path,
    events: Sequence[DeploymentEvent],
    reward: RewardConfig,
    human_budget: float,
    unsafe_budget: float,
    events_path: Path,
    encoder_path: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "kind", "protocol_version", "deployment_events_sha256", "event_count", "event_ids_sha256",
        "episode_ids_sha256", "frozen_at", "prior_unseen_attestation", "external_provenance_attestation",
        "assignment_protocol", "allowed_policy_state_sha256", "reward_config", "human_budget", "unsafe_budget",
        "code_revision", "encoder_sha256", "manifest_digest",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"deployment manifest missing fields: {missing}")
    if (value["schema_version"], value["kind"], value["protocol_version"]) != (
        DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        DEPLOYMENT_MANIFEST_KIND,
        LONGITUDINAL_PROTOCOL_VERSION,
    ):
        raise ValueError("deployment manifest schema or protocol version is unsupported")
    if not is_sha256(value["manifest_digest"]) or canonical_digest(value, {"manifest_digest"}) != value["manifest_digest"]:
        raise ValueError("deployment manifest digest mismatch")
    for field in ("prior_unseen_attestation", "external_provenance_attestation", "assignment_protocol", "code_revision"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"deployment manifest has empty {field}")
    if value["deployment_events_sha256"] != sha256_file(events_path):
        raise ValueError("deployment event file hash differs from deployment manifest")
    if not is_sha256(value["encoder_sha256"]) or value["encoder_sha256"] != sha256_file(encoder_path):
        raise ValueError("feature encoder hash differs from deployment manifest")
    if value["event_count"] != len(events):
        raise ValueError("deployment event count differs from deployment manifest")
    if value["event_ids_sha256"] != identifier_digest([event.event_id for event in events]):
        raise ValueError("deployment event IDs differ from deployment manifest")
    if value["episode_ids_sha256"] != identifier_digest(list(episode_events(events))):
        raise ValueError("deployment episode IDs differ from deployment manifest")
    frozen_at = parse_timestamp(value["frozen_at"], "manifest frozen_at")
    if any(parse_timestamp(event.decision_at) <= frozen_at for event in events):
        raise ValueError("every deployment decision must occur after manifest frozen_at")
    if value["reward_config"] != asdict(reward):
        raise ValueError("reward config differs from deployment manifest")
    if value["human_budget"] != human_budget or value["unsafe_budget"] != unsafe_budget:
        raise ValueError("budget values differ from deployment manifest")
    allowed = value["allowed_policy_state_sha256"]
    if not isinstance(allowed, Mapping) or set(allowed) != {"frozen", "adaptive"}:
        raise ValueError("deployment manifest must declare frozen and adaptive policy hashes")
    for mode, hashes in allowed.items():
        if not isinstance(hashes, list) or not hashes or any(not is_sha256(item) for item in hashes):
            raise ValueError(f"deployment manifest has invalid {mode} policy hashes")
    for event in events:
        if event.policy_state_sha256 not in allowed[event.policy_mode]:
            raise ValueError(f"deployment event references undeclared {event.policy_mode} policy state: {event.event_id}")
    return value
