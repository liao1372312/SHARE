from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from .types import AuthorityAction, BranchOutcome, RepairState

if TYPE_CHECKING:
    from .deployment import DeploymentEvent


EVIDENCE_SCHEMA_VERSION = 1


def parse_timestamp(value: str, field: str = "timestamp") -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def canonical_timestamp(value: str, field: str = "timestamp") -> str:
    return parse_timestamp(value, field).isoformat().replace("+00:00", "Z")


def correction_digest(value: Mapping[str, object]) -> str:
    canonical = {key: value[key] for key in sorted(value) if key != "correction_digest"}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ValidatedCorrection:
    correction_id: str
    event_id: str
    episode_id: str
    step: int
    observed_action: AuthorityAction
    observed_outcome_sha256: str
    validated_outcome: BranchOutcome
    validated_outcome_sha256: str
    validation_artifact_sha256: str
    validated_success: bool
    validated_unsafe: bool
    validated_at: str
    validator_id: str
    provenance: str
    correction_digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object], line_number: int | None = None) -> "ValidatedCorrection":
        prefix = f" at line {line_number}" if line_number is not None else ""
        required = {
            "correction_id", "event_id", "episode_id", "step", "observed_action", "observed_outcome_sha256",
            "validated_outcome", "validated_outcome_sha256", "validation_artifact_sha256", "validated_success",
            "validated_unsafe", "validated_at", "validator_id", "provenance", "correction_digest",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"validated correction missing fields{prefix}: {missing}")
        try:
            action = AuthorityAction(int(value["observed_action"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"validated correction has invalid observed_action{prefix}") from error
        if action == AuthorityAction.STOP:
            raise ValueError(f"validated correction cannot target Stop{prefix}")
        step = value["step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"validated correction has invalid step{prefix}")
        for field in ("correction_id", "event_id", "episode_id", "validator_id", "provenance", "correction_digest"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"validated correction has empty {field}{prefix}")
        for field in ("observed_outcome_sha256", "validated_outcome_sha256", "validation_artifact_sha256"):
            if not _is_sha256(value[field]):
                raise ValueError(f"validated correction has invalid {field}{prefix}")
        if not isinstance(value["validated_outcome"], Mapping):
            raise ValueError(f"validated correction validated_outcome must be an object{prefix}")
        validated_outcome = BranchOutcome(**value["validated_outcome"])
        if correction_digest(value["validated_outcome"]) != value["validated_outcome_sha256"]:
            raise ValueError(f"validated correction validated outcome digest mismatch{prefix}")
        if not isinstance(value["validated_success"], bool) or not isinstance(value["validated_unsafe"], bool):
            raise ValueError(f"validated correction success/unsafe values must be booleans{prefix}")
        if bool(value["validated_success"]) and bool(value["validated_unsafe"]):
            raise ValueError(f"validated correction cannot be both successful and unsafe{prefix}")
        if (validated_outcome.success, validated_outcome.unsafe) != (value["validated_success"], value["validated_unsafe"]):
            raise ValueError(f"validated correction outcome flags differ from validated_outcome{prefix}")
        if not _is_sha256(value["correction_digest"]) or correction_digest(value) != value["correction_digest"]:
            raise ValueError(f"validated correction digest mismatch{prefix}")
        return cls(
            correction_id=value["correction_id"].strip(),
            event_id=value["event_id"].strip(),
            episode_id=value["episode_id"].strip(),
            step=step,
            observed_action=action,
            observed_outcome_sha256=value["observed_outcome_sha256"].strip(),
            validated_outcome=validated_outcome,
            validated_outcome_sha256=value["validated_outcome_sha256"].strip(),
            validation_artifact_sha256=value["validation_artifact_sha256"].strip(),
            validated_success=bool(value["validated_success"]),
            validated_unsafe=bool(value["validated_unsafe"]),
            validated_at=canonical_timestamp(value["validated_at"], "validated_at"),
            validator_id=value["validator_id"].strip(),
            provenance=value["provenance"].strip(),
            correction_digest=value["correction_digest"].strip(),
        )

    def to_dict(self) -> dict:
        return {
            "correction_id": self.correction_id,
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "observed_action": int(self.observed_action),
            "observed_outcome_sha256": self.observed_outcome_sha256,
            "validated_outcome": {
                "success": self.validated_outcome.success,
                "unsafe": self.validated_outcome.unsafe,
                "quality_delta": self.validated_outcome.quality_delta,
                "latency": self.validated_outcome.latency,
                "execution_cost": self.validated_outcome.execution_cost,
                "human_minutes": self.validated_outcome.human_minutes,
                "next_step": self.validated_outcome.next_step,
                "terminal": self.validated_outcome.terminal,
                "metadata": dict(self.validated_outcome.metadata),
            },
            "validated_outcome_sha256": self.validated_outcome_sha256,
            "validation_artifact_sha256": self.validation_artifact_sha256,
            "validated_success": self.validated_success,
            "validated_unsafe": self.validated_unsafe,
            "validated_at": self.validated_at,
            "validator_id": self.validator_id,
            "provenance": self.provenance,
            "correction_digest": self.correction_digest,
        }


def load_validated_corrections(path: Path) -> list[ValidatedCorrection]:
    if not path.is_file():
        raise FileNotFoundError(path)
    corrections = []
    seen_ids: set[str] = set()
    seen_episode_steps: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            correction = ValidatedCorrection.from_dict(json.loads(line), line_number)
            if correction.correction_id in seen_ids:
                raise ValueError(f"duplicate correction_id at line {line_number}: {correction.correction_id}")
            location = (correction.episode_id, correction.step)
            if location in seen_episode_steps:
                raise ValueError(f"multiple corrections for one episode step at line {line_number}: {location}")
            seen_ids.add(correction.correction_id)
            seen_episode_steps.add(location)
            corrections.append(correction)
    return sorted(corrections, key=lambda item: (parse_timestamp(item.validated_at), item.correction_id))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class ValidatedEvidenceStore:
    """Temporal index of independently validated corrections.

    Corrections are visible only when their validation timestamp precedes the
    current deployment window. This is deliberately separate from paired branch
    outcomes, which may exist for evaluation but are not deployment evidence.
    """

    def __init__(self, corrections: Sequence[ValidatedCorrection]):
        self._corrections = tuple(corrections)
        self._by_location = {(item.episode_id, item.step): item for item in corrections}
        self._contexts: dict[tuple[str, int], tuple[str, str, str]] = {}
        self._events: dict[tuple[str, int], "DeploymentEvent"] = {}

    @classmethod
    def from_jsonl(cls, path: Path) -> "ValidatedEvidenceStore":
        return cls(load_validated_corrections(path))

    def visible_before(self, cutoff: datetime) -> tuple[ValidatedCorrection, ...]:
        normalized = cutoff.astimezone(timezone.utc)
        return tuple(item for item in self._corrections if parse_timestamp(item.validated_at) < normalized)

    def correction_for(self, episode_id: str, step: int) -> ValidatedCorrection | None:
        return self._by_location.get((episode_id, step))

    def bind_deployment_events(self, events: Sequence["DeploymentEvent"]) -> None:
        """Bind corrections to immutable observed actions and observed outcomes.

        This deliberately accepts one deployment branch per event. A correction
        cannot be used before that branch's outcome was observed, and it cannot
        alter the action that the deployed policy actually selected.
        """
        by_location = {(event.episode_id, event.state.step): event for event in events}
        completed_at = {}
        for event in events:
            observed = parse_timestamp(event.outcome_observed_at)
            completed_at[event.episode_id] = max(completed_at.get(event.episode_id, observed), observed)
        contexts = {}
        for correction in self._corrections:
            event = by_location.get((correction.episode_id, correction.step))
            if event is None:
                raise ValueError(
                    "validated correction references unknown observed deployment event: "
                    f"{correction.episode_id}:{correction.step}"
                )
            if correction.observed_action != event.selected_action:
                raise ValueError(
                    "validated correction action differs from observed deployment action: "
                    f"{correction.correction_id}"
                )
            if correction.event_id != event.event_id or correction.observed_outcome_sha256 != event.outcome_sha256:
                raise ValueError(
                    "validated correction does not bind the immutable observed event/outcome: "
                    f"{correction.correction_id}"
                )
            if (correction.validated_outcome.terminal, correction.validated_outcome.next_step) != (event.outcome.terminal, event.outcome.next_step):
                raise ValueError(
                    "validated correction cannot revise the observed transition structure: "
                    f"{correction.correction_id}"
                )
            if parse_timestamp(correction.validated_at, "validated_at") <= completed_at[event.episode_id]:
                raise ValueError(
                    "validated correction must occur after deployment episode termination: "
                    f"{correction.correction_id}"
                )
            contexts[(correction.episode_id, correction.step)] = (
                event.state.gap_type,
                event.state.root_type,
                event.state.domain,
            )
        self._contexts = contexts
        self._events = by_location

    def observed_event_for(self, correction: ValidatedCorrection) -> "DeploymentEvent":
        event = self._events.get((correction.episode_id, correction.step))
        if event is None:
            raise ValueError(f"validated correction has not been bound to an observed event: {correction.correction_id}")
        return event

    def feature_adjustment(self, state: RepairState, visible: Iterable[ValidatedCorrection]) -> RepairState:
        items = tuple(visible)
        if not items:
            return state
        target = (state.gap_type, state.root_type, state.domain)
        same_type = [
            item for item in items
            if item.episode_id != state.episode_id and self._contexts.get((item.episode_id, item.step)) == target
        ]
        if not same_type:
            return state
        unsafe_rate = sum(item.validated_unsafe for item in same_type) / len(same_type)
        failure_rate = sum(not item.validated_success for item in same_type) / len(same_type)
        evidence_uncertainty = 1.0 / sqrt(len(same_type) + 1)
        return RepairState(
            episode_id=state.episode_id,
            step=state.step,
            gap_type=state.gap_type,
            root_type=state.root_type,
            localization_score=state.localization_score,
            repair_utility=state.repair_utility,
            affected_ratio=state.affected_ratio,
            downstream_impact=state.downstream_impact,
            drift_risk=min(1.0, max(state.drift_risk, unsafe_rate)),
            previous_failures=min(1.0, max(state.previous_failures, failure_rate)),
            # Validation evidence reduces epistemic uncertainty; observed risk
            # remains represented independently by the risk/conflict fields.
            uncertainty=min(state.uncertainty, evidence_uncertainty),
            evidence_conflict=min(1.0, max(state.evidence_conflict, unsafe_rate)),
            remaining_budget=state.remaining_budget,
            complexity=state.complexity,
            domain=state.domain,
        )

    @staticmethod
    def snapshot(visible: Sequence[ValidatedCorrection]) -> dict:
        canonical = [item.to_dict() for item in visible]
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "validated_corrections": len(visible),
            "ai_corrections": sum(item.observed_action == AuthorityAction.AI for item in visible),
            "human_corrections": sum(item.observed_action == AuthorityAction.HUMAN for item in visible),
            "unsafe_corrections": sum(item.validated_unsafe for item in visible),
            "content_sha256": digest,
        }
