"""Input binding and outcome definitions for the Repair-Grounded paper protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .data import load_episodes
from .types import AuthorityAction, RepairEpisode


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identifier_digest(values: Sequence[str]) -> str:
    encoded = json.dumps(sorted(values), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bound_file(entry: dict[str, Any], base: Path) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("a required artifact path is still pending")
    path = (base / entry["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    return path


def load_observed_episodes(path: Path, import_report: Path) -> list[RepairEpisode]:
    """Recheck supplied branch artifacts; this does not authenticate their authors."""
    from scripts.import_share_real_paired_episodes import load_artifact_ledger, validate_real_paired_episodes

    report = json.loads(import_report.read_text(encoding="utf-8"))
    if report.get("evidence_kind") != "externally_collected_real_paired_outcomes":
        raise ValueError("an observed paired-outcome import report is required")
    if report.get("output_sha256") != sha256_file(path):
        raise ValueError("episode data do not match their import report")
    ledger_path = Path(report["artifact_ledger_path"])
    if sha256_file(ledger_path) != report.get("artifact_ledger_sha256"):
        raise ValueError("artifact ledger differs from the import report")
    episodes = load_episodes(path)
    if not episodes:
        raise ValueError("the supplied episode file is empty")
    ledger = load_artifact_ledger(ledger_path, Path(report["artifact_root"]))
    validate_real_paired_episodes(episodes, ledger)
    for episode in episodes:
        if type(episode.metadata.get("initial_unsafe")) is not bool:
            raise ValueError(f"{episode.episode_id} requires an explicit initial_unsafe flag from its trace")
        for index, step in enumerate(episode.steps):
            if step.state.step != index or step.state.episode_id != episode.episode_id:
                raise ValueError("state identifiers or step order do not match their episode")
            if not all(math.isfinite(v) for v in step.state.continuous()):
                raise ValueError("state coordinates must be finite")
            for action, outcome in step.outcomes.items():
                if type(outcome.success) is not bool or type(outcome.unsafe) is not bool:
                    raise ValueError("success and unsafe labels must be booleans")
                if action in {"human", "stop"} and not outcome.terminal:
                    raise ValueError("Human and Stop branches must terminate")
                if not outcome.terminal and not index < outcome.next_step < len(episode.steps):
                    raise ValueError("nonterminal AI branches must advance within the episode")
                if not all(math.isfinite(v) for v in (outcome.quality_delta, outcome.latency, outcome.execution_cost, outcome.human_minutes)):
                    raise ValueError("branch rewards and resource fields must be finite")
    return episodes


def require_disjoint(left: Sequence[RepairEpisode], right: Sequence[RepairEpisode]) -> None:
    for field in ("episode_id", "group_id", "source_workflow_id", "source_document_sha256", "normalized_source_fingerprint"):
        def values(episodes):
            result = []
            for episode in episodes:
                value = getattr(episode, field) if field in {"episode_id", "group_id"} else episode.metadata.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{episode.episode_id} lacks overlap-audit field {field}")
                result.append(value)
            return set(result)
        if values(left) & values(right):
            raise ValueError(f"development/evaluation overlap in {field}")


def paper_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Separate cumulative raw flags from ordinary terminal repair failures."""
    records = result["records"]
    autonomous = [r for r in records if not r["human_used"] and r["actions"][-1] != int(AuthorityAction.STOP)]
    output = {k: v for k, v in result.items() if k != "records"}
    output["autonomous_completed_episodes"] = len(autonomous)
    output["uar"] = sum(r["unsafe"] or not r["success"] for r in autonomous) / len(autonomous) if autonomous else None
    output["autonomous_terminal_failure_rate"] = sum(not r["success"] for r in autonomous) / len(autonomous) if autonomous else None
    output["flagged_episode_rate"] = sum(r["unsafe"] for r in records) / len(records)
    output["coverage"] = 1.0 - output["stopped_rate"]
    output["return"] = output["normalized_return"]
    # This historical diagnostic counts adverse decisions, including ordinary failures.
    output["adverse_decisions_per_episode"] = output.pop("unsafe_autonomy_cost_rate")
    return output
