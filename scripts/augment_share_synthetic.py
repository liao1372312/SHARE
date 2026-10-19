from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import (
    build_synthetic_episodes,
    infer_group_id,
    read_flare_rows,
    split_for_group,
    split_stratum,
    stable_uniform,
    validate_splits,
    write_episodes,
)


AUGMENTATION_VERSION = "share-synthetic-augmented-v1"
FAILURE_TYPES = (
    "planning_failure",
    "service_selection_failure",
    "parameter_binding_failure",
    "tool_execution_failure",
    "supervisor_failure",
    "answer_grounding_failure",
)
FAILURE_DEFINITIONS = {
    "planning_failure": ("plan", "planner", "AddSubtask"),
    "service_selection_failure": ("service_selection", "service_provider", "ReselectAPI"),
    "parameter_binding_failure": ("parameter_binding", "executor", "RebindParam"),
    "tool_execution_failure": ("api_invocation", "external_tool", "RetryTool"),
    "supervisor_failure": ("repair", "supervisor", "ReviseRepair"),
    "answer_grounding_failure": ("claim", "summarizer", "ReviseAnswer"),
}


def sha256_directory_jsonl(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.jsonl")):
        digest.update(file.name.encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def parse_failure_types(value: str) -> tuple[str, ...]:
    resolved = tuple(item.strip() for item in value.split(",") if item.strip())
    if not resolved:
        raise ValueError("failure types must be non-empty")
    unknown = sorted(set(resolved) - set(FAILURE_TYPES))
    if unknown:
        raise ValueError(f"unsupported failure types: {unknown}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("failure types must be unique")
    return resolved


def source_group_id(row: Mapping) -> str:
    """Bind all fault variants of one source workflow to a single split group."""
    return f"aug-{infer_group_id(row)}"


def choose_event(events: Sequence[Mapping], event_type: str, source_id: str, failure_type: str, seed: int) -> dict:
    candidates = [event for event in events if str(event.get("type")) == event_type]
    if not candidates:
        raise ValueError(f"source {source_id} has no {event_type} event for {failure_type}")
    index = min(len(candidates) - 1, int(stable_uniform(f"{seed}:{source_id}:{failure_type}") * len(candidates)))
    return candidates[index]


def affected_ids(events: Sequence[Mapping], faulty_node: str) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for event in events:
        for parent in event.get("meta", {}).get("depends_on", []):
            children[str(parent)].append(str(event["id"]))
    seen: set[str] = set()
    pending = list(children[faulty_node])
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(children[node])
    return sorted(seen)


def inject_failure(row: Mapping, failure_type: str, seed: int) -> dict:
    """Create one labeled synthetic fault from a trace with no injected failure."""
    if failure_type not in FAILURE_DEFINITIONS:
        raise ValueError(f"unsupported failure type: {failure_type}")
    if row.get("repair_label", {}).get("failed"):
        raise ValueError("augmentation accepts only source traces without a primary injected failure")
    variant = copy.deepcopy(dict(row))
    source_id = str(row["id"])
    events = variant["events"]
    event_type, faulty_agent, repair_action = FAILURE_DEFINITIONS[failure_type]
    if failure_type == "supervisor_failure":
        anchors = [event for event in events if str(event.get("type")) == "api_invocation"]
        anchor = anchors[-1] if anchors else events[-1]
        event_ids = {str(event["id"]) for event in events}
        repair_id = "share_augmented_supervisor_repair"
        while repair_id in event_ids:
            repair_id = f"{repair_id}_x"
        faulty = {
            "id": repair_id,
            "agent": "supervisor",
            "type": event_type,
            "input": None,
            "output": {"action": "RetryTool"},
            "meta": {"depends_on": [str(anchor["id"])], "repair_failed": True, "repeated_retry": 3},
            "status": "failed",
            "time": float(anchor.get("time", len(events))) + 0.01,
        }
        events.append(faulty)
    else:
        faulty = choose_event(events, event_type, source_id, failure_type, seed)
        faulty["meta"] = dict(faulty.get("meta", {}))
        if failure_type == "planning_failure":
            faulty["status"] = "failed"
            faulty["meta"].update({"missing_dependency": "synthetic missing prerequisite subtask", "invalid_dependency": True})
        elif failure_type == "service_selection_failure":
            faulty["status"] = "failed"
            faulty["meta"].update({"wrong_api": True, "hallucinated_api": "synthetic.nonexistent.service"})
        elif failure_type == "parameter_binding_failure":
            faulty["status"] = "failed"
            faulty["meta"].update({"missing_required": ["synthetic_required_parameter"]})
            faulty["output"] = {"error": "synthetic missing required parameter"}
        elif failure_type == "tool_execution_failure":
            faulty["status"] = "failed"
            faulty["meta"].update({"error": "synthetic tool execution failure"})
            faulty["output"] = {"error": "synthetic tool execution failure"}
        elif failure_type == "answer_grounding_failure":
            faulty["meta"].update({"unsupported": True})
            faulty["meta"].pop("supported_by", None)
    group = source_group_id(row)
    metadata = dict(variant.get("metadata", {}))
    metadata.update({
        "source": AUGMENTATION_VERSION,
        "source_row_id": source_id,
        "source_workflow_group": group,
        "primary_failure_injected": True,
        "failure_type": failure_type,
        "expected_repair_action": repair_action,
        "augmentation_seed": seed,
        "generator": "scripts/augment_share_synthetic.py",
        "evidence_warning": "Synthetic fault-injection augmentation only; not real paired authority outcomes.",
    })
    variant["id"] = f"{source_id}__share_aug__{failure_type}"
    variant["synthetic"] = True
    variant["metadata"] = metadata
    variant["repair_label"] = {
        "failed": True,
        "faulty_agent": faulty_agent,
        "failure_type": failure_type,
        "faulty_decision_node": str(faulty["id"]),
        "expected_repair_action": repair_action,
        "affected_slice": affected_ids(events, str(faulty["id"])),
        "annotation_source": "synthetic_fault_injection_augmentation",
    }
    return variant


def build_augmented_rows(rows: Iterable[Mapping], failure_types: Sequence[str], seed: int, limit_source_rows: int | None = None) -> tuple[list[dict], dict[str, dict]]:
    variants: list[dict] = []
    provenance: dict[str, dict] = {}
    source_count = 0
    for row in rows:
        if row.get("repair_label", {}).get("failed"):
            continue
        if limit_source_rows is not None and source_count >= limit_source_rows:
            break
        source_count += 1
        source_id = str(row["id"])
        group = source_group_id(row)
        provenance[source_id] = {
            "group_id": group,
            "split": split_for_group(group, seed, stratum=split_stratum(row)),
            "source_stratum": split_stratum(row),
        }
        for failure_type in failure_types:
            variants.append(inject_failure(row, failure_type, seed))
    if not variants:
        raise ValueError("no clean source traces were available for augmentation")
    return variants, provenance


def build_augmented_episodes(rows: Sequence[Mapping], provenance: Mapping[str, Mapping], seed: int, repair_budget: int, human_margin: float):
    episodes = build_synthetic_episodes(rows, seed, repair_budget, human_margin)
    adjusted = []
    for episode in episodes:
        source_id, separator, _ = episode.episode_id.rpartition("__share_aug__")
        if not separator or source_id not in provenance:
            raise ValueError(f"augmented episode lacks source provenance: {episode.episode_id}")
        source = provenance[source_id]
        adjusted.append(replace(
            episode,
            group_id=str(source["group_id"]),
            split=str(source["split"]),
            source=AUGMENTATION_VERSION,
            metadata={**episode.metadata, "source_stratum": source["source_stratum"]},
        ))
    validate_splits(adjusted)
    return adjusted


def write_dataset_card(path: Path, summary: Mapping) -> None:
    path.write_text(
        "\n".join([
            "# SHARE Synthetic Augmented Dataset",
            "",
            "This dataset is generated by deterministic fault injection over existing synthetic FLARE traces.",
            "It is intended only for implementation checks, stress tests, and development diagnostics.",
            "It is not real paired authority outcomes, an independent held-out evaluation set, or a practitioner-study substitute.",
            "",
            f"- Episodes: {summary['episodes']}",
            f"- Source clean traces: {summary['source_clean_traces']}",
            f"- Workflow groups: {summary['workflow_groups']}",
            f"- Failure variants per source trace: {len(summary['failure_types'])}",
            f"- Generator: `{summary['generator']}`",
            f"- Source SHA-256: `{summary['source_sha256']}`",
            "",
            "All variants of the same source workflow share one group and one split. Therefore the episode count must not be treated as an independent sample count.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand clean synthetic FLARE traces into explicitly labeled SHARE fault-injection episodes.")
    parser.add_argument("--data", type=Path, default=Path("datasets/flare_complexity"), help="Directory of synthetic FLARE JSONL traces.")
    parser.add_argument("--out", type=Path, default=Path("datasets/share_synthetic_augmented/episodes.jsonl"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--repair-budget", type=int, default=3)
    parser.add_argument("--human-margin", type=float, default=0.1)
    parser.add_argument("--failure-types", default=",".join(FAILURE_TYPES))
    parser.add_argument("--limit-source-rows", type=int, default=None, help="Optional deterministic cap for smoke tests; default uses every clean source trace.")
    args = parser.parse_args()
    if not args.data.is_dir():
        raise SystemExit("--data must be a directory containing JSONL traces")
    if args.repair_budget <= 0:
        raise SystemExit("--repair-budget must be positive")
    if args.limit_source_rows is not None and args.limit_source_rows <= 0:
        raise SystemExit("--limit-source-rows must be positive")
    failure_types = parse_failure_types(args.failure_types)
    all_rows = list(read_flare_rows(args.data))
    variants, provenance = build_augmented_rows(all_rows, failure_types, args.seed, args.limit_source_rows)
    episodes = build_augmented_episodes(variants, provenance, args.seed, args.repair_budget, args.human_margin)
    write_episodes(args.out, episodes)
    summary = {
        "schema_version": 1,
        "output": str(args.out),
        "generator": "scripts/augment_share_synthetic.py",
        "source": AUGMENTATION_VERSION,
        "source_directory": str(args.data),
        "source_sha256": sha256_directory_jsonl(args.data),
        "source_rows": len(all_rows),
        "source_clean_traces": len(provenance),
        "episodes": len(episodes),
        "workflow_groups": len({episode.group_id for episode in episodes}),
        "splits": dict(sorted(Counter(episode.split for episode in episodes).items())),
        "groups_by_split": dict(sorted(Counter({episode.group_id: episode.split for episode in episodes}.values()).items())),
        "failure_types": list(failure_types),
        "failure_type_counts": dict(sorted(Counter(episode.steps[0].state.gap_type for episode in episodes).items())),
        "decision_states": sum(len(episode.steps) for episode in episodes),
        "multi_step_episodes": sum(len(episode.steps) > 1 for episode in episodes),
        "human_beneficial_states": sum(step.human_beneficial for episode in episodes for step in episode.steps),
        "synthetic": True,
        "evidence_warning": "Synthetic fault-injection augmentation only. It cannot serve as real paired authority evidence, a new independent holdout, or practitioner-study evidence.",
    }
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_dataset_card(args.out.with_name("DATASET_CARD.md"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
