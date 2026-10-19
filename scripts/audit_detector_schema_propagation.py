"""Audit the effect of recognizing the generated missing_parameters key.

The correction is applied in memory. No detector code, frozen episode, or
archived prediction is modified. Run from the repository root with:

    python scripts/audit_detector_schema_propagation.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flare import detection
from flare.core import ExecutionGraph
from flare.localization import localize_faulty_decision
from flare.slicing import merged_backward_slice
from share.data import build_synthetic_episodes, load_episodes, read_flare_rows
from share.features import FeatureEncoder


def evaluate(rows: list[dict], corrected: bool):
    if corrected:
        detection._PARAM_KEYS.add("missing_parameters")
    else:
        detection._PARAM_KEYS.discard("missing_parameters")
    try:
        localizations = {}
        for row in rows:
            if not row["repair_label"].get("failed"):
                continue
            graph = ExecutionGraph.from_events(row["events"])
            failures = detection.detect_failures(graph)
            localizations[row["id"]] = localize_faulty_decision(
                graph, failures, merged_backward_slice(graph, failures)
            )
        episodes = build_synthetic_episodes(rows)
        return localizations, episodes
    finally:
        detection._PARAM_KEYS.discard("missing_parameters")


def main() -> None:
    rows = list(read_flare_rows(ROOT / "datasets/flare_complexity"))
    row_by_id = {row["id"]: row for row in rows}
    old_loc, old_episodes = evaluate(rows, corrected=False)
    new_loc, new_episodes = evaluate(rows, corrected=True)
    frozen = load_episodes(ROOT / "datasets/share_synthetic/episodes.jsonl")
    assert [asdict(ep) for ep in old_episodes] == [asdict(ep) for ep in frozen], (
        "The uncorrected rebuild no longer matches the frozen development episodes"
    )

    localization = Counter()
    localization_by_dataset = Counter()
    for episode_id, before in old_loc.items():
        after = new_loc[episode_id]
        dataset = row_by_id[episode_id]["dataset"]
        for name, changed in (
            ("top_node", before.node_id != after.node_id),
            ("ranking", list(before.scores) != list(after.scores)),
            ("candidate_scores", before.scores != after.scores),
            ("failure_label", before.failure_type != after.failure_type),
            ("confidence", before.confidence != after.confidence),
        ):
            if changed:
                localization[name] += 1
                localization_by_dataset[f"{dataset}:{name}"] += 1

    state_fields = Counter()
    outcome_fields = Counter()
    label_transitions = Counter()
    changed_episodes_by_split = Counter()
    changed_steps = 0
    for before, after in zip(old_episodes, new_episodes):
        assert before.episode_id == after.episode_id
        episode_changed = False
        for old_step, new_step in zip(before.steps, after.steps):
            old_state, new_state = asdict(old_step.state), asdict(new_step.state)
            step_changed = False
            for field in old_state:
                if old_state[field] != new_state[field]:
                    state_fields[field] += 1
                    step_changed = True
            for branch in ("ai", "human", "stop"):
                old_outcome, new_outcome = (
                    asdict(old_step.outcomes[branch]),
                    asdict(new_step.outcomes[branch]),
                )
                for field in old_outcome:
                    if old_outcome[field] != new_outcome[field]:
                        outcome_fields[f"{branch}:{field}"] += 1
                        step_changed = True
            if old_step.human_beneficial != new_step.human_beneficial:
                label_transitions[
                    f"{old_step.human_beneficial}->{new_step.human_beneficial}"
                ] += 1
                step_changed = True
            if step_changed:
                changed_steps += 1
                episode_changed = True
        if episode_changed:
            changed_episodes_by_split[before.split] += 1

    old_train = [step.state for ep in old_episodes if ep.split == "train" for step in ep.steps]
    new_train = [step.state for ep in new_episodes if ep.split == "train" for step in ep.steps]
    old_encoder, new_encoder = FeatureEncoder.fit(old_train), FeatureEncoder.fit(new_train)

    result = {
        "intervention": "add missing_parameters to detector parameter keys in memory",
        "raw_failed_cases": len(old_loc),
        "frozen_development_episodes_match_uncorrected_rebuild": True,
        "localization_changed": dict(localization),
        "localization_changed_by_dataset": dict(sorted(localization_by_dataset.items())),
        "development_episodes": len(old_episodes),
        "development_states": sum(len(ep.steps) for ep in old_episodes),
        "changed_episodes_by_split": dict(sorted(changed_episodes_by_split.items())),
        "changed_steps": changed_steps,
        "state_fields_changed": dict(sorted(state_fields.items())),
        "branch_outcome_fields_changed": dict(sorted(outcome_fields.items())),
        "human_benefit_label_transitions": dict(label_transitions),
        "training_encoder": {
            "training_states": len(old_train),
            "root_types_before": old_encoder.root_types,
            "root_types_after": new_encoder.root_types,
            "dimension_before": old_encoder.dimension,
            "dimension_after": new_encoder.dimension,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
