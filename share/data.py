from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from flare.core import ExecutionGraph
from flare.detection import detect_failures
from flare.localization import localize_faulty_decision
from flare.repair import select_repair_action
from flare.slicing import merged_backward_slice

from .environment import base_reward
from .types import AuthorityAction, BranchOutcome, PairedRepairStep, RepairEpisode, RepairState, RewardConfig


SIMULATOR_VERSION = "share-synthetic-v1"


def load_episodes(path: Path) -> List[RepairEpisode]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    episodes: List[RepairEpisode] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    episodes.append(RepairEpisode.from_dict(json.loads(line)))
    validate_splits(episodes)
    return episodes


def write_episodes(path: Path, episodes: Sequence[RepairEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")


def build_synthetic_episodes(rows: Iterable[Mapping[str, Any]], seed: int = 17, repair_budget: int = 3, human_margin: float = 0.1) -> List[RepairEpisode]:
    episodes = []
    reward_config = RewardConfig()
    for row in rows:
        label = row["repair_label"]
        if not label.get("failed"):
            continue
        episode_id = str(row["id"])
        group_id = infer_group_id(row)
        stratum = split_stratum(row)
        split = split_for_group(group_id, seed, stratum=stratum)
        graph = ExecutionGraph.from_events(row["events"])
        failures = detect_failures(graph)
        slice_nodes = merged_backward_slice(graph, failures)
        localization = localize_faulty_decision(graph, failures, slice_nodes)
        root = localization.failure_type if localization else str(label.get("failure_type", "unknown"))
        loc_score = localization.scores.get(localization.node_id, 0.0) if localization else 0.0
        score_norm = 1.0 / (1.0 + math.exp(-loc_score))
        affected = localization.affected if localization else set(label.get("affected_slice", []))
        affected_ratio = len(affected) / max(1, len(graph.nodes))
        expected_action = str(label.get("expected_repair_action", ""))
        predicted_action = select_repair_action(graph, localization).value if localization else "NoOp"
        action_match = float(predicted_action == expected_action)
        complexity = min(1.0, max(0.0, (len(graph.nodes) - 7) / 20.0))
        horizon = 1 + int(complexity >= 0.25) + int(complexity >= 0.7)
        horizon = min(repair_budget, horizon)
        steps: List[PairedRepairStep] = []
        history_failures = 0
        for step in range(horizon):
            uncertainty = min(1.0, 0.55 * (1.0 - score_norm) + 0.25 * complexity + 0.15 * history_failures)
            risk = min(1.0, 0.25 + 0.35 * affected_ratio + 0.2 * complexity + 0.15 * history_failures)
            state = RepairState(
                episode_id=episode_id,
                step=step,
                gap_type=str(label.get("failure_type", "unknown")),
                root_type=root,
                localization_score=score_norm,
                repair_utility=min(1.0, 0.35 + 0.45 * action_match + 0.2 * score_norm),
                affected_ratio=affected_ratio,
                downstream_impact=min(1.0, affected_ratio * (1.0 + complexity)),
                drift_risk=risk,
                previous_failures=min(1.0, history_failures / max(1, repair_budget)),
                uncertainty=uncertainty,
                evidence_conflict=float(label.get("failure_type") == "answer_grounding_failure") * 0.7,
                remaining_budget=(repair_budget - step) / repair_budget,
                complexity=complexity,
                domain=str(row.get("metadata", {}).get("original_domain", row.get("dataset", "unknown"))).lower(),
            )
            ai_probability = min(0.96, max(0.05, 0.15 + 0.5 * action_match + 0.3 * score_norm - 0.22 * complexity - 0.12 * step))
            human_probability = min(0.98, max(0.2, 0.88 - 0.16 * complexity - 0.08 * state.evidence_conflict))
            ai_success = stable_uniform(f"{seed}:{episode_id}:{step}:ai") < ai_probability
            human_success = stable_uniform(f"{seed}:{episode_id}:{step}:human") < human_probability
            ai_unsafe = not ai_success and stable_uniform(f"{seed}:{episode_id}:{step}:unsafe") < risk
            ai_terminal = ai_success or step == horizon - 1
            outcomes = {
                "ai": BranchOutcome(
                    success=ai_success,
                    unsafe=ai_unsafe,
                    quality_delta=0.8 if ai_success else -0.35,
                    latency=0.15 + 0.25 * affected_ratio,
                    execution_cost=0.12 + 0.35 * affected_ratio,
                    next_step=None if ai_terminal else step + 1,
                    terminal=ai_terminal,
                    metadata={"probability": ai_probability, "unsafe_probability": (1.0 - ai_probability) * risk, "source": SIMULATOR_VERSION},
                ),
                "human": BranchOutcome(
                    success=human_success,
                    unsafe=False,
                    quality_delta=0.9 if human_success else -0.15,
                    latency=0.55 + 0.35 * complexity,
                    execution_cost=0.18,
                    human_minutes=4.0 + 8.0 * complexity + 2.0 * step,
                    terminal=True,
                    metadata={"probability": human_probability, "source": SIMULATOR_VERSION},
                ),
                "stop": BranchOutcome(
                    success=False,
                    unsafe=False,
                    quality_delta=-0.1,
                    latency=0.05,
                    execution_cost=0.02,
                    terminal=True,
                    metadata={"source": SIMULATOR_VERSION},
                ),
            }
            steps.append(PairedRepairStep(state, outcomes, False))
            history_failures += int(not ai_success)
        steps = label_human_benefit(steps, reward_config, human_margin)
        episodes.append(
            RepairEpisode(
                episode_id=episode_id,
                group_id=group_id,
                split=split,
                source=SIMULATOR_VERSION,
                synthetic=True,
                steps=steps,
                metadata={"dataset": row.get("dataset"), "gold_action": expected_action, "simulator_version": SIMULATOR_VERSION},
            )
        )
    validate_splits(episodes)
    return episodes


def label_human_benefit(steps: Sequence[PairedRepairStep], config: RewardConfig, margin: float, gamma: float = 0.95) -> List[PairedRepairStep]:
    ai_returns = [0.0] * len(steps)
    for index in range(len(steps) - 1, -1, -1):
        ai = steps[index].outcomes["ai"]
        value = base_reward(ai, AuthorityAction.AI, config)
        if not ai.terminal and ai.next_step is not None:
            value += gamma * ai_returns[ai.next_step]
        ai_returns[index] = value
    labeled = []
    for index, step in enumerate(steps):
        human_return = base_reward(step.outcomes["human"], AuthorityAction.HUMAN, config)
        labeled.append(PairedRepairStep(step.state, step.outcomes, human_return - ai_returns[index] > margin))
    return labeled


def read_flare_rows(root: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    count = 0
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return


def split_for_group(group_id: str, seed: int, train: float = 0.7, validation: float = 0.15, stratum: str = "") -> str:
    value = stable_uniform(f"split:{seed}:{group_id}")
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def infer_group_id(row: Mapping[str, Any]) -> str:
    task = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", str(row.get("task", "")).lower())
    task = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<id>", task)
    task = re.sub(r"'[^']+'|\"[^\"]+\"", "<value>", task)
    task = re.sub(r"\s+", " ", task).strip()
    events = row.get("events", [])
    event_types = [str(event.get("type", "")) for event in events]
    dependencies = [len(event.get("meta", {}).get("depends_on", [])) for event in events]
    candidate_domains = sorted({str(item).split(".", 1)[0] for item in row.get("candidate_api_ids", [])[:12]})
    signature = json.dumps([task, event_types, dependencies, candidate_domains], separators=(",", ":"))
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]


def split_stratum(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    domain = str(metadata.get("original_domain", row.get("dataset", "unknown"))).lower()
    failure = str(row.get("repair_label", {}).get("failure_type", "unknown"))
    steps = int(metadata.get("trace_steps", sum(event.get("type") == "plan" for event in row.get("events", []))))
    complexity = "simple" if steps <= 2 else "complex"
    return f"{domain}:{failure}:{complexity}"


def stable_uniform(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def validate_splits(episodes: Sequence[RepairEpisode]) -> None:
    group_splits: Dict[str, set[str]] = {}
    ids = set()
    for episode in episodes:
        if episode.episode_id in ids:
            raise ValueError(f"duplicate episode id: {episode.episode_id}")
        ids.add(episode.episode_id)
        if episode.split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {episode.split}")
        group_splits.setdefault(episode.group_id, set()).add(episode.split)
        for step in episode.steps:
            if set(step.outcomes) != {"ai", "human", "stop"}:
                raise ValueError(f"episode {episode.episode_id} lacks paired authority outcomes")
            if step.state.step < 0 or step.state.step >= len(episode.steps):
                raise ValueError(f"episode {episode.episode_id} has inconsistent step index")
            for name, outcome in step.outcomes.items():
                if outcome.terminal and outcome.next_step is not None:
                    raise ValueError(f"terminal {name} branch has next_step")
                if not outcome.terminal and outcome.next_step is None:
                    raise ValueError(f"nonterminal {name} branch lacks next_step")
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"group leakage across splits: {leaked[:3]}")
