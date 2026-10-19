from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .data import stable_uniform
from .data import label_human_benefit
from .types import RewardConfig
from .types import BranchOutcome, PairedRepairStep, RepairEpisode


def corrupt_human_feedback(episodes: Sequence[RepairEpisode], rate: float, seed: int) -> list[RepairEpisode]:
    if not 0 <= rate <= 1:
        raise ValueError("corruption rate must be in [0, 1]")
    output = []
    for episode in episodes:
        steps = []
        for step in episode.steps:
            outcomes = dict(step.outcomes)
            human = outcomes["human"]
            if stable_uniform(f"corrupt:{seed}:{episode.episode_id}:{step.state.step}") < rate:
                outcomes["human"] = replace(
                    human,
                    success=not human.success,
                    quality_delta=-human.quality_delta,
                    metadata={**human.metadata, "corrupted": True, "corruption_rate": rate},
                )
            steps.append(PairedRepairStep(step.state, outcomes, step.human_beneficial))
        steps = label_human_benefit(steps, RewardConfig(), margin=0.1)
        output.append(replace(episode, steps=steps, metadata={**episode.metadata, "human_feedback_corruption": rate}))
    return output


def chronological_blocks(episodes: Sequence[RepairEpisode], blocks: int = 5) -> list[list[RepairEpisode]]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    ordered = sorted(episodes, key=lambda episode: str(episode.metadata.get("timestamp", episode.episode_id)))
    return [ordered[index::blocks] for index in range(blocks)]
