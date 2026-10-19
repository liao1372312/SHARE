from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import stable_uniform
from .types import RepairEpisode


@dataclass(frozen=True)
class GuardPartitions:
    fitting: tuple[RepairEpisode, ...]
    selection: tuple[RepairEpisode, ...]
    certification: tuple[RepairEpisode, ...]
    test: tuple[RepairEpisode, ...]


def guard_development_partitions(
    episodes: Sequence[RepairEpisode],
    seed: int = 17,
    selection_fraction: float = 0.2,
) -> GuardPartitions:
    """Split a train-only Guard development file without loading old outcomes.

    The frozen-replay selection screen must receive a separate file containing
    only original training groups. Historical validation and test records are
    intentionally absent from this interface, not merely left unevaluated.
    """
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be in (0, 1)")
    if not episodes:
        raise ValueError("SHARE-Guard development file is empty")
    non_train = sorted({episode.split for episode in episodes if episode.split != "train"})
    if non_train:
        raise ValueError(f"SHARE-Guard development file must contain only train episodes: {non_train}")
    fitting = []
    selection = []
    group_partition: dict[str, str] = {}
    for episode in episodes:
        partition = group_partition.setdefault(
            episode.group_id,
            "selection" if stable_uniform(f"share-guard:{seed}:{episode.group_id}") < selection_fraction else "fitting",
        )
        (selection if partition == "selection" else fitting).append(episode)
    if not fitting or not selection:
        raise ValueError("SHARE-Guard development file must yield non-empty fitting and selection groups")
    if {episode.group_id for episode in fitting} & {episode.group_id for episode in selection}:
        raise ValueError("group leakage across SHARE-Guard fitting and selection partitions")
    return GuardPartitions(tuple(fitting), tuple(selection), (), ())


def guard_partitions(episodes: Sequence[RepairEpisode], seed: int = 17, selection_fraction: float = 0.2) -> GuardPartitions:
    """Create group-isolated fit/selection/certification/test partitions.

    The dataset's validation split remains untouched certification data. Only
    original training groups are deterministically divided into fitting and
    candidate-selection groups.
    """
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be in (0, 1)")
    fitting = []
    selection = []
    certification = []
    test = []
    group_partition: dict[str, str] = {}
    for episode in episodes:
        if episode.split == "train":
            partition = group_partition.setdefault(
                episode.group_id,
                "selection" if stable_uniform(f"share-guard:{seed}:{episode.group_id}") < selection_fraction else "fitting",
            )
            (selection if partition == "selection" else fitting).append(episode)
        elif episode.split == "validation":
            certification.append(episode)
        elif episode.split == "test":
            test.append(episode)
        else:
            raise ValueError(f"unsupported split: {episode.split}")
    if not fitting or not selection or not certification or not test:
        raise ValueError("all SHARE-Guard protocol partitions must be non-empty")
    groups = [
        {episode.group_id for episode in partition}
        for partition in (fitting, selection, certification, test)
    ]
    if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
        raise ValueError("group leakage across SHARE-Guard protocol partitions")
    return GuardPartitions(tuple(fitting), tuple(selection), tuple(certification), tuple(test))
