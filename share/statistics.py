from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np


Record = Mapping[str, object]
Metric = Callable[[Record], float]


METRICS: Mapping[str, Metric] = {
    "repair_sr": lambda record: float(bool(record["success"])),
    "unsafe_autonomy_cost_rate": lambda record: float(record.get("unsafe_actions", bool(record["unsafe"]))),
    "hir": lambda record: float(bool(record["human_used"])),
    "normalized_return": lambda record: float(sum(record["rewards"])),
    "human_minutes": lambda record: float(record["human_minutes"]),
    "stopped_rate": lambda record: float(record["actions"][-1] == 2),
}


def cluster_bootstrap_ci(
    records: Sequence[Record],
    episode_groups: Mapping[str, str],
    metric: str,
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    """Hierarchical bootstrap of one policy's episode-level mean metric.

    Workflow groups are resampled first. Each selected group's episodes are then
    resampled with replacement at the original group size, matching the
    repository's declared cluster-aware inference protocol.
    """
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}")
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        episode_id = str(record["episode_id"])
        if episode_id not in episode_groups:
            raise ValueError(f"missing group_id for episode {episode_id}")
        grouped[episode_groups[episode_id]].append(METRICS[metric](record))
    if not grouped:
        raise ValueError("no records")
    rng = random.Random(seed)
    group_ids = sorted(grouped)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in group_ids:
            group_values = grouped[rng.choice(group_ids)]
            values.extend(rng.choice(group_values) for _ in range(len(group_values)))
        estimates.append(float(np.mean(values)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_differences(left: Sequence[Record], right: Sequence[Record], metric: str) -> dict[str, float]:
    getter = METRICS[metric]
    left_by_id = {str(record["episode_id"]): record for record in left}
    right_by_id = {str(record["episode_id"]): record for record in right}
    if set(left_by_id) != set(right_by_id):
        missing_left = sorted(set(right_by_id) - set(left_by_id))[:5]
        missing_right = sorted(set(left_by_id) - set(right_by_id))[:5]
        raise ValueError(f"unpaired episode records; missing_left={missing_left}, missing_right={missing_right}")
    return {
        episode_id: getter(left_by_id[episode_id]) - getter(right_by_id[episode_id])
        for episode_id in left_by_id
    }


def hierarchical_bootstrap_ci(
    differences: Mapping[str, float],
    episode_groups: Mapping[str, str],
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for episode_id, difference in differences.items():
        if episode_id not in episode_groups:
            raise ValueError(f"missing group_id for episode {episode_id}")
        grouped[episode_groups[episode_id]].append(float(difference))
    if not grouped:
        raise ValueError("no paired differences")
    rng = random.Random(seed)
    group_ids = sorted(grouped)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in group_ids:
            group = grouped[rng.choice(group_ids)]
            values.extend(rng.choice(group) for _ in group)
        estimates.append(float(np.mean(values)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_permutation_test(differences: Mapping[str, float], permutations: int = 10_000, seed: int = 17) -> float:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    values = list(differences.values())
    if not values:
        raise ValueError("no paired differences")
    observed = abs(float(np.mean(values)))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        estimate = abs(float(np.mean([value if rng.random() < 0.5 else -value for value in values])))
        exceed += estimate >= observed - 1e-15
    return (exceed + 1) / (permutations + 1)


def paired_effect_size(differences: Mapping[str, float]) -> float | None:
    values = np.asarray(list(differences.values()), dtype=float)
    if not len(values):
        raise ValueError("no paired differences")
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values) / standard_deviation) if standard_deviation > 0 else None


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted[index] = running
    return adjusted
