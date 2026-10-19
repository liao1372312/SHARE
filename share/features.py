from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .types import RepairState


@dataclass
class FeatureEncoder:
    gap_types: List[str]
    root_types: List[str]
    domains: List[str]
    mean: np.ndarray
    scale: np.ndarray
    disabled_groups: tuple[str, ...] = ()

    @classmethod
    def fit(cls, states: Iterable[RepairState]) -> "FeatureEncoder":
        items = list(states)
        if not items:
            raise ValueError("cannot fit feature encoder without states")
        continuous = np.asarray([state.continuous() for state in items], dtype=np.float32)
        mean = continuous.mean(axis=0)
        scale = continuous.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(
            sorted({state.gap_type for state in items}),
            sorted({state.root_type for state in items}),
            sorted({state.domain for state in items}),
            mean,
            scale,
        )

    @property
    def dimension(self) -> int:
        return len(self.gap_types) + 1 + len(self.root_types) + 1 + len(self.domains) + 1 + len(self.mean)

    def transform_one(self, state: RepairState) -> np.ndarray:
        categorical = np.concatenate(
            [
                self._one_hot(state.gap_type, self.gap_types),
                self._one_hot(state.root_type, self.root_types),
                self._one_hot(state.domain, self.domains),
            ]
        )
        continuous = (np.asarray(state.continuous(), dtype=np.float32) - self.mean) / self.scale
        if "causal" in self.disabled_groups:
            categorical[: len(self.gap_types) + 1 + len(self.root_types) + 1] = 0
            continuous[[0, 2, 3]] = 0
        if "risk" in self.disabled_groups:
            continuous[[4, 6, 7]] = 0
        if "history" in self.disabled_groups:
            continuous[5] = 0
        return np.concatenate([categorical, continuous]).astype(np.float32)

    def without(self, *groups: str) -> "FeatureEncoder":
        valid = {"causal", "risk", "history"}
        unknown = set(groups) - valid
        if unknown:
            raise ValueError(f"unknown feature groups: {sorted(unknown)}")
        return FeatureEncoder(self.gap_types, self.root_types, self.domains, self.mean.copy(), self.scale.copy(), tuple(groups))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "gap_types": self.gap_types, "root_types": self.root_types, "domains": self.domains,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(), "disabled_groups": list(self.disabled_groups),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureEncoder":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(value["gap_types"], value["root_types"], value["domains"], np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32), tuple(value.get("disabled_groups", [])))

    def transform(self, states: Sequence[RepairState]) -> np.ndarray:
        return np.stack([self.transform_one(state) for state in states])

    @staticmethod
    def _one_hot(value: str, vocabulary: List[str]) -> np.ndarray:
        result = np.zeros(len(vocabulary) + 1, dtype=np.float32)
        try:
            result[vocabulary.index(value)] = 1.0
        except ValueError:
            result[-1] = 1.0
        return result
