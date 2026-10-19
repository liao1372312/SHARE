from __future__ import annotations

import hashlib
import random
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from .environment import RepairAuthorityEnv, base_reward
from .features import FeatureEncoder
from .support import SupportKey, support_key
from .types import AuthorityAction, RepairEpisode, RewardConfig


ReplayTransition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray, float, float]


@dataclass(frozen=True)
class FrozenLoggedReplay:
    """Candidate-independent logged transitions and their fixed action support."""

    transitions: tuple[ReplayTransition, ...]
    support_counts: Mapping[tuple[SupportKey, int], int]
    next_support_keys: tuple[SupportKey | None, ...]
    behavior_seed: int
    behavior_policy: str
    content_sha256: str

    @property
    def unique_support_observations(self) -> int:
        return int(sum(self.support_counts.values()))


def collect_frozen_logged_replay(
    episodes: Sequence[RepairEpisode],
    encoder: FeatureEncoder,
    reward_config: RewardConfig | None = None,
    *,
    behavior_seed: int = 17,
    paired_counterfactual: bool = False,
) -> FrozenLoggedReplay:
    """Collect a fixed replay before any candidate is initialized.

    The primary behavior policy samples uniformly from feasible actions. It is
    intentionally independent of the candidate architecture and dual weights.
    The paired mode enumerates all feasible branches and is an explicit
    counterfactual-replay ablation.
    """
    if not episodes:
        raise ValueError("cannot collect replay from no episodes")
    config = reward_config or RewardConfig()
    transitions: list[ReplayTransition] = []
    next_support_keys: list[SupportKey | None] = []
    support_counts: Counter[tuple[SupportKey, int]] = Counter()
    observed: set[tuple[str, int, int]] = set()
    rng = random.Random(behavior_seed)
    for episode in sorted(episodes, key=lambda item: item.episode_id):
        if paired_counterfactual:
            for paired_step in episode.steps:
                state = paired_step.state
                feasible = [AuthorityAction.HUMAN, AuthorityAction.STOP]
                if state.remaining_budget > 0:
                    feasible.insert(0, AuthorityAction.AI)
                for action in feasible:
                    outcome = paired_step.outcome(action)
                    next_state = episode.steps[outcome.next_step].state if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None else None
                    transition = _transition(encoder, state, action, base_reward(outcome, action, config), next_state, next_state is None, float(action == AuthorityAction.HUMAN), float(outcome.unsafe or (outcome.terminal and action == AuthorityAction.AI and not outcome.success)))
                    transitions.append(transition)
                    next_support_keys.append(support_key(next_state) if next_state else None)
                    _record_support(support_counts, observed, state, action)
            continue
        env = RepairAuthorityEnv(episode, config)
        state = env.reset()
        while not env.done:
            action = rng.choice(list(env.feasible_actions()))
            result = env.step(action)
            unsafe = result.outcome.unsafe or (result.done and action == AuthorityAction.AI and not result.outcome.success)
            transitions.append(_transition(encoder, state, action, result.reward, result.next_state, result.done, float(action == AuthorityAction.HUMAN), float(unsafe)))
            next_support_keys.append(support_key(result.next_state) if result.next_state else None)
            _record_support(support_counts, observed, state, action)
            if result.next_state is not None:
                state = result.next_state
    immutable = tuple(transitions)
    immutable_next_keys = tuple(next_support_keys)
    return FrozenLoggedReplay(
        immutable,
        dict(support_counts),
        immutable_next_keys,
        behavior_seed,
        "paired_counterfactual_branches" if paired_counterfactual else "uniform_feasible_logged_behavior",
        # Keep the legacy content digest stable for post-hoc support runs. The
        # next-state support keys are deterministic metadata for the new mode.
        replay_sha256(immutable),
    )


def _transition(
    encoder: FeatureEncoder,
    state,
    action: AuthorityAction,
    reward: float,
    next_state,
    done: bool,
    human_cost: float,
    unsafe_cost: float,
) -> ReplayTransition:
    next_vector = encoder.transform_one(next_state) if next_state else np.zeros(encoder.dimension, dtype=np.float32)
    mask = feasible_mask(next_state) if next_state else np.zeros(len(AuthorityAction), dtype=bool)
    return (encoder.transform_one(state), int(action), float(reward), next_vector, bool(done), mask, float(human_cost), float(unsafe_cost))


def feasible_mask(state) -> np.ndarray:
    mask = np.ones(len(AuthorityAction), dtype=bool)
    if state is not None and state.remaining_budget <= 0:
        mask[AuthorityAction.AI] = False
    return mask


def _record_support(counts: Counter[tuple[SupportKey, int]], observed: set[tuple[str, int, int]], state, action: AuthorityAction) -> None:
    observation = (state.episode_id, state.step, int(action))
    if observation not in observed:
        observed.add(observation)
        counts[(support_key(state), int(action))] += 1


def replay_sha256(transitions: Sequence[ReplayTransition], next_support_keys: Sequence[SupportKey | None] = ()) -> str:
    if next_support_keys and len(transitions) != len(next_support_keys):
        raise ValueError("next support keys must align with replay transitions")
    digest = hashlib.sha256()
    for index, (state, action, reward, next_state, done, mask, human, unsafe) in enumerate(transitions):
        digest.update(np.asarray(state, dtype=np.float32).tobytes())
        digest.update(struct.pack("!id?ff", int(action), float(reward), bool(done), float(human), float(unsafe)))
        digest.update(np.asarray(next_state, dtype=np.float32).tobytes())
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
        if next_support_keys:
            digest.update(repr(next_support_keys[index]).encode("utf-8"))
    return digest.hexdigest()
