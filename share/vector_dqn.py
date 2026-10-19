from __future__ import annotations

import copy
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dqn import DQNConfig, deterministic_action, feasible_mask, masked_conservative_argmax
from .environment import RepairAuthorityEnv
from .features import FeatureEncoder
from .policies import Policy
from .replay import FrozenLoggedReplay
from .support import SupportKey, mask_unsupported_ai_next_actions, support_key
from .types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


VectorTransition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray, float, float, SupportKey | None]


class VectorQNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], len(AuthorityAction) * 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).view(-1, len(AuthorityAction), 3)


@dataclass(frozen=True)
class VectorDQNConfig:
    dqn: DQNConfig
    human_dual: float
    unsafe_dual: float
    human_cost_discount: float = 1.0
    unsafe_cost_discount: float = 1.0

    def __post_init__(self) -> None:
        if self.human_dual < 0.0 or self.unsafe_dual < 0.0:
            raise ValueError("fixed dual values must be non-negative")
        if not 0.0 <= self.human_cost_discount <= 1.0 or not 0.0 <= self.unsafe_cost_discount <= 1.0:
            raise ValueError("cost discounts must be in [0, 1]")


class VectorDQNPolicy(Policy):
    """One policy-consistent, fixed-dual three-head Double-DQN critic."""

    name = "share_guard_vector"

    def __init__(self, encoder: FeatureEncoder, config: VectorDQNConfig, reward_config: RewardConfig | None = None):
        self.encoder = encoder
        self.config = config
        self.reward_config = reward_config or RewardConfig()
        self._seed_all(self.config.dqn.seed)
        torch.set_num_threads(max(1, self.config.dqn.torch_threads))
        self.device = torch.device(self.config.dqn.device)
        self.online = VectorQNetwork(encoder.dimension, self.config.dqn.hidden_sizes).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.dqn.learning_rate)
        self.replay: Deque[VectorTransition] = deque(maxlen=self.config.dqn.replay_capacity)
        self.logged_action_support: Counter[tuple[SupportKey, int]] = Counter()
        self._support_observations: set[tuple[str, int, int]] = set()
        self.decision_steps = 0
        self.gradient_steps = 0

    def fit(self, episodes: Sequence[RepairEpisode]) -> "VectorDQNPolicy":
        rng = random.Random(self.config.dqn.seed)
        if self.config.dqn.paired_replay:
            self._populate_paired_replay(episodes)
        for _ in range(self.config.dqn.training_epochs):
            shuffled = list(episodes)
            rng.shuffle(shuffled)
            if self.config.dqn.paired_replay:
                for _ in range(self.config.dqn.updates_per_epoch):
                    self._optimize()
                    self.decision_steps += 1
            else:
                self._collect_exploratory_replay(shuffled, rng)
        return self

    def fit_replay(self, frozen_replay: FrozenLoggedReplay, updates: int | None = None) -> "VectorDQNPolicy":
        """Fit from one candidate-independent replay shared by every comparator."""
        self.replay = deque(
            (transition + (next_key,))
            for transition, next_key in zip(frozen_replay.transitions, frozen_replay.next_support_keys)
        )
        self.logged_action_support = Counter(frozen_replay.support_counts)
        self._support_observations = set()
        total_updates = updates if updates is not None else self.config.dqn.training_epochs * max(1, len(self.replay))
        if total_updates < 0:
            raise ValueError("updates must be non-negative")
        for _ in range(total_updates):
            self._optimize()
            self.decision_steps += 1
        return self

    def _populate_paired_replay(self, episodes: Sequence[RepairEpisode]) -> None:
        for episode in episodes:
            for paired_step in episode.steps:
                state = paired_step.state
                for action in (AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP):
                    if action == AuthorityAction.AI and state.remaining_budget <= 0:
                        continue
                    outcome = paired_step.outcome(action)
                    next_state = episode.steps[outcome.next_step].state if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None else None
                    next_vector = self.encoder.transform_one(next_state) if next_state else np.zeros(self.encoder.dimension, dtype=np.float32)
                    mask = feasible_mask(next_state) if next_state else np.zeros(len(AuthorityAction), dtype=bool)
                    unsafe = outcome.unsafe or (outcome.terminal and action == AuthorityAction.AI and not outcome.success)
                    self.replay.append((self.encoder.transform_one(state), int(action), self._reward(outcome, action), next_vector, next_state is None, mask, float(action == AuthorityAction.HUMAN), float(unsafe), support_key(next_state) if next_state else None))
                    self._record_support(state, action)

    def _collect_exploratory_replay(self, episodes: Sequence[RepairEpisode], rng: random.Random) -> None:
        for episode in episodes:
            env = RepairAuthorityEnv(episode, self.reward_config)
            state = env.reset()
            while not env.done:
                feasible = env.feasible_actions()
                action = self._training_action(state, feasible, rng)
                result = env.step(action)
                next_vector = self.encoder.transform_one(result.next_state) if result.next_state else np.zeros(self.encoder.dimension, dtype=np.float32)
                mask = feasible_mask(result.next_state) if result.next_state else np.zeros(len(AuthorityAction), dtype=bool)
                unsafe = result.outcome.unsafe or (result.done and action == AuthorityAction.AI and not result.outcome.success)
                self.replay.append((self.encoder.transform_one(state), int(action), result.reward, next_vector, result.done, mask, float(action == AuthorityAction.HUMAN), float(unsafe), support_key(result.next_state) if result.next_state else None))
                self._record_support(state, action)
                self._optimize()
                self.decision_steps += 1
                if result.next_state is not None:
                    state = result.next_state

    def _reward(self, outcome, action: AuthorityAction) -> float:
        from .environment import base_reward

        return base_reward(outcome, action, self.reward_config)

    def _training_action(self, state: RepairState, feasible: Sequence[AuthorityAction], rng: random.Random) -> AuthorityAction:
        progress = min(1.0, self.decision_steps / max(1, self.config.dqn.epsilon_decay_steps))
        epsilon = self.config.dqn.epsilon_start + progress * (self.config.dqn.epsilon_end - self.config.dqn.epsilon_start)
        return rng.choice(list(feasible)) if rng.random() < epsilon else self.select(state, feasible)

    def _record_support(self, state: RepairState, action: AuthorityAction) -> None:
        observation = (state.episode_id, state.step, int(action))
        if observation not in self._support_observations:
            self._support_observations.add(observation)
            self.logged_action_support[(support_key(state), int(action))] += 1

    def _optimize(self) -> None:
        if len(self.replay) < self.config.dqn.batch_size:
            return
        batch = random.sample(self.replay, self.config.dqn.batch_size)
        states, actions, rewards, next_states, dones, masks, human_costs, unsafe_costs, next_keys = zip(*batch)
        state_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        next_masks = np.stack(masks)
        if self.config.dqn.support_mode == "predecision_mask":
            next_masks = mask_unsupported_ai_next_actions(next_masks, next_keys, self.logged_action_support, self.config.dqn.minimum_ai_support)
        mask_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        human_t = torch.as_tensor(human_costs, dtype=torch.float32, device=self.device)
        unsafe_t = torch.as_tensor(unsafe_costs, dtype=torch.float32, device=self.device)
        row_index = torch.arange(len(batch), device=self.device)
        predicted = self.online(state_t)[row_index, action_t]
        with torch.no_grad():
            online_next = self.online(next_t)
            scalar_next = self._scalarize(online_next).masked_fill(~mask_t, -torch.inf)
            next_actions = masked_conservative_argmax(scalar_next, mask_t)
            target_next = self.target(next_t)[row_index, next_actions]
            target_next = torch.where(done_t[:, None], torch.zeros_like(target_next), target_next)
            targets = torch.stack((
                reward_t + self.config.dqn.gamma * target_next[:, 0],
                human_t + self.config.human_cost_discount * target_next[:, 1],
                unsafe_t + self.config.unsafe_cost_discount * target_next[:, 2],
            ), dim=1)
        loss = sum(F.smooth_l1_loss(predicted[:, head], targets[:, head]) for head in range(3))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.dqn.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _raw_values(self, state: RepairState) -> np.ndarray:
        return self._raw_values_batch([state])[0]

    def _raw_values_batch(self, states: Sequence[RepairState], batch_size: int = 8192) -> np.ndarray:
        if not states:
            return np.empty((0, len(AuthorityAction), 3), dtype=np.float32)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                vectors = torch.as_tensor(
                    self.encoder.transform(states[start : start + batch_size]),
                    dtype=torch.float32,
                    device=self.device,
                )
                chunks.append(self.online(vectors).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _scalarize(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., 0] - self.config.human_dual * values[..., 1] - self.config.unsafe_dual * values[..., 2]

    def scalar_values(self, state: RepairState) -> np.ndarray:
        values = self._raw_values(state)
        return values[:, 0] - self.config.human_dual * values[:, 1] - self.config.unsafe_dual * values[:, 2]

    def scalar_values_batch(self, states: Sequence[RepairState]) -> np.ndarray:
        values = self._raw_values_batch(states)
        return values[:, :, 0] - self.config.human_dual * values[:, :, 1] - self.config.unsafe_dual * values[:, :, 2]

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        values = self.scalar_values(state)
        return deterministic_action(values, feasible)

    def escalation_score(self, state: RepairState) -> float:
        values = self.scalar_values(state)
        return float(values[AuthorityAction.HUMAN] - values[AuthorityAction.AI])

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
