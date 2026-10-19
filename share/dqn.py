from __future__ import annotations

import copy
import os
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Deque, List, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .environment import RepairAuthorityEnv, base_reward
from .features import FeatureEncoder
from .policies import Policy
from .replay import FrozenLoggedReplay
from .support import SupportKey, mask_unsupported_ai_next_actions, support_key
from .types import AuthorityAction, RepairEpisode, RepairState, RewardConfig


@dataclass
class DQNConfig:
    hidden_sizes: Tuple[int, int] = (128, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.95
    replay_capacity: int = 50_000
    batch_size: int = 64
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    dual_learning_rate_human: float = 5e-3
    dual_learning_rate_unsafe: float = 5e-3
    human_budget: float = 0.2
    unsafe_budget: float = 0.05
    training_epochs: int = 30
    constrained: bool = True
    human_constraint: bool = True
    unsafe_constraint: bool = True
    double_dqn: bool = True
    seed: int = 0
    torch_threads: int = 1
    paired_replay: bool = False
    updates_per_epoch: int = 1_000
    constraint_confidence_z: float = 1.64
    fixed_human_dual: float = 0.0
    fixed_unsafe_dual: float = 0.0
    checkpoint_selection: bool = True
    support_mode: str = "posthoc_stop"
    minimum_ai_support: int = 3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.support_mode not in {"posthoc_stop", "predecision_mask", "calibrated_router", "joint_calibrated"}:
            raise ValueError("support_mode must be posthoc_stop, predecision_mask, calibrated_router, or joint_calibrated")
        if self.minimum_ai_support <= 0:
            raise ValueError("minimum_ai_support must be positive")
        if self.device != "cpu" and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")


class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], len(AuthorityAction)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray, float, float, SupportKey | None]

# Equal scores use the predeclared conservative order Stop, Human, then AI.
ACTION_TIE_BREAK = {
    AuthorityAction.STOP: 0,
    AuthorityAction.HUMAN: 1,
    AuthorityAction.AI: 2,
}


def deterministic_action(values: np.ndarray, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
    return max(feasible, key=lambda action: (float(values[action]), -ACTION_TIE_BREAK[action]))


def masked_conservative_argmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batch action argmax with the same Stop, Human, AI exact-tie order."""
    best = torch.full((values.shape[0],), -torch.inf, dtype=values.dtype, device=values.device)
    selected = torch.full((values.shape[0],), int(AuthorityAction.STOP), dtype=torch.int64, device=values.device)
    for action in (AuthorityAction.AI, AuthorityAction.HUMAN, AuthorityAction.STOP):
        admissible = mask[:, action]
        better_or_tied = admissible & (values[:, action] >= best)
        best = torch.where(better_or_tied, values[:, action], best)
        selected = torch.where(better_or_tied, torch.full_like(selected, int(action)), selected)
    return selected


class DQNPolicy(Policy):
    name = "share_controller"

    def __init__(self, encoder: FeatureEncoder, config: DQNConfig | None = None, reward_config: RewardConfig | None = None):
        self.encoder = encoder
        self.config = config or DQNConfig()
        self.reward_config = reward_config or RewardConfig()
        self._seed_all(self.config.seed)
        torch.set_num_threads(max(1, self.config.torch_threads))
        self.device = torch.device(self.config.device)
        self.online = QNetwork(encoder.dimension, self.config.hidden_sizes).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay: Deque[Transition] = deque(maxlen=self.config.replay_capacity)
        self.beta_h = float(self.config.fixed_human_dual)
        self.beta_u = float(self.config.fixed_unsafe_dual)
        self.gradient_steps = 0
        self.decision_steps = 0
        self.human_calibration_penalty = 0.0
        self.ai_calibration_penalty = 0.0
        self.feasible_checkpoint_found = False
        self.calibration_human_target = self.config.human_budget
        self.calibration_unsafe_target = self.config.unsafe_budget
        self.calibration_validation_hir = float("nan")
        self.calibration_validation_unsafe = float("nan")
        self.logged_action_support: Counter[tuple[SupportKey, int]] = Counter()
        self._support_observations: set[tuple[str, int, int]] = set()
        self.training_history: list[dict[str, float | int | str]] = []
        self.selected_checkpoint_diagnostics: dict[str, float | int | str] | None = None

    def fit(self, episodes: Sequence[RepairEpisode], validation: Sequence[RepairEpisode] = ()) -> "DQNPolicy":
        rng = random.Random(self.config.seed)
        selection = validation or episodes
        best_key: tuple[float, float] | None = None
        best_state = None
        if self.config.paired_replay:
            self._populate_paired_replay(episodes)
        for epoch in range(1, self.config.training_epochs + 1):
            shuffled = list(episodes)
            rng.shuffle(shuffled)
            if self.config.paired_replay:
                for _ in range(self.config.updates_per_epoch):
                    self._optimize()
                    self.decision_steps += 1
            else:
                self._collect_exploratory_replay(shuffled, rng)
            if self.config.constrained and shuffled:
                rate_h, rate_u = self._greedy_constraint_rates(shuffled)
                if self.config.human_constraint:
                    self.beta_h = max(0.0, self.beta_h + self.config.dual_learning_rate_human * (rate_h - self.config.human_budget))
                if self.config.unsafe_constraint:
                    self.beta_u = max(0.0, self.beta_u + self.config.dual_learning_rate_unsafe * (rate_u - self.config.unsafe_budget))
            diagnostics = self.policy_diagnostics(selection, epoch=epoch, phase="epoch_end")
            self.training_history.append(diagnostics)
            if self.config.checkpoint_selection:
                val_h = float(diagnostics["hir"])
                val_u = float(diagnostics["unsafe_autonomy_cost_rate"])
                val_return = float(diagnostics["normalized_return"])
                violation = 0.0
                if self.config.constrained and self.config.human_constraint:
                    violation += max(0.0, val_h - self.config.human_budget)
                if self.config.constrained and self.config.unsafe_constraint:
                    violation += max(0.0, val_u - self.config.unsafe_budget)
                feasible = violation <= 1e-12
                key = (1.0 if feasible else -violation, val_return)
                if best_key is None or key > best_key:
                    best_key = key
                    best_state = copy.deepcopy(self.online.state_dict())
                    self.feasible_checkpoint_found = feasible
        if best_state is not None:
            self.online.load_state_dict(best_state)
            self.target.load_state_dict(best_state)
        self.selected_checkpoint_diagnostics = self.policy_diagnostics(
            selection, epoch=self.config.training_epochs, phase="selected_checkpoint_pre_calibration"
        )
        return self

    def fit_replay(self, frozen_replay: FrozenLoggedReplay, updates: int | None = None) -> "DQNPolicy":
        """Fit a fixed-dual candidate from externally frozen logged transitions."""
        if self.config.constrained:
            raise ValueError("fit_replay is for fixed-dual candidates; set constrained=False")
        self.replay = deque(
            (transition + (next_key,))
            for transition, next_key in zip(frozen_replay.transitions, frozen_replay.next_support_keys)
        )
        self.logged_action_support = Counter(frozen_replay.support_counts)
        self._support_observations = set()
        total_updates = updates if updates is not None else self.config.training_epochs * max(1, len(self.replay))
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
                feasible = [AuthorityAction.HUMAN, AuthorityAction.STOP]
                if state.remaining_budget > 0:
                    feasible.insert(0, AuthorityAction.AI)
                for action in feasible:
                    outcome = paired_step.outcome(action)
                    next_state = episode.steps[outcome.next_step].state if action == AuthorityAction.AI and not outcome.terminal and outcome.next_step is not None else None
                    next_vector = self.encoder.transform_one(next_state) if next_state else np.zeros(self.encoder.dimension, dtype=np.float32)
                    mask = feasible_mask(next_state) if next_state else np.zeros(len(AuthorityAction), dtype=bool)
                    reward = base_reward(outcome, action, self.reward_config)
                    unsafe = outcome.unsafe or (outcome.terminal and action == AuthorityAction.AI and not outcome.success)
                    self.replay.append((self.encoder.transform_one(state), int(action), reward, next_vector, next_state is None, mask, float(action == AuthorityAction.HUMAN), float(unsafe), support_key(next_state) if next_state else None))
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
                unsafe = result.outcome.unsafe or (result.done and not result.outcome.success and action == AuthorityAction.AI)
                self.replay.append((self.encoder.transform_one(state), int(action), result.reward, next_vector, result.done, mask, float(action == AuthorityAction.HUMAN), float(unsafe), support_key(result.next_state) if result.next_state else None))
                self._record_support(state, action)
                self._optimize()
                self.decision_steps += 1
                if result.next_state is not None:
                    state = result.next_state

    def _training_action(self, state: RepairState, feasible: Sequence[AuthorityAction], rng: random.Random) -> AuthorityAction:
        progress = min(1.0, self.decision_steps / max(1, self.config.epsilon_decay_steps))
        epsilon = self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
        return rng.choice(list(feasible)) if rng.random() < epsilon else self.select(state, feasible)

    def _record_support(self, state: RepairState, action: AuthorityAction) -> None:
        observation = (state.episode_id, state.step, int(action))
        if observation not in self._support_observations:
            self._support_observations.add(observation)
            self.logged_action_support[(support_key(state), int(action))] += 1

    def _optimize(self) -> None:
        if len(self.replay) < self.config.batch_size:
            return
        batch = random.sample(self.replay, self.config.batch_size)
        if len(batch[0]) == 8:
            # Observed-event longitudinal replay predates support-key metadata.
            states, actions, base_rewards, next_states, dones, masks, human_costs, unsafe_costs = zip(*batch)
            next_keys = (None,) * len(batch)
        else:
            states, actions, base_rewards, next_states, dones, masks, human_costs, unsafe_costs, next_keys = zip(*batch)
        state_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(base_rewards, dtype=torch.float32, device=self.device)
        reward_t -= self.beta_h * torch.as_tensor(human_costs, dtype=torch.float32, device=self.device)
        reward_t -= self.beta_u * torch.as_tensor(unsafe_costs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks = np.stack(masks)
        if self.config.support_mode == "predecision_mask":
            next_masks = mask_unsupported_ai_next_actions(next_masks, next_keys, self.logged_action_support, self.config.minimum_ai_support)
        mask_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        predicted = self.online(state_t).gather(1, action_t[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_t).masked_fill(~mask_t, -torch.inf)
            if self.config.double_dqn:
                next_actions = masked_conservative_argmax(online_next, mask_t)
                target_values = self.target(next_t).gather(1, next_actions[:, None]).squeeze(1)
            else:
                target_values = self.target(next_t).masked_fill(~mask_t, -torch.inf).max(dim=1).values
            target_values = torch.where(done_t.bool(), torch.zeros_like(target_values), target_values)
            targets = reward_t + self.config.gamma * (1.0 - done_t) * target_values
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _raw_values(self, state: RepairState) -> np.ndarray:
        return self._raw_values_batch([state])[0]

    def _raw_values_batch(self, states: Sequence[RepairState], batch_size: int = 8192) -> np.ndarray:
        if not states:
            return np.empty((0, len(AuthorityAction)), dtype=np.float32)
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

    def values(self, state: RepairState) -> np.ndarray:
        values = self._raw_values(state)
        values[AuthorityAction.HUMAN] -= self.human_calibration_penalty
        values[AuthorityAction.AI] -= self.ai_calibration_penalty
        return values

    def values_batch(self, states: Sequence[RepairState]) -> np.ndarray:
        values = self._raw_values_batch(states)
        values[:, AuthorityAction.HUMAN] -= self.human_calibration_penalty
        values[:, AuthorityAction.AI] -= self.ai_calibration_penalty
        return values

    def calibrate_constraints(self, episodes: Sequence[RepairEpisode], human_budget: float, unsafe_budget: float, grid_size: int = 25) -> "DQNPolicy":
        """Joint validation calibration of authority costs; never reads test outcomes."""
        if not episodes:
            return self
        count = len(episodes)
        z = max(0.0, self.config.constraint_confidence_z)
        enforce_human = self.config.constrained and self.config.human_constraint
        enforce_unsafe = self.config.constrained and self.config.unsafe_constraint
        self.calibration_human_target = max(0.0, human_budget - z * np.sqrt(human_budget * (1.0 - human_budget) / count)) if enforce_human else 1.0
        self.calibration_unsafe_target = max(0.0, unsafe_budget - z * np.sqrt(unsafe_budget * (1.0 - unsafe_budget) / count)) if enforce_unsafe else 1.0
        raw_values = [self._raw_values(step.state) for episode in episodes for step in episode.steps]
        human_gaps = np.asarray([value[AuthorityAction.HUMAN] - value[AuthorityAction.STOP] for value in raw_values])
        ai_gaps = np.asarray([value[AuthorityAction.AI] - value[AuthorityAction.STOP] for value in raw_values])
        quantiles = np.linspace(0.0, 1.0, grid_size)
        human_candidates = np.unique(np.concatenate(([0.0], np.quantile(human_gaps, quantiles), [float(human_gaps.max() + 1.0)]))) if enforce_human else np.asarray([0.0])
        ai_candidates = np.unique(np.concatenate(([0.0], np.quantile(ai_gaps, quantiles), [float(ai_gaps.max() + 1.0)]))) if enforce_unsafe else np.asarray([0.0])
        best = None
        fallback = None
        original = (self.human_calibration_penalty, self.ai_calibration_penalty)
        for human_penalty in human_candidates:
            for ai_penalty in ai_candidates:
                self.human_calibration_penalty = float(human_penalty)
                self.ai_calibration_penalty = float(ai_penalty)
                hir, unsafe, value = self._greedy_stats(episodes)
                violation = 0.0
                if enforce_human:
                    violation += max(0.0, hir - self.calibration_human_target)
                if enforce_unsafe:
                    violation += max(0.0, unsafe - self.calibration_unsafe_target)
                candidate = (value, -hir, -unsafe, float(human_penalty), float(ai_penalty))
                if violation <= 1e-12 and (best is None or candidate > best):
                    best = candidate
                fallback_candidate = (-violation, value, -hir, -unsafe, float(human_penalty), float(ai_penalty))
                if fallback is None or fallback_candidate > fallback:
                    fallback = fallback_candidate
        if best is not None:
            self.human_calibration_penalty = best[3]
            self.ai_calibration_penalty = best[4]
            self.feasible_checkpoint_found = True
        elif fallback is not None:
            self.human_calibration_penalty = fallback[4]
            self.ai_calibration_penalty = fallback[5]
            self.feasible_checkpoint_found = False
        else:
            self.human_calibration_penalty, self.ai_calibration_penalty = original
        self.calibration_validation_hir, self.calibration_validation_unsafe, _ = self._greedy_stats(episodes)
        return self

    def calibrate_human_budget(self, episodes: Sequence[RepairEpisode], target_rate: float, tolerance: float = 0.005) -> "DQNPolicy":
        """Validation-only calibration allowed by the paper's budget-matched protocol."""
        self.calibration_human_target = target_rate
        self.calibration_unsafe_target = 1.0
        if target_rate >= 1:
            self.human_calibration_penalty = -1e6
            self.calibration_validation_hir, self.calibration_validation_unsafe, _ = self._greedy_stats(episodes)
            self.feasible_checkpoint_found = True
            return self
        if target_rate <= 0:
            self.human_calibration_penalty = 1e6
            self.calibration_validation_hir, self.calibration_validation_unsafe, _ = self._greedy_stats(episodes)
            self.feasible_checkpoint_found = self.calibration_validation_hir <= target_rate + 1e-12
            return self
        raw_advantages = []
        with torch.no_grad():
            for episode in episodes:
                for step in episode.steps:
                    vector = torch.as_tensor(self.encoder.transform([step.state]), dtype=torch.float32, device=self.device)
                    values = self.online(vector).squeeze(0).detach().cpu().numpy()
                    raw_advantages.append(float(values[AuthorityAction.HUMAN] - max(values[AuthorityAction.AI], values[AuthorityAction.STOP])))
        if not raw_advantages:
            return self
        low = min(raw_advantages) - 1.0
        high = max(raw_advantages) + 1.0
        best_penalty = high
        best_error = float("inf")
        for _ in range(40):
            candidate = (low + high) / 2.0
            self.human_calibration_penalty = candidate
            rate = self._validation_hir(episodes)
            error = abs(rate - target_rate)
            if error < best_error:
                best_error = error
                best_penalty = candidate
            if rate > target_rate:
                low = candidate
            else:
                high = candidate
            if error <= tolerance:
                break
        self.human_calibration_penalty = best_penalty
        self.calibration_validation_hir, self.calibration_validation_unsafe, _ = self._greedy_stats(episodes)
        self.feasible_checkpoint_found = self.calibration_validation_hir <= target_rate + tolerance
        return self

    def _validation_hir(self, episodes: Sequence[RepairEpisode]) -> float:
        human_episodes = 0
        for episode in episodes:
            env = RepairAuthorityEnv(episode, self.reward_config)
            state = env.reset()
            used = False
            while not env.done:
                action = self.select(state, env.feasible_actions())
                used = used or action == AuthorityAction.HUMAN
                result = env.step(action)
                if result.next_state is not None:
                    state = result.next_state
            human_episodes += int(used)
        return human_episodes / len(episodes) if episodes else 0.0

    def _greedy_constraint_rates(self, episodes: Sequence[RepairEpisode]) -> tuple[float, float]:
        human, unsafe, _ = self._greedy_stats(episodes)
        return human, unsafe

    def _greedy_stats(self, episodes: Sequence[RepairEpisode]) -> tuple[float, float, float]:
        human = unsafe = 0
        total_return = 0.0
        for episode in episodes:
            env = RepairAuthorityEnv(episode, self.reward_config)
            state = env.reset()
            used = False
            while not env.done:
                action = self.select(state, env.feasible_actions())
                result = env.step(action)
                total_return += result.reward
                used = used or action == AuthorityAction.HUMAN
                unsafe += int(result.outcome.unsafe or (result.done and action == AuthorityAction.AI and not result.outcome.success))
                if result.next_state is not None:
                    state = result.next_state
            human += int(used)
        count = max(1, len(episodes))
        return human / count, unsafe / count, total_return / count

    def policy_diagnostics(
        self,
        episodes: Sequence[RepairEpisode],
        *,
        epoch: int,
        phase: str,
    ) -> dict[str, float | int | str]:
        """Summarize greedy actions, Q values, and duals on a declared split."""
        action_counts = Counter({action: 0 for action in AuthorityAction})
        raw_values: list[np.ndarray] = []
        adjusted_values: list[np.ndarray] = []
        human_episodes = unsafe_actions = stopped_episodes = successes = 0
        total_return = 0.0
        for episode in episodes:
            env = RepairAuthorityEnv(episode, self.reward_config)
            state = env.reset()
            used_human = stopped = False
            final_success = False
            while not env.done:
                raw = self._raw_values(state)
                adjusted = raw.copy()
                adjusted[AuthorityAction.HUMAN] -= self.human_calibration_penalty
                adjusted[AuthorityAction.AI] -= self.ai_calibration_penalty
                action = deterministic_action(adjusted, env.feasible_actions())
                raw_values.append(raw)
                adjusted_values.append(adjusted)
                action_counts[action] += 1
                used_human = used_human or action == AuthorityAction.HUMAN
                stopped = stopped or action == AuthorityAction.STOP
                result = env.step(action)
                total_return += result.reward
                final_success = result.outcome.success
                unsafe_actions += int(
                    result.outcome.unsafe
                    or (result.done and action == AuthorityAction.AI and not result.outcome.success)
                )
                if result.next_state is not None:
                    state = result.next_state
            human_episodes += int(used_human)
            stopped_episodes += int(stopped)
            successes += int(final_success)
        decision_states = sum(action_counts.values())
        count = max(1, len(episodes))
        raw_mean = np.mean(np.stack(raw_values), axis=0) if raw_values else np.zeros(len(AuthorityAction))
        adjusted_mean = (
            np.mean(np.stack(adjusted_values), axis=0)
            if adjusted_values
            else np.zeros(len(AuthorityAction))
        )
        return {
            "epoch": int(epoch),
            "phase": phase,
            "episodes": len(episodes),
            "decision_states": decision_states,
            "repair_sr": successes / count,
            "hir": human_episodes / count,
            "unsafe_autonomy_cost_rate": unsafe_actions / count,
            "stopped_rate": stopped_episodes / count,
            "normalized_return": total_return / count,
            "p_ai": action_counts[AuthorityAction.AI] / max(1, decision_states),
            "p_human": action_counts[AuthorityAction.HUMAN] / max(1, decision_states),
            "p_stop": action_counts[AuthorityAction.STOP] / max(1, decision_states),
            "q_ai_raw_mean": float(raw_mean[AuthorityAction.AI]),
            "q_human_raw_mean": float(raw_mean[AuthorityAction.HUMAN]),
            "q_stop_raw_mean": float(raw_mean[AuthorityAction.STOP]),
            "q_ai_adjusted_mean": float(adjusted_mean[AuthorityAction.AI]),
            "q_human_adjusted_mean": float(adjusted_mean[AuthorityAction.HUMAN]),
            "q_stop_adjusted_mean": float(adjusted_mean[AuthorityAction.STOP]),
            "beta_h": float(self.beta_h),
            "beta_u": float(self.beta_u),
            "human_calibration_penalty": float(self.human_calibration_penalty),
            "ai_calibration_penalty": float(self.ai_calibration_penalty),
        }

    def select(self, state: RepairState, feasible: Sequence[AuthorityAction]) -> AuthorityAction:
        values = self.values(state)
        return deterministic_action(values, feasible)

    def escalation_score(self, state: RepairState) -> float:
        values = self.values(state)
        return float(values[AuthorityAction.HUMAN] - values[AuthorityAction.AI])

    def save(self, path: str) -> None:
        config = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in self.config.__dict__.items()
        }
        torch.save({"model": self.online.state_dict(), "config": config, "reward_config": asdict(self.reward_config), "beta_h": float(self.beta_h), "beta_u": float(self.beta_u), "human_calibration_penalty": float(self.human_calibration_penalty), "ai_calibration_penalty": float(self.ai_calibration_penalty), "feasible_checkpoint_found": bool(self.feasible_checkpoint_found), "calibration_human_target": float(self.calibration_human_target), "calibration_unsafe_target": float(self.calibration_unsafe_target), "calibration_validation_hir": float(self.calibration_validation_hir), "calibration_validation_unsafe": float(self.calibration_validation_unsafe)}, path)

    @classmethod
    def load(
        cls,
        path: str,
        encoder: FeatureEncoder,
        reward_config: RewardConfig | None = None,
        *,
        trusted_checkpoint: bool = False,
        require_reward_config: bool = False,
        device: str | None = None,
    ) -> "DQNPolicy":
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            if not trusted_checkpoint:
                raise
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if require_reward_config and "reward_config" not in checkpoint:
            raise ValueError("checkpoint lacks the required serialized reward config")
        stored_reward = RewardConfig(**checkpoint["reward_config"]) if "reward_config" in checkpoint else RewardConfig()
        if reward_config is not None and "reward_config" in checkpoint and stored_reward != reward_config:
            raise ValueError("checkpoint reward config differs from requested reward config")
        config = dict(checkpoint["config"])
        if device is not None:
            config["device"] = device
        elif str(config.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            config["device"] = "cpu"
        policy = cls(encoder, DQNConfig(**config), reward_config or stored_reward)
        policy.online.load_state_dict(checkpoint["model"])
        policy.target.load_state_dict(checkpoint["model"])
        policy.beta_h = float(checkpoint.get("beta_h", 0.0))
        policy.beta_u = float(checkpoint.get("beta_u", 0.0))
        policy.human_calibration_penalty = float(checkpoint.get("human_calibration_penalty", 0.0))
        policy.ai_calibration_penalty = float(checkpoint.get("ai_calibration_penalty", 0.0))
        policy.feasible_checkpoint_found = bool(checkpoint.get("feasible_checkpoint_found", False))
        policy.calibration_human_target = float(checkpoint.get("calibration_human_target", policy.config.human_budget))
        policy.calibration_unsafe_target = float(checkpoint.get("calibration_unsafe_target", policy.config.unsafe_budget))
        policy.calibration_validation_hir = float(checkpoint.get("calibration_validation_hir", "nan"))
        policy.calibration_validation_unsafe = float(checkpoint.get("calibration_validation_unsafe", "nan"))
        return policy

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def feasible_mask(state: RepairState | None) -> np.ndarray:
    mask = np.ones(len(AuthorityAction), dtype=bool)
    if state is not None and state.remaining_budget <= 0:
        mask[AuthorityAction.AI] = False
    return mask


class DQNEnsemblePolicy(DQNPolicy):
    name = "share_ensemble"

    def __init__(self, members: Sequence[DQNPolicy], encoder: FeatureEncoder, config: DQNConfig | None = None, reward_config: RewardConfig | None = None):
        if not members:
            raise ValueError("ensemble requires at least one member")
        super().__init__(encoder, config or members[0].config, reward_config or members[0].reward_config)
        self.members = list(members)

    def _raw_values(self, state: RepairState) -> np.ndarray:
        return np.mean(np.stack([member._raw_values(state) for member in self.members]), axis=0)

    def _raw_values_batch(self, states: Sequence[RepairState], batch_size: int = 8192) -> np.ndarray:
        """Average member predictions for batched and scalar inference alike.

        The ensemble owns an unused network through ``DQNPolicy`` for API
        compatibility.  Falling back to the parent batch implementation would
        therefore return that random network rather than the member average.
        """
        if not states:
            return np.empty((0, len(AuthorityAction)), dtype=np.float32)
        return np.mean(
            np.stack(
                [member._raw_values_batch(states, batch_size=batch_size) for member in self.members]
            ),
            axis=0,
        )
