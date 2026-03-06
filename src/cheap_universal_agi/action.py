from __future__ import annotations

import numpy as np

from .config import ActionConfig


class BasalGangliaTDLambda:
    def __init__(self, cfg: ActionConfig, state_dim: int = 256, goal_dim: int = 64, seed: int = 0):
        self.cfg = cfg
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.n_actions = cfg.n_actions
        rng = np.random.default_rng(seed)
        self.W_value = rng.normal(0.0, 1.0 / np.sqrt(state_dim), size=(self.n_actions, state_dim)).astype(
            np.float32
        )
        self.goal_proj = rng.normal(0.0, 1.0 / np.sqrt(goal_dim), size=(self.n_actions, goal_dim)).astype(
            np.float32
        )
        self.suppression = np.full(self.n_actions, 0.5, dtype=np.float32)
        self.eligibility = np.zeros((self.n_actions, state_dim), dtype=np.float32)
        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.prev_value: float = 0.0

    def values(self, state: np.ndarray, goal: np.ndarray, valence_offset: np.ndarray | None = None) -> np.ndarray:
        v = self.W_value @ state.astype(np.float32)
        v += self.goal_proj @ goal.astype(np.float32)
        if valence_offset is not None:
            v = v + valence_offset.astype(np.float32)
        return v

    def select_action(
        self, state: np.ndarray, goal: np.ndarray, valence_offset: np.ndarray | None = None
    ) -> tuple[int, np.ndarray]:
        v = self.values(state, goal, valence_offset=valence_offset)
        scores = v - self.suppression
        action = int(np.argmax(scores))
        self.prev_state = state.copy()
        self.prev_action = action
        self.prev_value = float(v[action])
        return action, scores.astype(np.float32)

    def td_update(self, reward: float, next_state: np.ndarray, goal: np.ndarray) -> float:
        if self.prev_state is None or self.prev_action is None:
            return 0.0
        next_values = self.values(next_state, goal)
        td_target = reward + self.cfg.gamma * float(np.max(next_values))
        delta = td_target - self.prev_value

        self.eligibility *= self.cfg.gamma * self.cfg.td_lambda
        self.eligibility[self.prev_action] += self.prev_state
        self.W_value += 0.01 * delta * self.eligibility

        # Learn suppression thresholds to avoid always-on actions.
        self.suppression[self.prev_action] += -0.005 * delta
        self.suppression = np.clip(self.suppression, -1.0, 1.5)
        return float(delta)
