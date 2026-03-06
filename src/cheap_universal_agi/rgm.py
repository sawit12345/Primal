from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import RGMLevelConfig


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-8)


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x) + 1e-8
    return x / norm


@dataclass(slots=True)
class RGMLevelState:
    posterior: np.ndarray
    reconstruction: np.ndarray
    reconstruction_error: float
    map_state: int


class SparseRGMLevel:
    """
    Lightweight discrete-state RGM level with sparse D and B structures.

    - D: sparse block-like latent-to-input mapping (learned rows)
    - B: transition counts over latent states per path
    - Online structure growth when reconstruction error remains high
    """

    def __init__(self, cfg: RGMLevelConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.input_dim: int | None = None
        self.n_states = cfg.n_states
        self.D: np.ndarray | None = None
        self.B = np.ones((cfg.n_paths, cfg.max_states, cfg.max_states), dtype=np.float32) * 1e-3
        self.active_mask = np.zeros(cfg.max_states, dtype=bool)
        self.active_mask[: self.n_states] = True

    def _init_if_needed(self, input_dim: int):
        if self.D is not None:
            return
        self.input_dim = input_dim
        self.D = np.zeros((self.cfg.max_states, input_dim), dtype=np.float32)
        for i in range(self.n_states):
            idx = self.rng.choice(input_dim, size=min(self.cfg.block_nonzero, input_dim), replace=False)
            row = np.zeros(input_dim, dtype=np.float32)
            row[idx] = self.rng.normal(0.0, 1.0, size=idx.size).astype(np.float32)
            self.D[i] = _normalize(row).astype(np.float32)

    def _active_rows(self) -> np.ndarray:
        assert self.D is not None
        return self.D[self.active_mask]

    def infer(self, x: np.ndarray) -> RGMLevelState:
        assert self.D is not None
        x_n = _normalize(x.astype(np.float32))
        D_act = self._active_rows()
        logits = D_act @ x_n
        posterior = _softmax(logits)
        reconstruction = posterior @ D_act
        err = float(np.mean((x_n - reconstruction) ** 2))
        map_state_local = int(np.argmax(posterior))
        map_state_global = int(np.flatnonzero(self.active_mask)[map_state_local])
        return RGMLevelState(
            posterior=posterior.astype(np.float32),
            reconstruction=reconstruction.astype(np.float32),
            reconstruction_error=err,
            map_state=map_state_global,
        )

    def update(
        self,
        x: np.ndarray,
        prev_state: int | None,
        path: int = 0,
        lr: float = 0.08,
    ) -> RGMLevelState:
        self._init_if_needed(x.shape[0])
        assert self.D is not None
        out = self.infer(x)
        x_n = _normalize(x.astype(np.float32))

        # Local free-energy style update (Hebbian-like toward current evidence).
        winner = out.map_state
        self.D[winner] = _normalize((1.0 - lr) * self.D[winner] + lr * x_n)

        # Optional update for runner-up for smoother adaptation.
        active_ids = np.flatnonzero(self.active_mask)
        local_logits = self._active_rows() @ x_n
        if local_logits.shape[0] > 1:
            second_local = int(np.argpartition(local_logits, -2)[-2])
            second_global = int(active_ids[second_local])
            self.D[second_global] = _normalize((1.0 - 0.25 * lr) * self.D[second_global] + (0.25 * lr) * x_n)

        # Transition learning for path-conditioned B tensor.
        if prev_state is not None:
            self.B[path, prev_state, winner] += 1.0

        # Structure growth if current representation cannot explain input.
        if out.reconstruction_error > self.cfg.growth_threshold and self.n_states < self.cfg.max_states:
            self._grow_state(x_n)
            out = self.infer(x)
        return out

    def _grow_state(self, x_n: np.ndarray):
        free_slots = np.flatnonzero(~self.active_mask)
        if free_slots.size == 0:
            return
        new_id = int(free_slots[0])
        self.D[new_id] = x_n
        self.active_mask[new_id] = True
        self.n_states += 1


class HierarchicalRGM:
    def __init__(self, level_cfgs: tuple[RGMLevelConfig, ...], seed: int = 0):
        self.levels = [
            SparseRGMLevel(cfg=cfg, seed=seed + i * 17) for i, cfg in enumerate(level_cfgs)
        ]

    def encode(self, x: np.ndarray, train: bool = False) -> tuple[list[RGMLevelState], np.ndarray]:
        states: list[RGMLevelState] = []
        cur = x.astype(np.float32)
        prev_map: int | None = None
        for level in self.levels:
            if train:
                st = level.update(cur, prev_state=prev_map, path=0)
            else:
                level._init_if_needed(cur.shape[0])
                st = level.infer(cur)
            states.append(st)
            # states generate paths/states recursively: next level sees posterior over this level
            cur = st.posterior
            prev_map = st.map_state
        return states, cur

    def fit(self, data: Iterable[np.ndarray], epochs: int = 1) -> dict[str, float]:
        recon_errors: list[float] = []
        count = 0
        for _ in range(epochs):
            for x in data:
                states, _ = self.encode(x, train=True)
                recon_errors.extend([s.reconstruction_error for s in states])
                count += 1
        return {
            "samples": float(count),
            "mean_reconstruction_error": float(np.mean(recon_errors)) if recon_errors else 0.0,
            "final_state_counts": float(sum(level.n_states for level in self.levels)),
        }

    def top_belief(self, x: np.ndarray) -> np.ndarray:
        _, top = self.encode(x, train=False)
        return top
