from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CortexConfig


@dataclass(slots=True)
class LayerStep:
    state: np.ndarray
    prediction: np.ndarray
    error: np.ndarray
    mean_abs_error: float
    active: bool


class PredictiveCodingLayer:
    def __init__(self, in_dim: int, hidden_dim: int, lr: float, error_gate_threshold: float, seed: int = 0):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.error_gate_threshold = error_gate_threshold
        rng = np.random.default_rng(seed)
        self.W_in = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(hidden_dim, in_dim)).astype(np.float32)
        self.W_pred = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(in_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.b_p = np.zeros(in_dim, dtype=np.float32)
        self.state = np.zeros(hidden_dim, dtype=np.float32)

    def step(self, x: np.ndarray, topdown: np.ndarray | None = None) -> LayerStep:
        td = 0.0 if topdown is None else topdown
        self.state = np.maximum(0.0, self.W_in @ x + td + self.b_h)
        pred = self.W_pred @ self.state + self.b_p
        err = x - pred
        mae = float(np.mean(np.abs(err)))
        active = mae >= self.error_gate_threshold
        if active:
            # Local update only (no multilayer backprop).
            self.W_pred += self.lr * np.outer(err, self.state).astype(np.float32)
            d_state = (self.W_pred.T @ err) * (self.state > 0.0)
            self.W_in += self.lr * np.outer(d_state, x).astype(np.float32)
            self.b_h += self.lr * d_state.astype(np.float32)
            self.b_p += self.lr * err.astype(np.float32)
        return LayerStep(
            state=self.state.copy(),
            prediction=pred.astype(np.float32),
            error=err.astype(np.float32),
            mean_abs_error=mae,
            active=active,
        )


class RecurrentLayer6:
    def __init__(self, input_dim: int, hidden_dim: int, lr: float, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.R = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(hidden_dim, dtype=np.float32)
        self.h = np.zeros(hidden_dim, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        prev = self.h.copy()
        self.h = np.maximum(0.0, self.W @ x + self.R @ prev + self.b)
        # Small local recurrent update to reduce one-step prediction error.
        pred_prev = np.maximum(0.0, self.W @ x + self.R @ prev + self.b)
        err = self.h - pred_prev
        self.R += self.lr * np.outer(err, prev).astype(np.float32) * 0.1
        return self.h.copy()


class CorticalHierarchy:
    """
    Six-layer predictive coding cortex:
    - Layers 1..5 predictive coding
    - Layer 6 recurrent language / sequential context
    """

    def __init__(self, cfg: CortexConfig, seed: int = 0):
        self.cfg = cfg
        self.layers: list[PredictiveCodingLayer] = []
        in_dim = cfg.input_dim
        for i, hidden in enumerate(cfg.layer_dims):
            layer = PredictiveCodingLayer(
                in_dim=in_dim,
                hidden_dim=hidden,
                lr=cfg.learning_rate,
                error_gate_threshold=cfg.error_gate_threshold,
                seed=seed + i * 13,
            )
            self.layers.append(layer)
            in_dim = hidden
        self.layer6 = RecurrentLayer6(
            input_dim=128, hidden_dim=cfg.language_dim, lr=cfg.learning_rate, seed=seed + 101
        )
        self.last_errors = np.zeros(len(self.layers), dtype=np.float32)

    def step(
        self,
        visual_it: np.ndarray,
        auditory_a2: np.ndarray,
        thalamic_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        x = np.concatenate([visual_it, auditory_a2]).astype(np.float32)
        if x.shape[0] != self.cfg.input_dim:
            raise ValueError(f"Cortex expected input dim {self.cfg.input_dim}, got {x.shape[0]}")
        if thalamic_mask is not None:
            x = x * thalamic_mask

        layer6_state = self.layer6.step(auditory_a2)
        topdown = None
        outputs: list[LayerStep] = []
        cur = x
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                td = 0.2 * layer6_state[: layer.hidden_dim]
            else:
                td = topdown
            out = layer.step(cur, topdown=td)
            outputs.append(out)
            self.last_errors[i] = out.mean_abs_error
            if out.active:
                cur = out.state
                topdown = out.state * 0.05
            else:
                # Gated layer stays quiet: do not propagate error signal upwards.
                cur = np.zeros(layer.hidden_dim, dtype=np.float32)
                topdown = np.zeros(layer.hidden_dim, dtype=np.float32)

        return {
            "layer5": outputs[-1].state.astype(np.float32),
            "layer6": layer6_state.astype(np.float32),
            "mean_error": float(np.mean(self.last_errors)),
            "active_layers": float(np.sum([o.active for o in outputs])),
            "errors": self.last_errors.copy(),
        }
