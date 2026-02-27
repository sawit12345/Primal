"""Active inference core: predictive coding, free energy, and action generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(temperature, 1e-4)
    shifted = logits / temp
    shifted -= np.max(shifted)
    probs = np.exp(shifted)
    return probs / (np.sum(probs) + 1e-12)


@dataclass
class PredictiveCodingLayer:
    """Single-layer predictive coding encoder/decoder."""

    input_dim: int
    latent_dim: int
    learning_rate: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.encoder = rng.normal(0.0, 0.12, size=(self.latent_dim, self.input_dim))
        self.decoder = rng.normal(0.0, 0.12, size=(self.input_dim, self.latent_dim))

    def encode(self, observation: np.ndarray, learn: bool = False) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation, dtype=np.float64).ravel()
        latent = np.tanh(self.encoder @ observation)
        reconstruction = self.decoder @ latent
        prediction_error = observation - reconstruction

        if learn:
            self.encoder += self.learning_rate * np.outer(
                (1.0 - latent**2) * (self.decoder.T @ prediction_error), observation
            )
            self.decoder += self.learning_rate * np.outer(prediction_error, latent)

        return latent, prediction_error


@dataclass
class FreeEnergyEngine:
    """Computes variational and expected free energy surrogates."""

    complexity_weight: float = 0.1
    ambiguity_weight: float = 0.25
    epistemic_weight: float = 0.2

    def variational_free_energy(self, prediction_error: np.ndarray, precision: np.ndarray | float) -> float:
        precision_array = np.asarray(precision, dtype=np.float64)
        pe = np.asarray(prediction_error, dtype=np.float64)
        weighted_error = np.sum(precision_array * (pe**2))
        complexity = self.complexity_weight * np.log1p(np.sum(np.abs(precision_array)))
        return float(0.5 * weighted_error + complexity)

    def expected_free_energy(
        self,
        risk: float,
        ambiguity: float,
        epistemic_value: float,
        urgency_alpha: float,
    ) -> float:
        return float(
            urgency_alpha * risk
            + self.ambiguity_weight * ambiguity
            - self.epistemic_weight * epistemic_value
        )


@dataclass
class RecursiveLinearDynamics:
    """Fast action-conditioned world model learned via recursive least squares."""

    state_dim: int
    action_dim: int
    forgetting: float = 0.995
    prior_covariance: float = 500.0

    def __post_init__(self) -> None:
        input_dim = self.state_dim + self.action_dim + 1
        self.theta = np.zeros((self.state_dim, input_dim), dtype=np.float64)
        self.covariance = np.eye(input_dim, dtype=np.float64) * self.prior_covariance
        self.residual_var = np.ones(self.state_dim, dtype=np.float64)

    def _compose(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action, np.ones(1, dtype=np.float64)], axis=0)

    def predict(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        x = self._compose(state, action)
        prediction = self.theta @ x
        uncertainty = np.abs(self.residual_var)
        return prediction, uncertainty

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        next_state = np.asarray(next_state, dtype=np.float64).ravel()

        x = self._compose(state, action)
        px = self.covariance @ x
        denom = self.forgetting + float(x.T @ px)
        gain = px / (denom + 1e-12)

        prediction = self.theta @ x
        error = next_state - prediction

        self.theta += np.outer(error, gain)
        self.covariance = (self.covariance - np.outer(gain, x) @ self.covariance) / self.forgetting
        self.residual_var = 0.95 * self.residual_var + 0.05 * (error**2)
        return error


@dataclass
class ActionGenerator:
    """Selects actions that minimize expected free energy."""

    free_energy: FreeEnergyEngine = field(default_factory=FreeEnergyEngine)

    def choose(
        self,
        state: np.ndarray,
        candidate_actions: np.ndarray,
        model: RecursiveLinearDynamics,
        target_state: np.ndarray,
        ambiguity: float,
        urgency_alpha: float,
        temperature: float,
    ) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
        scores = []
        predictions = []
        epistemic_values = []

        for action in candidate_actions:
            pred_next, uncertainty = model.predict(state, action)
            risk = float(np.mean((pred_next - target_state) ** 2))
            epistemic = float(np.mean(np.sqrt(uncertainty)))
            efe = self.free_energy.expected_free_energy(
                risk=risk,
                ambiguity=ambiguity,
                epistemic_value=epistemic,
                urgency_alpha=urgency_alpha,
            )
            scores.append(-efe)
            predictions.append(pred_next)
            epistemic_values.append(epistemic)

        logits = np.asarray(scores, dtype=np.float64)
        probs = _softmax(logits, temperature=temperature)
        index = int(np.argmax(probs))
        diagnostics = {
            "policy_probs": probs,
            "epistemic_values": np.asarray(epistemic_values),
            "logits": logits,
        }
        return index, np.asarray(predictions[index]), diagnostics
