"""Theory-theory ensemble: multiple competing latent hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.temporal_decay import MarkovTemporalDecay


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=-1e6, posinf=1e6, neginf=-1e6)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / (np.sum(exp) + 1e-12)


@dataclass
class _Hypothesis:
    weights: np.ndarray
    log_evidence: float = 0.0


@dataclass
class TheoryTheoryEnsemble:
    """Maintains and updates multiple context-sensitive hypotheses."""

    state_dim: int
    action_dim: int
    num_hypotheses: int = 5
    learning_rate: float = 0.08
    seed: int = 0
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        input_dim = self.state_dim + self.action_dim + 1
        self.hypotheses = [
            _Hypothesis(weights=rng.normal(0.0, 0.05, size=(self.state_dim, input_dim)))
            for _ in range(self.num_hypotheses)
        ]

    def _compose_input(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action, np.ones(1, dtype=np.float64)], axis=0)

    def predict(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        x = self._compose_input(state, action)

        predictions = np.asarray([hypothesis.weights @ x for hypothesis in self.hypotheses])
        posterior = self.posterior()
        mixture_prediction = np.sum(predictions * posterior[:, None], axis=0)
        return mixture_prediction, predictions

    def posterior(self) -> np.ndarray:
        log_evidence = np.asarray([hypothesis.log_evidence for hypothesis in self.hypotheses], dtype=np.float64)
        return _softmax(log_evidence)

    def ambiguity(self) -> float:
        posterior = self.posterior()
        return float(-np.sum(posterior * np.log(posterior + 1e-12)))

    def update(self, state: np.ndarray, action: np.ndarray, target_next_state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        target = np.asarray(target_next_state, dtype=np.float64).ravel()
        x = self._compose_input(state, action)

        predictions = np.asarray([hypothesis.weights @ x for hypothesis in self.hypotheses])
        errors = target[None, :] - predictions
        sq_error = np.sum(errors**2, axis=1)
        log_likelihoods = -0.5 * sq_error

        for idx, hypothesis in enumerate(self.hypotheses):
            hypothesis.log_evidence = self.decay.blend_scalar(
                hypothesis.log_evidence,
                float(log_likelihoods[idx]),
            )

        post = _softmax(np.asarray([hypothesis.log_evidence for hypothesis in self.hypotheses]))

        for idx, hypothesis in enumerate(self.hypotheses):
            gradient = np.outer(errors[idx], x)
            hypothesis.weights += self.learning_rate * post[idx] * gradient

        return post
