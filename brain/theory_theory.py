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
    weights_by_horizon: np.ndarray
    log_evidence: float = 0.0
    prediction_dispersion: float = 0.0


@dataclass
class TheoryTheoryEnsemble:
    """Maintains and updates multiple context-sensitive hypotheses."""

    state_dim: int
    action_dim: int
    num_hypotheses: int = 5
    learning_rate: float = 0.08
    horizons: tuple[int, ...] = (1, 2, 3)
    seed: int = 0
    decay: MarkovTemporalDecay = field(default_factory=MarkovTemporalDecay)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        input_dim = self.state_dim + self.action_dim + 1
        num_horizons = len(self.horizons)
        self.hypotheses = [
            _Hypothesis(weights_by_horizon=rng.normal(0.0, 0.05, size=(num_horizons, self.state_dim, input_dim)))
            for _ in range(self.num_hypotheses)
        ]
        self._horizon_to_index = {horizon: idx for idx, horizon in enumerate(self.horizons)}

    def _compose_input(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action, np.ones(1, dtype=np.float64)], axis=0)

    def _predict_hypothesis(self, hypothesis: _Hypothesis, x: np.ndarray) -> np.ndarray:
        return np.asarray([weights @ x for weights in hypothesis.weights_by_horizon], dtype=np.float64)

    def predict(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        x = self._compose_input(state, action)

        all_predictions = np.asarray([self._predict_hypothesis(hypothesis, x) for hypothesis in self.hypotheses])
        one_step_idx = self._horizon_to_index[1]
        predictions = all_predictions[:, one_step_idx, :]
        posterior = self.posterior()
        mixture_prediction = np.sum(predictions * posterior[:, None], axis=0)
        return mixture_prediction, predictions

    def predict_multiple(self, state: np.ndarray, action: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        x = self._compose_input(state, action)

        all_predictions = np.asarray([self._predict_hypothesis(hypothesis, x) for hypothesis in self.hypotheses])
        posterior = self.posterior()

        mixture: dict[int, np.ndarray] = {}
        per_hypothesis: dict[int, np.ndarray] = {}
        for horizon, horizon_idx in self._horizon_to_index.items():
            predictions = all_predictions[:, horizon_idx, :]
            mixture[horizon] = np.sum(predictions * posterior[:, None], axis=0)
            per_hypothesis[horizon] = predictions

        return mixture, per_hypothesis

    def posterior(self) -> np.ndarray:
        log_evidence = np.asarray([hypothesis.log_evidence for hypothesis in self.hypotheses], dtype=np.float64)
        return _softmax(log_evidence)

    def ambiguity(self) -> float:
        posterior = self.posterior()
        entropy = float(-np.sum(posterior * np.log(posterior + 1e-12)))
        dispersion = float(np.mean([hypothesis.prediction_dispersion for hypothesis in self.hypotheses]))
        return entropy + 0.25 * dispersion

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        target_next_state: np.ndarray,
        future_targets: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        targets: dict[int, np.ndarray] = {1: np.asarray(target_next_state, dtype=np.float64).ravel()}
        if future_targets is not None:
            for horizon, target in future_targets.items():
                if horizon in self._horizon_to_index:
                    targets[horizon] = np.asarray(target, dtype=np.float64).ravel()

        x = self._compose_input(state, action)

        all_predictions = np.asarray([self._predict_hypothesis(hypothesis, x) for hypothesis in self.hypotheses])
        one_step_idx = self._horizon_to_index[1]
        one_step_predictions = all_predictions[:, one_step_idx, :]
        one_step_errors = targets[1][None, :] - one_step_predictions
        sq_error = np.sum(one_step_errors**2, axis=1)
        log_likelihoods = -0.5 * sq_error

        for idx, hypothesis in enumerate(self.hypotheses):
            hypothesis.log_evidence = self.decay.blend_scalar(
                hypothesis.log_evidence,
                float(log_likelihoods[idx]),
            )

        stacked_one_step = one_step_predictions
        mean_prediction = np.mean(stacked_one_step, axis=0)
        per_hypothesis_dispersion = np.mean((stacked_one_step - mean_prediction[None, :]) ** 2, axis=1)
        for idx, hypothesis in enumerate(self.hypotheses):
            hypothesis.prediction_dispersion = self.decay.blend_scalar(
                hypothesis.prediction_dispersion,
                float(per_hypothesis_dispersion[idx]),
            )

        post = _softmax(np.asarray([hypothesis.log_evidence for hypothesis in self.hypotheses]))

        for idx, hypothesis in enumerate(self.hypotheses):
            for horizon, target in targets.items():
                horizon_idx = self._horizon_to_index[horizon]
                prediction = all_predictions[idx, horizon_idx]
                error = target - prediction
                gradient = np.outer(error, x)
                horizon_scale = 1.0 / float(horizon)
                hypothesis.weights_by_horizon[horizon_idx] += self.learning_rate * post[idx] * horizon_scale * gradient

        return post
