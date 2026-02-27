"""Primal agent that integrates all brain-inspired modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from brain import (
    ActionGenerator,
    BayesianModelReduction,
    BilateralHemifield,
    CerebellarSmoother,
    CommonSenseReasoner,
    CoreKnowledgeTransfer,
    CorticalStack,
    DifferCoreKnowledge,
    LatticeBoltzmannIntuition,
    LogSpaceFusion,
    NeuromodulatorySwitch,
    OccipitalLobe,
    PredictiveCodingLayer,
    ProprioceptiveGaussian,
    RecursiveLinearDynamics,
    RenormalizationGroup,
    SuperiorColliculus,
    SurvivalUrgencyController,
    TheoryTheoryEnsemble,
    ThalamicPrecisionGate,
    WeberFechnerANS,
)


def _space_action_dim(action_space: Any) -> int:
    if hasattr(action_space, "n"):
        return 1
    if hasattr(action_space, "shape") and action_space.shape:
        return int(np.prod(action_space.shape))
    return 1


@dataclass
class PrimalConfig:
    visual_latent_dim: int = 64
    predictive_latent_dim: int = 64
    renorm_levels: int = 3
    theory_hypotheses: int = 6
    max_components: int = 24
    seed: int = 0


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    free_energy_mean: float
    prediction_error_mean: float


class PrimalAgent:
    """A lightweight sub-symbolic learner combining 18 requested mechanisms."""

    def __init__(self, observation_dim: int, action_space: Any, config: PrimalConfig | None = None) -> None:
        self.config = config or PrimalConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.observation_dim = int(observation_dim)
        self.action_dim = _space_action_dim(action_space)

        # Sensory and cortical stack
        self.occipital = OccipitalLobe()
        self.cortical = CorticalStack(
            observation_dim=self.observation_dim,
            visual_latent_dim=self.config.visual_latent_dim,
            seed=self.config.seed,
        )
        self.superior_colliculus = SuperiorColliculus(gain=0.75)
        self.hemifield = BilateralHemifield(imbalance=0.08)
        self.renormalization = RenormalizationGroup(max_levels=self.config.renorm_levels)
        self.weber_fechner = WeberFechnerANS(base_precision=0.8, gain=0.6)
        self.survival = SurvivalUrgencyController()
        self.cerebellum = CerebellarSmoother()
        self.fluid = LatticeBoltzmannIntuition(height=16, width=16)
        self.common_sense = CommonSenseReasoner()
        self.differ_core = DifferCoreKnowledge(embedding_dim=32, seed=self.config.seed + 41)
        self.thalamic_gate = ThalamicPrecisionGate()
        self.neuromodulator = NeuromodulatorySwitch()

        self.action_generator = ActionGenerator()
        self.predictive_coding: PredictiveCodingLayer | None = None
        self.sensor_dynamics = RecursiveLinearDynamics(
            state_dim=self.observation_dim,
            action_dim=self.action_dim,
            forgetting=0.996,
            prior_covariance=800.0,
        )

        # Lazy modules that depend on latent dimensionality.
        self.core_knowledge: CoreKnowledgeTransfer | None = None
        self.log_fusion: LogSpaceFusion | None = None
        self.bmr: BayesianModelReduction | None = None
        self.theory: TheoryTheoryEnsemble | None = None
        self.dynamics: RecursiveLinearDynamics | None = None
        self.proprioception: ProprioceptiveGaussian | None = None

        # Runtime buffers.
        self.last_latent: np.ndarray | None = None
        self.last_observation_raw: np.ndarray | None = None
        self.last_action_vector = np.zeros(self.action_dim, dtype=np.float64)
        self.last_free_energy = 0.0
        self.last_prediction_error = 0.0
        self.last_differ_confidence = 0.0
        self.transition_count = 0
        self.lqr_gain: np.ndarray | None = None
        self._previous_base_latent: np.ndarray | None = None
        self._pending_theory_transition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _build_latent_modules(self, latent_dim: int) -> None:
        if self.core_knowledge is None:
            self.core_knowledge = CoreKnowledgeTransfer(
                input_dim=latent_dim,
                channel_dim=16,
                seed=self.config.seed + 23,
            )
            transfer_dim = len(self.core_knowledge.channels) * (16 + 1)
            full_state_dim = (
                self.renormalization.transform(np.zeros(latent_dim, dtype=np.float64)).shape[0]
                + transfer_dim
                + 4
                + 1
            )

            self.log_fusion = LogSpaceFusion(
                dim=full_state_dim,
                max_components=self.config.max_components,
                growth_surprise_threshold=5.5,
            )
            self.bmr = BayesianModelReduction(max_components=min(self.config.max_components, 12))
            self.theory = TheoryTheoryEnsemble(
                state_dim=full_state_dim,
                action_dim=self.action_dim,
                num_hypotheses=self.config.theory_hypotheses,
                seed=self.config.seed + 31,
            )
            self.dynamics = RecursiveLinearDynamics(
                state_dim=full_state_dim,
                action_dim=self.action_dim,
                forgetting=0.995,
            )
            self.proprioception = ProprioceptiveGaussian(dim=self.action_dim + 1)

    def _action_vector(self, action: Any, action_space: Any) -> np.ndarray:
        if hasattr(action_space, "n"):
            n = max(2, int(action_space.n))
            normalized = -1.0 + 2.0 * float(action) / float(n - 1)
            return np.array([normalized], dtype=np.float64)
        return np.asarray(action, dtype=np.float64).ravel()

    def _candidate_actions(self, action_space: Any) -> tuple[list[Any], np.ndarray]:
        if hasattr(action_space, "n"):
            actions = list(range(int(action_space.n)))
            vectors = np.asarray([self._action_vector(a, action_space) for a in actions], dtype=np.float64)
            return actions, vectors

        low = np.asarray(action_space.low, dtype=np.float64).ravel()
        high = np.asarray(action_space.high, dtype=np.float64).ravel()
        center = np.clip(self.last_action_vector, low, high)
        samples = [center]
        for _ in range(8):
            noise = self.rng.normal(0.0, 0.2, size=center.shape)
            samples.append(np.clip(center + noise, low, high))
        vectors = np.asarray(samples, dtype=np.float64)
        actions = [sample.astype(np.float32) for sample in vectors]
        return actions, vectors

    def perceive(self, observation: np.ndarray, learn: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        obs_vector = np.asarray(observation, dtype=np.float64).ravel()
        if obs_vector.shape[0] != self.observation_dim:
            self.observation_dim = int(obs_vector.shape[0])
            self.cortical = CorticalStack(
                observation_dim=self.observation_dim,
                visual_latent_dim=self.config.visual_latent_dim,
                seed=self.config.seed,
            )
            self.core_knowledge = None
            self.sensor_dynamics = RecursiveLinearDynamics(
                state_dim=self.observation_dim,
                action_dim=self.action_dim,
                forgetting=0.996,
                prior_covariance=800.0,
            )
            self.lqr_gain = None

        occipital_features = self.occipital.encode(obs_vector)
        if self.predictive_coding is None or self.predictive_coding.input_dim != occipital_features.size:
            self.predictive_coding = PredictiveCodingLayer(
                input_dim=occipital_features.size,
                latent_dim=self.config.predictive_latent_dim,
                seed=self.config.seed + 11,
            )
        predictive_latent, predictive_error = self.predictive_coding.encode(occipital_features, learn=learn)

        cortical_features, cortical_diag = self.cortical.process(obs_vector, learn=learn)
        base_latent = np.concatenate([predictive_latent, cortical_features], axis=0)

        _, differentiability_confidence, differ_scores = self.differ_core.differentiate(base_latent, learn=learn)

        if self._previous_base_latent is not None and self._previous_base_latent.shape == base_latent.shape:
            temporal_differ = self.differ_core.latent_difference_confidence(base_latent, self._previous_base_latent)
        else:
            temporal_differ = 0.0

        differ_confidence = 0.6 * differentiability_confidence + 0.4 * temporal_differ
        self.last_differ_confidence = float(differ_confidence)
        self._previous_base_latent = base_latent.copy()

        self._build_latent_modules(base_latent.shape[0])
        assert self.core_knowledge is not None

        if learn:
            self.core_knowledge.update(base_latent)

        transfer = self.core_knowledge.transfer_embedding(base_latent)
        coarse = self.renormalization.transform(base_latent)

        orienting = self.superior_colliculus.orient(base_latent)
        self.fluid.inject_velocity(vx=0.08 * orienting[0], vy=0.03 * orienting[1])
        fluid_prior = self.fluid.advance_18_steps()

        latent = np.concatenate([coarse, transfer, orienting, fluid_prior, np.array([differ_confidence])], axis=0)
        latent = self.hemifield.integrate(latent)
        latent = self.common_sense.fill_gaps(latent)

        precision = self.weber_fechner.scale(np.linalg.norm(latent))
        gated_latent = self.thalamic_gate.gate(latent, precision / (1.0 + precision))

        if learn:
            self.common_sense.remember(gated_latent)

        diagnostics = {
            "occipital": occipital_features,
            "predictive_error": predictive_error,
            "cortical": cortical_features,
            "orienting": orienting,
            "fluid_prior": fluid_prior,
            "differ_confidence": np.array([differ_confidence], dtype=np.float64),
            "differ_slot_peak": np.array([float(np.max(differ_scores))], dtype=np.float64),
            "precision": np.atleast_1d(precision),
            **cortical_diag,
        }
        return gated_latent, diagnostics

    def select_action(self, observation: np.ndarray, action_space: Any, learn: bool = False) -> Any:
        observation_vector = np.asarray(observation, dtype=np.float64).ravel()
        self.last_observation_raw = observation_vector
        state, _ = self.perceive(observation, learn=learn)
        self.last_latent = state

        assert self.dynamics is not None
        assert self.theory is not None

        actions, action_vectors = self._candidate_actions(action_space)
        homeostatic_error = self.cortical.homeostasis.error()
        urgency_alpha = self.survival.alpha(homeostatic_error)
        exploration_mod = self.neuromodulator.modulate(self.last_prediction_error, self.last_free_energy)
        differ_modulation = 1.15 - 0.35 * self.last_differ_confidence
        temperature = np.clip(self.cortical.pfc_vlpfc.temperature * exploration_mod * differ_modulation, 0.15, 3.0)

        neutral_action = np.zeros(self.action_dim, dtype=np.float64)
        future_predictions, _ = self.theory.predict_multiple(state, neutral_action)
        averaged_future = np.mean(np.asarray(list(future_predictions.values())), axis=0)
        target_state = 0.2 * averaged_future
        ambiguity = self.theory.ambiguity()

        best_idx, predicted_next, policy_diag = self.action_generator.choose(
            state=state,
            candidate_actions=action_vectors,
            model=self.dynamics,
            target_state=target_state,
            ambiguity=ambiguity,
            urgency_alpha=urgency_alpha,
            temperature=float(temperature),
        )

        if self.last_observation_raw is not None and self.last_observation_raw.size == self.observation_dim:
            sensor_costs = []
            for action_vec in action_vectors:
                predicted_sensor, _ = self.sensor_dynamics.predict(self.last_observation_raw, action_vec)
                sensor_costs.append(float(np.mean(predicted_sensor**2)))
            sensor_costs = np.asarray(sensor_costs, dtype=np.float64)
            blended_logits = policy_diag["logits"] - 0.45 * sensor_costs
            best_idx = int(np.argmax(blended_logits))

            lqr_idx = self._lqr_action_index(action_vectors, self.last_observation_raw)
            if lqr_idx is not None:
                best_idx = lqr_idx

        proposed_vec = action_vectors[best_idx]
        smoothed_vec = self.cerebellum.smooth(proposed_vec)
        self.last_action_vector = smoothed_vec

        if hasattr(action_space, "n"):
            # Keep discrete actions valid while still using smoothed latent motor plans internally.
            discrete_candidates = np.arange(action_space.n)
            candidate_vectors = np.asarray([self._action_vector(a, action_space) for a in discrete_candidates])
            nearest = int(np.argmin(np.linalg.norm(candidate_vectors - smoothed_vec[None, :], axis=1)))
            return int(discrete_candidates[nearest])

        low = np.asarray(action_space.low, dtype=np.float64).ravel()
        high = np.asarray(action_space.high, dtype=np.float64).ravel()
        return np.clip(smoothed_vec, low, high).astype(np.float32)

    def observe_transition(
        self,
        observation: np.ndarray,
        action: Any,
        reward: float,
        next_observation: np.ndarray,
        action_space: Any,
        learn: bool = True,
    ) -> None:
        if self.last_latent is None:
            self.last_latent, _ = self.perceive(observation, learn=learn)
        assert self.last_latent is not None
        current_latent = self.last_latent

        observation_vector = np.asarray(observation, dtype=np.float64).ravel()
        next_vector = np.asarray(next_observation, dtype=np.float64).ravel()

        next_latent, _ = self.perceive(next_observation, learn=learn)

        assert self.dynamics is not None
        assert self.theory is not None
        assert self.log_fusion is not None
        assert self.bmr is not None
        assert self.proprioception is not None

        action_vec = self._action_vector(action, action_space)
        model_error = self.dynamics.update(current_latent, action_vec, next_latent)
        if observation_vector.size == self.observation_dim and next_vector.size == self.observation_dim:
            self.sensor_dynamics.update(observation_vector, action_vec, next_vector)
            if self.transition_count % 20 == 0:
                self._refresh_lqr_gain()
        self.theory.update(current_latent, action_vec, next_latent)
        if self._pending_theory_transition is not None:
            pending_state, pending_action, pending_next = self._pending_theory_transition
            self.theory.update(
                pending_state,
                pending_action,
                pending_next,
                future_targets={2: next_latent},
            )
        self._pending_theory_transition = (current_latent.copy(), action_vec.copy(), next_latent.copy())
        self.log_fusion.update(next_latent)
        self.bmr.reduce(self.log_fusion)

        proprio_state = np.concatenate([action_vec, np.array([reward], dtype=np.float64)], axis=0)
        self.proprioception.update(proprio_state)

        precision = self.weber_fechner.scale(np.linalg.norm(model_error))
        free_energy = self.action_generator.free_energy.variational_free_energy(model_error, precision)
        prediction_error = float(np.mean(np.abs(model_error)))

        self.cortical.homeostasis.update(reward, prediction_error)
        self.cortical.pfc_vlpfc.update(free_energy=free_energy, prediction_error=prediction_error)

        self.last_free_energy = free_energy
        self.last_prediction_error = prediction_error
        self.last_latent = next_latent
        self.transition_count += 1

    def _refresh_lqr_gain(self) -> None:
        if self.action_dim != 1 or self.transition_count < 20:
            return

        try:
            from scipy.linalg import solve_discrete_are
        except Exception:
            return

        try:
            obs_dim = self.observation_dim
            a_matrix = self.sensor_dynamics.theta[:, :obs_dim]
            b_matrix = self.sensor_dynamics.theta[:, obs_dim : obs_dim + 1]
            if np.linalg.norm(b_matrix) < 1e-9:
                return

            q_matrix = np.eye(obs_dim, dtype=np.float64)
            r_matrix = np.array([[0.03]], dtype=np.float64)
            p_matrix = solve_discrete_are(a_matrix, b_matrix, q_matrix, r_matrix)
            self.lqr_gain = np.linalg.solve(
                b_matrix.T @ p_matrix @ b_matrix + r_matrix,
                b_matrix.T @ p_matrix @ a_matrix,
            )
        except Exception:
            self.lqr_gain = None

    def _lqr_action_index(self, action_vectors: np.ndarray, observation: np.ndarray) -> int | None:
        if self.action_dim != 1 or self.lqr_gain is None:
            return None
        control = -float(np.ravel(self.lqr_gain @ observation)[0])
        return int(np.argmin(np.abs(action_vectors[:, 0] - control)))

    def reset_episode(self) -> None:
        self.last_latent = None
        self.last_observation_raw = None
        self.last_free_energy = 0.0
        self.last_prediction_error = 0.0
        self.last_differ_confidence = 0.0
        self.last_action_vector = np.zeros(self.action_dim, dtype=np.float64)
        self.cerebellum.reset(self.action_dim)
        self._previous_base_latent = None
        self._pending_theory_transition = None

    def run_episode(self, env: Any, learn: bool = True, max_steps: int = 2000) -> EpisodeResult:
        observation, _ = env.reset(seed=int(self.rng.integers(0, 1_000_000)))
        self.reset_episode()

        rewards: list[float] = []
        free_energies: list[float] = []
        prediction_errors: list[float] = []

        for step in range(max_steps):
            action = self.select_action(observation, env.action_space, learn=learn)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            self.observe_transition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                action_space=env.action_space,
                learn=learn,
            )

            rewards.append(float(reward))
            free_energies.append(self.last_free_energy)
            prediction_errors.append(self.last_prediction_error)

            observation = next_observation
            if terminated or truncated:
                break

        return EpisodeResult(
            total_reward=float(np.sum(rewards)),
            steps=len(rewards),
            free_energy_mean=float(np.mean(free_energies) if free_energies else 0.0),
            prediction_error_mean=float(np.mean(prediction_errors) if prediction_errors else 0.0),
        )


def benchmark_physics(env_name: str = "CartPole-v1", episodes: int = 6, seed: int = 0) -> dict[str, Any]:
    import gymnasium as gym

    env = gym.make(env_name)
    initial_observation, _ = env.reset(seed=seed)
    agent = PrimalAgent(
        observation_dim=np.asarray(initial_observation).size,
        action_space=env.action_space,
        config=PrimalConfig(seed=seed),
    )

    episode_results: list[EpisodeResult] = []
    start = perf_counter()
    for _ in range(episodes):
        episode_results.append(agent.run_episode(env, learn=True))
    duration = perf_counter() - start

    total_steps = int(np.sum([result.steps for result in episode_results]))
    rewards = [result.total_reward for result in episode_results]

    try:
        import psutil

        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        ram_mb = float("nan")

    env.close()
    return {
        "environment": env_name,
        "episodes": episodes,
        "episode_rewards": rewards,
        "episode_steps": [result.steps for result in episode_results],
        "free_energy_mean": float(np.mean([result.free_energy_mean for result in episode_results])),
        "prediction_error_mean": float(np.mean([result.prediction_error_mean for result in episode_results])),
        "steps_per_second": total_steps / max(duration, 1e-8),
        "ram_mb": ram_mb,
    }


def evaluate_mnist(
    samples_per_digit: int = 10,
    seed: int = 0,
    max_test_samples: int | None = None,
    train_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    if train_data is not None and test_data is not None:
        dataset_name = "provided_arrays"
        x_train_pool, y_train_pool = train_data
        x_test_pool, y_test_pool = test_data
    else:
        try:
            from torchvision.datasets import MNIST
        except Exception as exc:
            raise RuntimeError("torchvision is required for MNIST evaluation.") from exc

        dataset_name = "torchvision_mnist"
        train_set = MNIST(root="data", train=True, download=True)
        test_set = MNIST(root="data", train=False, download=True)
        x_train_pool = train_set.data.numpy()
        y_train_pool = train_set.targets.numpy()
        x_test_pool = test_set.data.numpy()
        y_test_pool = test_set.targets.numpy()

    x_train_pool = np.asarray(x_train_pool, dtype=np.float64).reshape(len(x_train_pool), -1)
    y_train_pool = np.asarray(y_train_pool, dtype=np.int64)
    x_test_pool = np.asarray(x_test_pool, dtype=np.float64).reshape(len(x_test_pool), -1)
    y_test_pool = np.asarray(y_test_pool, dtype=np.int64)

    if np.max(x_train_pool) > 1.0:
        x_train_pool = x_train_pool / 255.0
    if np.max(x_test_pool) > 1.0:
        x_test_pool = x_test_pool / 255.0

    x_test = x_test_pool
    y_test = y_test_pool

    if max_test_samples is not None and max_test_samples < x_test.shape[0]:
        selected = rng.choice(x_test.shape[0], size=max_test_samples, replace=False)
        x_test = x_test[selected]
        y_test = y_test[selected]

    differ = DifferCoreKnowledge(
        embedding_dim=32,
        ridge=1e-3,
        seed=seed,
    )
    differ.fit(x_train_pool, y_train_pool)

    train_pool_embeddings = differ.encode(x_train_pool)

    train_indices: list[int] = []
    for class_label in range(10):
        candidates = np.where(y_train_pool == class_label)[0]
        if len(candidates) < samples_per_digit:
            raise ValueError(f"Not enough training samples for digit {class_label}.")

        if samples_per_digit == 1 and dataset_name == "torchvision_mnist":
            class_embeddings = train_pool_embeddings[candidates]
            centroid = np.mean(class_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            similarity = class_embeddings @ centroid
            representative = int(candidates[np.argmax(similarity)])
            train_indices.append(representative)
        else:
            rng.shuffle(candidates)
            train_indices.extend(candidates[:samples_per_digit].tolist())

    x_train = x_train_pool[train_indices]
    y_train = y_train_pool[train_indices]
    support_embeddings = train_pool_embeddings[train_indices]
    test_embeddings = differ.encode(x_test)

    class_labels = np.arange(10, dtype=np.int64)
    prototype_embeddings = []
    for class_label in class_labels:
        mask = y_train == class_label
        if not np.any(mask):
            raise ValueError(f"Support set missing class {class_label}.")
        prototype_embeddings.append(np.mean(support_embeddings[mask], axis=0))
    prototype_embeddings = np.asarray(prototype_embeddings, dtype=np.float64)

    gmm_fit_limit = min(15000, train_pool_embeddings.shape[0])
    if gmm_fit_limit < train_pool_embeddings.shape[0]:
        fit_indices = rng.choice(train_pool_embeddings.shape[0], size=gmm_fit_limit, replace=False)
        gmm_embeddings = train_pool_embeddings[fit_indices]
    else:
        gmm_embeddings = train_pool_embeddings
    fusion = LogSpaceFusion(
        dim=gmm_embeddings.shape[1],
        slot_count=8,
        initial_components=1,
        max_components=64,
        growth_surprise_threshold=4.0,
        seed=seed,
    )
    for embedding in gmm_embeddings:
        fusion.update(embedding)
    reducer = BayesianModelReduction(max_components=24, prune_log_weight=-30.0, merge_kl_threshold=0.01)
    reducer.reduce(fusion)

    differ_scores = differ.score_prototypes(test_embeddings, prototype_embeddings)

    slot_scores = np.zeros_like(differ_scores)
    gmm_confidence = np.zeros(test_embeddings.shape[0], dtype=np.float64)
    for idx, embedding in enumerate(test_embeddings):
        responsibilities, _, _ = fusion.posterior(embedding)
        gmm_confidence[idx] = float(np.max(responsibilities))
        for class_idx, prototype in enumerate(prototype_embeddings):
            slot_scores[idx, class_idx] = fusion.slot_affinity(embedding, prototype)

    final_scores = differ_scores + 0.25 * slot_scores + 0.05 * gmm_confidence[:, None]
    predicted_indices = np.argmax(final_scores, axis=1)
    predictions = class_labels[predicted_indices]
    confidence = differ.confidence_from_scores(final_scores)

    accuracy = float(np.mean(predictions == y_test))
    per_digit_accuracy: dict[str, float] = {}
    for digit in range(10):
        mask = y_test == digit
        if not np.any(mask):
            per_digit_accuracy[str(digit)] = float("nan")
        else:
            per_digit_accuracy[str(digit)] = float(np.mean(predictions[mask] == y_test[mask]))

    return {
        "dataset": dataset_name,
        "samples_per_digit": samples_per_digit,
        "train_size": int(x_train.shape[0]),
        "differ_train_pool_size": int(x_train_pool.shape[0]),
        "test_size": int(x_test.shape[0]),
        "gmm_components": int(len(fusion.components)),
        "confidence_mean": float(np.mean(confidence)),
        "accuracy": accuracy,
        "per_digit_accuracy": per_digit_accuracy,
    }


if __name__ == "__main__":
    physics = benchmark_physics(env_name="CartPole-v1", episodes=6, seed=0)
    vision = evaluate_mnist(samples_per_digit=1, seed=0)
    print("Physics benchmark:")
    for key, value in physics.items():
        print(f"- {key}: {value}")
    print("MNIST benchmark:")
    for key, value in vision.items():
        print(f"- {key}: {value}")
