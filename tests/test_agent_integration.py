from __future__ import annotations

import numpy as np
import pytest

from agent import PrimalAgent, evaluate_mnist


def test_primal_agent_runs_cartpole_episode() -> None:
    gym = pytest.importorskip("gymnasium")

    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=0)
    agent = PrimalAgent(observation_dim=np.asarray(observation).size, action_space=env.action_space)
    result = agent.run_episode(env, learn=True, max_steps=100)
    env.close()

    assert result.steps > 0
    assert np.isfinite(result.total_reward)
    assert np.isfinite(result.free_energy_mean)


def test_mnist_evaluation_returns_metrics() -> None:
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(0)
    x_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []

    for digit in range(10):
        prototype = np.zeros(28 * 28, dtype=np.float64)
        prototype[digit] = 1.0
        train_samples = prototype[None, :] + 0.01 * rng.normal(size=(2, 28 * 28))
        test_samples = prototype[None, :] + 0.01 * rng.normal(size=(3, 28 * 28))
        x_train_parts.append(train_samples)
        y_train_parts.append(np.full(2, digit, dtype=np.int64))
        x_test_parts.append(test_samples)
        y_test_parts.append(np.full(3, digit, dtype=np.int64))

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    report = evaluate_mnist(
        samples_per_digit=1,
        seed=0,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test),
    )

    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["dataset"] == "provided_arrays"
    assert report["train_size"] == 10
    assert report["test_size"] == 30
