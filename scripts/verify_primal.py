"""Verification entrypoint for Primal physics and image benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from agent import benchmark_physics, evaluate_mnist


def _physics_envs() -> list[tuple[str, int]]:
    envs: list[tuple[str, int]] = [("CartPole-v1", 0), ("CartPole-v1", 1)]
    try:
        import gymnasium as gym

        probe = gym.make("ALE/Pong-v5")
        probe.close()
        envs.append(("ALE/Pong-v5", 2))
    except Exception:
        pass
    return envs


def main() -> None:
    report = {
        "physics": [benchmark_physics(env_name=env_name, episodes=6, seed=seed) for env_name, seed in _physics_envs()],
        "vision": evaluate_mnist(samples_per_digit=10, seed=0),
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
