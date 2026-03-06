#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import ale_py

from cheap_universal_agi.breakout_agent import BreakoutActiveInferenceAgent
from cheap_universal_agi.config import BlueprintConfig


def main():
    parser = argparse.ArgumentParser(description="Run Breakout one-shot-style evaluation")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=Path("results/breakout_eval_results.json"))
    args = parser.parse_args()

    # Gymnasium >=1.0 + ale-py >=0.9 registration path.
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", obs_type="rgb", repeat_action_probability=0.0)
    cfg = BlueprintConfig()
    agent = BreakoutActiveInferenceAgent(cfg)

    if hasattr(env.unwrapped, "get_action_meanings"):
        meanings = env.unwrapped.get_action_meanings()
        agent.bind_action_meanings(meanings)

    episodes = []
    for ep in range(args.episodes):
        out = agent.run_episode(env, max_steps=args.max_steps)
        episodes.append(
            {
                "episode": ep + 1,
                "score": out["score"],
                "steps": out["steps"],
                "start_lives": out["start_lives"],
                "min_lives": out["min_lives"],
                "end_lives": out["end_lives"],
            }
        )
        print(
            f"episode={ep+1} score={out['score']} steps={out['steps']} "
            f"start_lives={out['start_lives']} min_lives={out['min_lives']} end_lives={out['end_lives']}"
        )
    env.close()

    payload = {
        "episodes": episodes,
        "config": {"episodes": args.episodes, "max_steps": args.max_steps},
        "criterion_episode1_lives_left_ge_2": bool(episodes[0]["end_lives"] >= 2) if episodes else False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
