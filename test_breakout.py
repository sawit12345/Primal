import gymnasium as gym
import ale_py
from agent import Agent
import numpy as np

gym.register_envs(ale_py)

def test_breakout():
    print("Testing Gymnasium Breakout...")
    env = gym.make('ALE/Breakout-v5', obs_type='rgb')
    agent = Agent(action_space_size=env.action_space.n)

    # Test for 1 episode or until it fails (2 lives left means losing a life usually, but we run full episode)
    obs, info = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step = 0
    lives_lost = 0
    initial_lives = info.get('lives', 5)

    # Run until the episode ends
    while not (done or truncated):
        action = agent.step(obs, 0, done, info)
        next_obs, reward, done, truncated, info = env.step(action)

        current_lives = info.get('lives', initial_lives)
        if current_lives < initial_lives:
            lives_lost += (initial_lives - current_lives)
            initial_lives = current_lives

        obs = next_obs
        total_reward += reward
        step += 1

        if step % 1000 == 0:
            print(f"Step: {step}, Reward: {reward}, Total Reward: {total_reward}, Lives: {current_lives}")

    print(f"Breakout test completed. Total steps: {step}, Total reward: {total_reward}, Lives lost: {lives_lost}")
    env.close()

if __name__ == "__main__":
    test_breakout()
