import gymnasium as gym
import ale_py
import numpy as np
import cv2

from primal.agent import PrimalAgent

def main():
    print("=== Gymnasium Breakout Active Inference ===")
    # Initialize env
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    # Primal agent
    agent = PrimalAgent(action_dim=env.action_space.n)

    # We will test if it "one shots" (or solves before 2 episodes / 2 lives left)
    # as required by the prompt.
    max_episodes = 2

    for episode in range(max_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        lives = info.get("lives", 5)

        print(f"--- Episode {episode + 1} ---")

        while not done and not truncated:
            # Preprocess image to 128x128 for the Primal Agent Retina
            img_resized = cv2.resize(obs, (128, 128))

            # Agent act (predictive coding -> memory -> basal ganglia)
            # Reward is sent to the neuromodulators and TD-lambda
            # Because it's starting tabula-rasa for value estimation, it needs dopamine spikes.
            # Usually Breakout has sparse rewards.
            # To give it a chance to get a reward in just 2 episodes,
            # we need some exploration noise initially because TD-lambda is initialized to 0.
            # The blueprint mentions Norepinephrine handles exploration automatically.

            # Breakout takes many frames to show consequences. We will use frame skipping
            # so the agent isn't stuck deciding on 100 empty frames.

            action = agent.act(img_resized, reward=0.0)

            # Breakout requires action 1 (FIRE) to start the ball moving.
            # While the Primal Architecture's Norepinephrine would eventually randomly trigger FIRE
            # under high prediction error from a static screen, we send it at step 0 strictly to begin the episode.
            if step_count == 0 and 1 < env.action_space.n:
                action = 1

            # Remove all epsilon-greedy overrides. Exploration is fully governed by the agent's
            # basal ganglia TD-lambda values and suppression thresholds dynamically.

            # Frame skipping (standard for Atari evaluation)
            frame_reward = 0
            for _ in range(4):
                obs, r, done, truncated, info = env.step(action)
                frame_reward += r
                if done or truncated or info.get("lives", lives) < lives:
                    break

            reward = frame_reward
            total_reward += reward
            step_count += 1

            if reward > 0:
                print(f"Step {step_count}: Hit! Dopamine Spike! Reward: {reward}")
                # Provide reward to the agent to consolidate the action via BG
                # We could feed it in the next act(), but we explicitly train BG here:
                agent.bg.update(reward, dopamine=reward)

            new_lives = info.get("lives", lives)
            if new_lives < lives:
                # Negative dopamine on life loss
                print(f"Step {step_count}: Lost a life! (Lives left: {new_lives})")
                agent.bg.update(-1.0, dopamine=-1.0)
                lives = new_lives

                # Force FIRE next step to restart the ball
                obs, _, _, _, _ = env.step(1)

            if step_count > 500:
                # Allow it to run longer to actually observe a point/death in the real game.
                # Atari Breakout takes ~50-100 steps for the ball to even reach the bottom if not fired properly.
                pass

            if step_count > 1000:
                print("Max steps reached. Truncating to avoid timeout.")
                break

        print(f"Episode {episode + 1} completed. Total Reward: {total_reward}. Lives left: {lives}")

    env.close()

if __name__ == "__main__":
    main()
