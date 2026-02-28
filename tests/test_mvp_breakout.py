"""MVP Breakout test for Phase -1."""

import time
import psutil
import os
import numpy as np

# Import gymnasium with proper registration
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from primal.agent import PrimalAgent


def compute_entropy(probs):
    """Compute entropy of probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def run_episode(agent, env, episode_num):
    """Run a single episode and return statistics."""
    obs, info = env.reset()
    
    ep_score = 0
    ep_steps = 0
    aligned_steps = 0
    fe_values = []
    action_counts = np.zeros(agent.n_actions)
    
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / (1024**3)  # GB
    start_time = time.perf_counter()
    
    done = False
    agent.reset()
    
    while not done:
        # Get ball and paddle positions before action
        ball_x, paddle_x = agent.get_ball_and_paddle_x()
        
        action = agent.act(obs)
        action_counts[action] += 1
        
        # Check alignment (paddle moving toward ball)
        if ball_x is not None and paddle_x is not None:
            diff = ball_x - paddle_x
            if abs(diff) < 0.1:  # Already close
                aligned_steps += 1
            elif diff > 0 and action == 2:  # Ball right, move right
                aligned_steps += 1
            elif diff < 0 and action == 3:  # Ball left, move left
                aligned_steps += 1
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        fe = agent.update(obs, action, reward, next_obs, terminated or truncated)
        
        fe_values.append(fe)
        ep_score += reward
        ep_steps += 1
        
        obs = next_obs
        done = terminated or truncated
        
        # Print progress every 500 steps
        if ep_steps % 500 == 0:
            elapsed = time.perf_counter() - start_time
            its = ep_steps / elapsed
            print(f"  Step {ep_steps}, score={ep_score}, FE={fe:.2f}, {its:.1f} it/s")
    
    elapsed = time.perf_counter() - start_time
    its = ep_steps / elapsed
    end_ram = process.memory_info().rss / (1024**3)
    
    # Compute action distribution entropy
    action_probs = action_counts / action_counts.sum()
    entropy = compute_entropy(action_probs)
    
    paddle_alignment = aligned_steps / ep_steps if ep_steps > 0 else 0
    mean_fe = np.mean(fe_values) if fe_values else 0
    
    return {
        'score': ep_score,
        'steps': ep_steps,
        'paddle_alignment': paddle_alignment,
        'entropy': entropy,
        'its': its,
        'ram_gb': end_ram,
        'mean_fe': mean_fe
    }


def main():
    print("=" * 60)
    print("MVP BREAKOUT TEST - Phase -1")
    print("=" * 60)
    
    # Create environment
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    
    # Create agent
    agent = PrimalAgent(
        obs_shape=(210, 160, 3),
        n_actions=4,
        mode='atari'
    )
    
    print(f"Agent initialized: {agent.n_actions} actions")
    print(f"Feature dim: {agent.feature_dim}")
    print()
    
    # Run 2 episodes
    results = []
    for ep in range(1, 3):
        print(f"\n--- Episode {ep} ---")
        stats = run_episode(agent, env, ep)
        results.append(stats)
        
        print(f"Score: {stats['score']}")
        print(f"Steps: {stats['steps']}")
        print(f"Paddle alignment: {stats['paddle_alignment']:.3f}")
        print(f"Entropy: {stats['entropy']:.3f}")
        print(f"Speed: {stats['its']:.1f} it/s")
        print(f"RAM: {stats['ram_gb']:.2f} GB")
        print(f"Mean FE: {stats['mean_fe']:.2f}")
    
    env.close()
    
    # Evaluate results
    print("\n" + "=" * 60)
    print("MVP RESULTS")
    print("=" * 60)
    
    ep1 = results[0]
    ep2 = results[1]
    
    # Check criteria
    checks = []
    
    # Ep2 score >= 5
    score_pass = ep2['score'] >= 5
    checks.append(("ep2 score >= 5", score_pass, ep2['score'], 5))
    
    # Entropy > 0.10
    entropy_pass = ep2['entropy'] > 0.10
    checks.append(("ep2 entropy > 0.10", entropy_pass, ep2['entropy'], 0.10))
    
    # it/s >= 30
    speed_pass = ep2['its'] >= 30
    checks.append(("it/s >= 30", speed_pass, ep2['its'], 30))
    
    # Paddle alignment > 0.45
    align_pass = ep2['paddle_alignment'] > 0.45
    checks.append(("paddle_alignment > 0.45", align_pass, ep2['paddle_alignment'], 0.45))
    
    # Print results
    for name, passed, actual, threshold in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} (got {actual:.2f}, need {threshold})")
    
    all_passed = all(c[1] for c in checks)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL MVP CHECKS PASSED!")
    else:
        print("SOME MVP CHECKS FAILED - Debug required")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    main()
