"""
Quick manual test of MazeEnv in isolation.

Runs N episodes with random actions and prints results.
Set RENDER=True to open the Pygame window.
"""

import random
from maze_env import MazeEnv

RENDER    = True   # set False for headless
N_EPISODES = 3
SEED       = 42

with MazeEnv(seed=SEED, render_mode="human" if RENDER else None) as env:
    print(f"obs_size={env.obs_size}  n_actions={env.n_actions}\n")

    for ep in range(1, N_EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = random.randint(0, env.n_actions - 1)
            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        result = "SUCCESS" if total_reward > 0 else "FAIL"
        print(f"Ep {ep}: {steps:3d} steps | return={total_reward:+.3f} | {result}")
