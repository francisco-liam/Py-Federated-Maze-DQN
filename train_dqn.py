"""
Single-client DQN training loop for the Unity maze environment.

Usage
-----
1. Open the Unity project in the Editor and press Play.
   (The Editor must be in Play mode before this script connects.)

2. In a separate terminal, from the project root:
       cd train
       python train_dqn.py

3. Checkpoints are saved to train/checkpoints/ every SAVE_EVERY episodes.
   To resume training from a checkpoint, set RESUME_FROM below.

Design notes
------------
- One env.reset() is called at the start. After that, ML-Agents handles
  all per-episode resets internally when the agent calls EndEpisode().
- The training loop is step-based, not episode-based: we count env steps
  for epsilon decay and training frequency regardless of episode boundaries.
- Evaluation runs the greedy policy (epsilon=0) for EVAL_EPISODES episodes
  and reports mean return and success rate.
- "Success" is defined as positive total return for an episode, which
  corresponds to reaching the exit (reward +1.0) minus accumulated step
  penalties. An episode where the agent reaches the exit quickly will
  have positive return.
"""

import os
import sys
from collections import deque
from typing import Optional

import numpy as np

from dqn_agent import DQNAgent
from maze_env import MazeEnv

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Network
OBS_SIZE    = 26   # 5x5 local grid (25) + step fraction (1)
N_ACTIONS   = 5    # NoOp, Up, Left, Down, Right
HIDDEN_SIZE = 128

# Optimization
LR          = 1e-3
GAMMA       = 0.99
BATCH_SIZE  = 64

# Replay buffer
BUFFER_CAPACITY = 50_000
WARMUP_STEPS    = 1_000   # don't train until this many transitions are stored
TRAIN_FREQ      = 4       # gradient step every N env steps

# Target network
TARGET_UPDATE_FREQ = 1_000  # hard sync every N gradient steps

# Exploration
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_STEPS = 10_000   # linear decay over first 10k env steps

# Training duration
MAX_EPISODES    = 2_000
EVAL_EVERY      = 50    # run evaluation every N episodes
EVAL_EPISODES   = 10    # episodes per evaluation run
SAVE_EVERY      = 200   # save checkpoint every N episodes
LOG_EVERY       = 10    # print training stats every N episodes

# Device
DEVICE = "cpu"   # "cuda" if GPU is available and desired

# Resume from checkpoint (set to a path string to load, e.g. "checkpoints/dqn_ep200.pt")
RESUME_FROM: Optional[str] = None

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DQNAgent, env: MazeEnv, n_episodes: int) -> dict:
    """
    Run the greedy policy for n_episodes episodes.

    The environment is NOT reset between evaluation and training —
    ML-Agents handles episode resets automatically.
    """
    returns   = []
    successes = 0

    for _ in range(n_episodes):
        # After a done=True step, env.step() automatically starts a new episode.
        # We need the initial obs for this new episode, which was returned as
        # next_obs in the previous done=True step. We track it via `obs`.
        ep_return = 0.0
        done      = False

        while not done:
            action = agent.select_greedy_action(obs)   # `obs` from outer scope
            obs, reward, done = env.step(action)
            ep_return += reward

        returns.append(ep_return)
        if ep_return > 0.0:
            successes += 1

    return {
        "mean_return":  float(np.mean(returns)),
        "std_return":   float(np.std(returns)),
        "success_rate": successes / n_episodes,
    }

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    os.makedirs("checkpoints", exist_ok=True)

    agent = DQNAgent(
        obs_size           = OBS_SIZE,
        n_actions          = N_ACTIONS,
        hidden_size        = HIDDEN_SIZE,
        lr                 = LR,
        gamma              = GAMMA,
        batch_size         = BATCH_SIZE,
        buffer_capacity    = BUFFER_CAPACITY,
        target_update_freq = TARGET_UPDATE_FREQ,
        eps_start          = EPS_START,
        eps_end            = EPS_END,
        eps_decay_steps    = EPS_DECAY_STEPS,
        warmup_steps       = WARMUP_STEPS,
        train_freq         = TRAIN_FREQ,
        device             = DEVICE,
    )

    if RESUME_FROM:
        agent.load(RESUME_FROM)
        print(f"Resumed from checkpoint: {RESUME_FROM}")
        print(f"  steps_done={agent.steps_done}  episodes_done={agent.episodes_done}")

    with MazeEnv(seed=0) as env:
        print(f"obs_size={env.obs_size}  n_actions={env.n_actions}")

        # Single reset at the very start. ML-Agents handles all subsequent
        # per-episode resets when the Unity agent calls EndEpisode().
        obs = env.reset()

        recent_returns: deque = deque(maxlen=100)

        episode    = agent.episodes_done
        ep_return  = 0.0
        ep_steps   = 0
        last_loss  = None

        # ----------------------------------------------------------------
        # Main step loop (not episode loop — episodes are implicit)
        # ----------------------------------------------------------------
        while episode < MAX_EPISODES:
            action           = agent.select_action(obs)
            next_obs, reward, done = env.step(action)

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.maybe_train()
            if loss is not None:
                last_loss = loss

            ep_return += reward
            ep_steps  += 1
            obs = next_obs   # next_obs is new episode start obs when done=True

            if done:
                episode += 1
                agent.episodes_done = episode
                recent_returns.append(ep_return)

                # ---- Per-episode logging ----
                if episode % LOG_EVERY == 0:
                    avg = np.mean(recent_returns) if recent_returns else float("nan")
                    loss_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
                    print(
                        f"Ep {episode:5d} | "
                        f"return {ep_return:+.3f} | "
                        f"avg100 {avg:+.3f} | "
                        f"eps {agent.epsilon():.3f} | "
                        f"steps {agent.steps_done:,d} | "
                        f"buf {len(agent.buffer):,d} | "
                        f"loss {loss_str}"
                    )

                # ---- Evaluation ----
                if episode % EVAL_EVERY == 0:
                    # `obs` is now the start of the new episode (from the done step).
                    # Pass it into evaluate() via the outer-scope reference.
                    stats = evaluate(agent, env, EVAL_EPISODES)
                    print(
                        f"  [EVAL ep={episode}] "
                        f"mean_return={stats['mean_return']:+.3f} "
                        f"±{stats['std_return']:.3f}  "
                        f"success_rate={stats['success_rate']:.0%}"
                    )

                # ---- Checkpoint ----
                if episode % SAVE_EVERY == 0:
                    ckpt_path = os.path.join("checkpoints", f"dqn_ep{episode}.pt")
                    agent.save(ckpt_path)
                    print(f"  Saved: {ckpt_path}")

                ep_return = 0.0
                ep_steps  = 0

        print(f"\nTraining complete. Total env steps: {agent.steps_done:,d}")
        final_path = os.path.join("checkpoints", "dqn_final.pt")
        agent.save(final_path)
        print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    train()
