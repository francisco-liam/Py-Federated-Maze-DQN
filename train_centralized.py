"""
Centralized DQN upper bound — one agent, all client maze seeds pooled.

Usage
-----
    python train_centralized.py

Design
------
A single DQN agent trains on seeds {0, 42, 99} in round-robin order
(one episode per seed per cycle).  This is the upper bound for the FL
comparison: the agent sees all environment distributions but without
the communication / aggregation constraints of federated learning.

Compute budget is matched to the FL run:
    FL:          50 rounds × 30 eps/round × 3 clients = 4 500 client-episodes
    Centralized: 1 500 cycles × 3 seeds              = 4 500 agent-episodes

Evaluation uses the held-out seed=7 (not seen during training) to
measure generalisation, matching the FL evaluation protocol.

Output
------
    checkpoints/centralized_final.pt
    checkpoints/summary_centralized.txt
"""

import os
from collections import deque

import numpy as np
import torch

from agent import DQNAgent
from maze_env import MazeEnv

# ---------------------------------------------------------------------------
# Config — match train_federated.py for a fair comparison
# ---------------------------------------------------------------------------

CLIENT_SEEDS  = [0, 42, 99]
HELD_OUT_SEED = 7

MAZE_W, MAZE_H = 21, 21
OBSTACLE_COUNT = 0
SHAPING_SCALE  = 0.05
N_STACK        = 4
CELL_OBS_PX    = 8
MAX_EP_STEPS   = 200
N_ACTIONS      = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Match total episode budget to FL run:
# FL: 50 rounds × 30 eps × 3 clients = 4500 episodes total
# Centralized: 1500 cycles × 3 seeds = 4500 episodes
N_CYCLES   = 1_500   # each cycle = one episode on each seed
EVAL_EVERY = 150     # evaluate every N cycles (matches FL's EVAL_EVERY=5 rounds × 30 eps)
EVAL_EPISODES = 20
EVAL_EPS      = 0.05
SAVE_EVERY    = 300
LOG_EVERY     = 50   # print rolling stats every N cycles

LR             = 1e-4
GAMMA          = 0.99
BATCH_SIZE     = 32
# Larger buffer for centralized — agent sees 3× as many distinct transitions
BUFFER_CAP     = 50_000
WARMUP_STEPS   = 2_000
TRAIN_FREQ     = 4
TARGET_UPDATE  = 500
EPS_START      = 1.0
EPS_END        = 0.1
# Match FL decay: 30 eps × 200 steps × 20 rounds = 120k steps
EPS_DECAY      = 120_000


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DQNAgent, env: MazeEnv, n_episodes: int) -> dict:
    returns, ep_lengths, successes = [], [], 0
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            ep_r, ep_len, done = 0.0, 0, False
            while not done:
                if np.random.rand() < EVAL_EPS:
                    action = np.random.randint(agent.n_actions)
                else:
                    action = agent.select_greedy_action(obs)
                obs, r, done = env.step(action)
                ep_r += r
                ep_len += 1
            returns.append(ep_r)
            ep_lengths.append(ep_len)
            if ep_r > 0:
                successes += 1
    return dict(
        mean_return   = float(np.mean(returns)),
        std_return    = float(np.std(returns)),
        success_rate  = successes / n_episodes,
        mean_ep_len   = float(np.mean(ep_lengths)),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    os.makedirs("checkpoints", exist_ok=True)

    env_kwargs = dict(
        width=MAZE_W, height=MAZE_H,
        obstacle_count=OBSTACLE_COUNT, shaping_scale=SHAPING_SCALE,
        n_stack=N_STACK, cell_obs_px=CELL_OBS_PX,
        max_episode_steps=MAX_EP_STEPS,
    )

    envs = [MazeEnv(seed=s, **env_kwargs) for s in CLIENT_SEEDS]
    obs_shape = envs[0].obs_size

    agent = DQNAgent(
        obs_shape=obs_shape, n_actions=N_ACTIONS, device=DEVICE,
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAP, target_update_freq=TARGET_UPDATE,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay_steps=EPS_DECAY,
        warmup_steps=WARMUP_STEPS, train_freq=TRAIN_FREQ,
    )

    eval_env = MazeEnv(seed=HELD_OUT_SEED, **env_kwargs)
    eval_history = []   # (cycle, mean_return, success_rate, mean_ep_len)

    print("=" * 68)
    print("  Centralized DQN — upper bound")
    print(f"  Seeds: {CLIENT_SEEDS}   Held-out: seed {HELD_OUT_SEED}")
    print(f"  Cycles: {N_CYCLES}   (= {N_CYCLES * len(CLIENT_SEEDS)} total episodes)")
    print(f"  Device: {DEVICE}")
    print("=" * 68)
    print(f"  obs_shape={obs_shape}\n")

    hdr = f"{'Cycle':>6}  " + "  ".join(f"s{s}avg" for s in CLIENT_SEEDS) + \
          f"  {'EvalReturn':>10}  {'HeldSR':>6}  {'AvgLen':>6}"
    print(hdr)
    print("-" * len(hdr))

    recent: list[deque] = [deque(maxlen=50) for _ in CLIENT_SEEDS]
    obs_list = [e.reset() for e in envs]

    for cycle in range(1, N_CYCLES + 1):

        # --- One episode per seed (round-robin) ---
        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            ep_r, done = 0.0, False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done = env.step(action)
                agent.store(obs, action, reward, next_obs, done)
                agent.maybe_train()
                ep_r += reward
                obs = next_obs
            obs_list[idx] = obs   # next episode starts here (env auto-resets)
            recent[idx].append(ep_r)

        # --- Logging ---
        if cycle % LOG_EVERY == 0:
            avgs = "  ".join(
                f"{np.mean(r):+.2f}" if r else "  n/a" for r in recent
            )
            print(f"{cycle:>6}  {avgs}", flush=True)

        # --- Evaluation on held-out seed ---
        eval_str = ""
        if cycle % EVAL_EVERY == 0:
            stats = evaluate(agent, eval_env, EVAL_EPISODES)
            eval_history.append((cycle, stats["mean_return"],
                                 stats["success_rate"], stats["mean_ep_len"]))
            avgs = "  ".join(
                f"{np.mean(r):+.2f}" if r else "  n/a" for r in recent
            )
            eval_str = (f"{stats['mean_return']:>+10.3f}  "
                        f"{stats['success_rate']:>5.0%}  "
                        f"{stats['mean_ep_len']:>6.1f}")
            print(f"{cycle:>6}  {avgs}  {eval_str}", flush=True)

        if cycle % SAVE_EVERY == 0:
            torch.save(agent.online_net.state_dict(),
                       f"checkpoints/centralized_cycle{cycle}.pt")

    # Final eval
    final = evaluate(agent, eval_env, 50)
    torch.save(agent.online_net.state_dict(),
               "checkpoints/centralized_final.pt")

    for e in envs:
        e.close()
    eval_env.close()

    # Summary
    summary_lines = [
        "=" * 68,
        "SUMMARY — Centralized DQN (upper bound)",
        "=" * 68,
        f"seeds={CLIENT_SEEDS}  held_out={HELD_OUT_SEED}",
        f"cycles={N_CYCLES}  total_eps={N_CYCLES * len(CLIENT_SEEDS)}",
        "",
        "Eval history on held-out seed=7:",
    ]
    for cyc, mr, sr, el in eval_history:
        bar = "#" * int(sr * 20)
        summary_lines.append(
            f"  cycle={cyc:>4}  mean_return={mr:+.3f}  "
            f"success={sr:.0%}  avg_len={el:.1f}  [{bar:<20}]"
        )
    summary_lines += [
        "",
        f"Final 50-ep eval:  mean_return={final['mean_return']:+.3f}"
        f"  ±{final['std_return']:.3f}"
        f"  success_rate={final['success_rate']:.0%}"
        f"  avg_ep_len={final['mean_ep_len']:.1f}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    print("\n" + summary_text)

    with open("checkpoints/summary_centralized.txt", "w") as f:
        f.write(summary_text)
    print("Summary saved to checkpoints/summary_centralized.txt")


if __name__ == "__main__":
    main()
