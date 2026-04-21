"""
Local-only DQN baseline — 3 independent agents, no parameter sharing.

Usage
-----
    python train_local_only.py

Design
------
Three DQN agents each train independently on their assigned maze seed
{0, 42, 99} for the same episode budget per agent as the FL run:
    FL:         50 rounds × 30 eps/round = 1 500 local episodes per client
    Local-only: 1 500 episodes per agent

After training, each agent is evaluated on the held-out seed=7 to measure
generalisation *without* any knowledge sharing.  Results are reported
per-agent and as a mean across agents.

This is the lower bound for the FL comparison (above local-only = FL helped).

Output
------
    checkpoints/local_seed{N}_final.pt   (one per client seed)
    checkpoints/summary_local_only.txt
"""

import os
from collections import deque

import numpy as np
import torch

from agent import DQNAgent
from maze_env import MazeEnv

# ---------------------------------------------------------------------------
# Config — match train_federated.py
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

# Match FL per-client episode budget: 50 rounds × 30 eps = 1500 eps per client
EPISODES_PER_AGENT = 1_500
EVAL_EVERY         = 150   # evaluate every N episodes (matches FL EVAL_EVERY cadence)
EVAL_EPISODES      = 20
EVAL_EPS           = 0.05
LOG_EVERY          = 50

LR            = 1e-4
GAMMA         = 0.99
BATCH_SIZE    = 32
BUFFER_CAP    = 15_000
WARMUP_STEPS  = 2_000
TRAIN_FREQ    = 4
TARGET_UPDATE = 500
EPS_START     = 1.0
EPS_END       = 0.1
EPS_DECAY     = 120_000


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DQNAgent, seed: int, env_kwargs: dict, n_episodes: int) -> dict:
    env = MazeEnv(seed=seed, **env_kwargs)
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
    env.close()
    return dict(
        mean_return  = float(np.mean(returns)),
        std_return   = float(np.std(returns)),
        success_rate = successes / n_episodes,
        mean_ep_len  = float(np.mean(ep_lengths)),
    )


# ---------------------------------------------------------------------------
# Single-agent training
# ---------------------------------------------------------------------------

def train_agent(seed: int, obs_shape: tuple, env_kwargs: dict) -> tuple[DQNAgent, list]:
    """Train one agent on *seed* for EPISODES_PER_AGENT episodes.
    Returns (agent, eval_history) where eval_history is list of
    (episode, mean_return, success_rate, mean_ep_len).
    """
    print(f"\n{'='*68}")
    print(f"  Training agent  seed={seed}")
    print(f"{'='*68}")

    agent = DQNAgent(
        obs_shape=obs_shape, n_actions=N_ACTIONS, device=DEVICE,
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAP, target_update_freq=TARGET_UPDATE,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay_steps=EPS_DECAY,
        warmup_steps=WARMUP_STEPS, train_freq=TRAIN_FREQ,
    )

    env = MazeEnv(seed=seed, **env_kwargs)
    obs = env.reset()
    recent = deque(maxlen=50)
    eval_history = []

    for ep in range(1, EPISODES_PER_AGENT + 1):
        ep_r, done = 0.0, False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            agent.maybe_train()
            ep_r += reward
            obs = next_obs
        recent.append(ep_r)

        if ep % LOG_EVERY == 0:
            avg = float(np.mean(recent))
            print(
                f"  [seed={seed}] ep {ep:>4}/{EPISODES_PER_AGENT}"
                f"  avg50={avg:+.2f}  eps={agent.epsilon():.3f}",
                flush=True,
            )

        if ep % EVAL_EVERY == 0:
            stats = evaluate(agent, HELD_OUT_SEED, env_kwargs, EVAL_EPISODES)
            eval_history.append((ep, stats["mean_return"],
                                 stats["success_rate"], stats["mean_ep_len"]))
            print(
                f"  [seed={seed}] EVAL ep={ep}"
                f"  held_out_return={stats['mean_return']:+.3f}"
                f"  held_out_SR={stats['success_rate']:.0%}"
                f"  avg_len={stats['mean_ep_len']:.1f}",
                flush=True,
            )

    env.close()
    torch.save(agent.online_net.state_dict(),
               f"checkpoints/local_seed{seed}_final.pt")
    print(f"  Saved: checkpoints/local_seed{seed}_final.pt")
    return agent, eval_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("checkpoints", exist_ok=True)

    env_kwargs = dict(
        width=MAZE_W, height=MAZE_H,
        obstacle_count=OBSTACLE_COUNT, shaping_scale=SHAPING_SCALE,
        n_stack=N_STACK, cell_obs_px=CELL_OBS_PX,
        max_episode_steps=MAX_EP_STEPS,
    )

    _tmp      = MazeEnv(seed=0, **env_kwargs)
    obs_shape = _tmp.obs_size
    _tmp.close()

    print("=" * 68)
    print("  Local-only DQN — no parameter sharing (lower bound)")
    print(f"  Seeds: {CLIENT_SEEDS}   Held-out: seed {HELD_OUT_SEED}")
    print(f"  Episodes per agent: {EPISODES_PER_AGENT}   Device: {DEVICE}")
    print(f"  obs_shape={obs_shape}")
    print("=" * 68)

    all_results  = {}
    all_histories = {}

    for seed in CLIENT_SEEDS:
        agent, history = train_agent(seed, obs_shape, env_kwargs)
        final = evaluate(agent, HELD_OUT_SEED, env_kwargs, 50)
        all_results[seed]   = final
        all_histories[seed] = history

    # ---- Summary ----
    summary_lines = [
        "=" * 68,
        "SUMMARY — Local-only DQN (lower bound, no sharing)",
        "=" * 68,
        f"seeds={CLIENT_SEEDS}  held_out={HELD_OUT_SEED}",
        f"episodes_per_agent={EPISODES_PER_AGENT}",
        "",
    ]

    for seed in CLIENT_SEEDS:
        summary_lines.append(f"  Agent seed={seed} — eval history on held-out seed={HELD_OUT_SEED}:")
        for ep, mr, sr, el in all_histories[seed]:
            bar = "#" * int(sr * 20)
            summary_lines.append(
                f"    ep={ep:>4}  mean_return={mr:+.3f}  "
                f"success={sr:.0%}  avg_len={el:.1f}  [{bar:<20}]"
            )
        r = all_results[seed]
        summary_lines.append(
            f"  Final 50-ep:  mean_return={r['mean_return']:+.3f}"
            f"  ±{r['std_return']:.3f}"
            f"  success_rate={r['success_rate']:.0%}"
            f"  avg_ep_len={r['mean_ep_len']:.1f}"
        )
        summary_lines.append("")

    # Cross-agent mean
    mean_sr  = float(np.mean([all_results[s]["success_rate"] for s in CLIENT_SEEDS]))
    mean_ret = float(np.mean([all_results[s]["mean_return"]  for s in CLIENT_SEEDS]))
    mean_len = float(np.mean([all_results[s]["mean_ep_len"]  for s in CLIENT_SEEDS]))
    summary_lines += [
        f"  Mean across agents:  success_rate={mean_sr:.0%}"
        f"  mean_return={mean_ret:+.3f}"
        f"  avg_ep_len={mean_len:.1f}",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    print("\n" + summary_text)

    with open("checkpoints/summary_local_only.txt", "w") as f:
        f.write(summary_text)
    print("Summary saved to checkpoints/summary_local_only.txt")


if __name__ == "__main__":
    main()
