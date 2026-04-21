"""
prepare_fl.py — Two-stage preparation script for the FL experiment.

Stage 1  Maze diversity audit
  Generates mazes for the three FL client seeds plus the held-out eval seed
  and reports structural metrics to confirm the non-IID property.

Stage 2  Baseline DQN training (1 000 episodes, seed=0)
  Trains a single-client DQN on the seed-0 maze and saves the checkpoint.
  The final summary line is designed to be copied back for review.

Usage
-----
    python3 prepare_fl.py

Output files
------------
  checkpoints/baseline_seed0_ep1000.pt  — baseline model weights
"""

import os
from collections import deque

import numpy as np
import torch

from agent import DQNAgent
from maze_env import MazeEnv
from maze_env.definitions import MazeCellType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIENT_SEEDS  = [0, 42, 99]          # one per FL client
HELD_OUT_SEED = 7                    # never seen during training
ALL_SEEDS     = CLIENT_SEEDS + [HELD_OUT_SEED]

MAZE_W, MAZE_H   = 21, 21
OBSTACLE_COUNT   = 0                 # baseline: no obstacles
SHAPING_SCALE    = 0.05
N_STACK          = 4
CELL_OBS_PX      = 8
MAX_EP_STEPS     = 200

# Training (matches train_dqn.py)
LR               = 1e-4
GAMMA            = 0.99
BATCH_SIZE       = 32
BUFFER_CAPACITY  = 200_000
WARMUP_STEPS     = 5_000
TRAIN_FREQ       = 4
TARGET_UPDATE    = 1_000
EPS_START        = 1.0
EPS_END          = 0.1
EPS_DECAY_STEPS  = 50_000
MAX_EPISODES     = 1_000
EVAL_EVERY       = 100
EVAL_EPISODES    = 10
LOG_EVERY        = 10
EVAL_EPS         = 0.05
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bfs_path_length(builder, start, exit_cell):
    """Return BFS distance from start to exit using the builder's cells."""
    from collections import deque as _deque
    dist = {start: 0}
    q = _deque([start])
    while q:
        cur = q.popleft()
        if cur == exit_cell:
            return dist[cur]
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nxt = (cur[0]+dx, cur[1]+dy)
            if nxt not in dist and builder.is_walkable(nxt):
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return -1   # unreachable


def _junction_count(builder):
    """Count corridor junctions: walkable cells with ≥ 3 open neighbours."""
    junctions = 0
    for pos, cell in builder.cells.items():
        if cell == MazeCellType.Wall:
            continue
        open_nb = sum(
            1 for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]
            if builder.is_walkable((pos[0]+dx, pos[1]+dy))
        )
        if open_nb >= 3:
            junctions += 1
    return junctions


def _floor_cells(builder):
    return sum(1 for c in builder.cells.values() if c != MazeCellType.Wall)


def _corridor_straightness(builder):
    """
    Fraction of floor cells that lie in a straight run (exactly 2 open
    neighbours that are collinear).  Higher = more straight corridors,
    lower = more twisty / room-like.
    """
    straight = 0
    floor    = 0
    for pos, cell in builder.cells.items():
        if cell == MazeCellType.Wall:
            continue
        floor += 1
        x, y = pos
        h_open = builder.is_walkable((x-1,y)) and builder.is_walkable((x+1,y))
        v_open = builder.is_walkable((x,y-1)) and builder.is_walkable((x,y+1))
        if h_open ^ v_open:   # exactly one axis open end-to-end
            straight += 1
    return straight / floor if floor else 0.0


# ---------------------------------------------------------------------------
# Stage 1 — Maze diversity audit
# ---------------------------------------------------------------------------

def audit_mazes():
    print("=" * 60)
    print("STAGE 1 — MAZE DIVERSITY AUDIT")
    print("=" * 60)
    print(f"{'Seed':>6}  {'Role':>8}  {'Start':>8}  {'Exit':>8}  "
          f"{'BFS':>5}  {'Floor':>6}  {'Junc':>5}  {'Straight':>8}")
    print("-" * 65)

    results = {}
    for seed in ALL_SEEDS:
        env = MazeEnv(
            seed=seed, width=MAZE_W, height=MAZE_H,
            obstacle_count=OBSTACLE_COUNT, shaping_scale=0.0,
            max_episode_steps=MAX_EP_STEPS,
        )
        b    = env._builder
        bfs  = _bfs_path_length(b, b.start_cell, b.exit_cell)
        flr  = _floor_cells(b)
        junc = _junction_count(b)
        strn = _corridor_straightness(b)
        role = "held-out" if seed == HELD_OUT_SEED else "client"
        print(f"{seed:>6}  {role:>8}  {str(b.start_cell):>8}  "
              f"{str(b.exit_cell):>8}  {bfs:>5}  {flr:>6}  "
              f"{junc:>5}  {strn:>8.3f}")
        results[seed] = dict(bfs=bfs, floor=flr, junc=junc, straight=strn,
                             start=b.start_cell, exit=b.exit_cell)
        env.close()

    print()
    # Pairwise BFS distance difference as a diversity proxy
    seeds = CLIENT_SEEDS
    bfs_vals = [results[s]["bfs"] for s in seeds]
    print(f"Client BFS distances:   {bfs_vals}")
    print(f"BFS range:              {max(bfs_vals) - min(bfs_vals)} steps")
    junc_vals = [results[s]["junc"] for s in seeds]
    print(f"Client junction counts: {junc_vals}")
    print(f"Junction range:         {max(junc_vals) - min(junc_vals)} cells")
    print()
    verdict = (
        "GOOD — mazes are structurally diverse (non-IID)"
        if (max(bfs_vals) - min(bfs_vals)) >= 10
        else "WARN — mazes may be too similar; consider different seeds"
    )
    print(f"Diversity verdict: {verdict}")
    print()
    return results


# ---------------------------------------------------------------------------
# Stage 2 — Baseline training (seed=0, 1 000 episodes)
# ---------------------------------------------------------------------------

def train_baseline():
    print("=" * 60)
    print("STAGE 2 — BASELINE TRAINING  (seed=0, 1 000 episodes)")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/baseline_seed0_ep1000.pt"

    with MazeEnv(
        seed=0, width=MAZE_W, height=MAZE_H,
        obstacle_count=OBSTACLE_COUNT, shaping_scale=SHAPING_SCALE,
        n_stack=N_STACK, cell_obs_px=CELL_OBS_PX,
        max_episode_steps=MAX_EP_STEPS,
    ) as env:
        agent = DQNAgent(
            obs_shape=env.obs_size, n_actions=env.n_actions,
            lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY, target_update_freq=TARGET_UPDATE,
            eps_start=EPS_START, eps_end=EPS_END, eps_decay_steps=EPS_DECAY_STEPS,
            warmup_steps=WARMUP_STEPS, train_freq=TRAIN_FREQ, device=DEVICE,
        )
        print(f"obs_shape={env.obs_size}  device={DEVICE}\n")

        obs           = env.reset()
        recent        = deque(maxlen=100)
        episode       = 0
        ep_return     = 0.0
        last_loss     = None
        eval_history  = []   # (episode, mean_return, success_rate)

        while episode < MAX_EPISODES:
            action = agent.select_action(obs)
            next_obs, reward, done = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.maybe_train()
            if loss is not None:
                last_loss = loss
            ep_return += reward
            obs = next_obs

            if done:
                episode += 1
                agent.episodes_done = episode
                recent.append(ep_return)

                if episode % LOG_EVERY == 0:
                    avg      = np.mean(recent) if recent else float("nan")
                    loss_str = f"{last_loss:.4f}" if last_loss else "n/a"
                    print(
                        f"Ep {episode:5d} | return {ep_return:+.3f} | "
                        f"avg100 {avg:+.3f} | eps {agent.epsilon():.3f} | "
                        f"steps {agent.steps_done:,d} | "
                        f"buf {len(agent.buffer):,d} | loss {loss_str}"
                    )

                if episode % EVAL_EVERY == 0:
                    obs, stats = _evaluate(agent, env, obs)
                    eval_history.append((episode, stats["mean_return"], stats["success_rate"]))
                    print(
                        f"  [EVAL ep={episode}] "
                        f"mean_return={stats['mean_return']:+.3f} "
                        f"±{stats['std_return']:.3f}  "
                        f"success_rate={stats['success_rate']:.0%}"
                    )

                ep_return = 0.0

        agent.save(ckpt_path)
        print(f"\nCheckpoint saved: {ckpt_path}")
        return eval_history


def _evaluate(agent, env, obs):
    returns, successes = [], 0
    for _ in range(EVAL_EPISODES):
        ep_r, done = 0.0, False
        while not done:
            if np.random.rand() < EVAL_EPS:
                action = np.random.randint(agent.n_actions)
            else:
                action = agent.select_greedy_action(obs)
            obs, r, done = env.step(action)
            ep_r += r
        returns.append(ep_r)
        if ep_r > 0:
            successes += 1
    stats = dict(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        success_rate=successes / EVAL_EPISODES,
    )
    return obs, stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(maze_results, eval_history):
    print()
    print("=" * 60)
    print("SUMMARY — PASTE THIS BACK")
    print("=" * 60)
    print()
    print("--- MAZE DIVERSITY ---")
    for seed in ALL_SEEDS:
        r    = maze_results[seed]
        role = "held-out" if seed == HELD_OUT_SEED else "client "
        print(f"  seed={seed:>3} ({role})  start={r['start']}  exit={r['exit']}  "
              f"bfs={r['bfs']}  floor={r['floor']}  junc={r['junc']}  "
              f"straight={r['straight']:.3f}")

    print()
    print("--- BASELINE EVAL HISTORY (seed=0, 1000 ep) ---")
    for ep, mean_r, sr in eval_history:
        bar = "#" * int(sr * 20)
        print(f"  ep={ep:>4}  mean_return={mean_r:+.3f}  success={sr:.0%}  [{bar:<20}]")

    final_sr   = eval_history[-1][2] if eval_history else 0.0
    final_ret  = eval_history[-1][1] if eval_history else 0.0
    verdict = (
        "BASELINE SOLID — ready to build FL"
        if final_sr >= 0.8
        else "BASELINE WEAK — may need more episodes before FL"
    )
    print()
    print(f"Final eval:  success_rate={final_sr:.0%}  mean_return={final_ret:+.3f}")
    print(f"Verdict:     {verdict}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    maze_results = audit_mazes()
    eval_history = train_baseline()
    print_summary(maze_results, eval_history)
