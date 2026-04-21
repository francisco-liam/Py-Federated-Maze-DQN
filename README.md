# Federated DQN Maze

A pixel-based Deep Q-Network (DQN) agent trained to navigate procedurally generated mazes using only a limited line-of-sight view — then extended to a federated learning (FL) experiment where multiple agents train on different maze topologies and periodically share weights.

---

## Overview

The environment is a pure Python maze with Pygame rendering. The DQN agent observes a 5×5-cell egocentric window rendered to pixels (40×40), with walls occluding sight lines. Four frames are stacked (Atari-style) and a step-fraction channel is appended to break perceptual aliasing.

The project has two training modes:

1. **Single-agent baseline** (`train_dqn.py`) — one agent, one fixed maze seed
2. **Federated training** (`train_federated.py`) — 3 clients each on a different maze seed, aggregated via FedAvg or FedProx, evaluated on a held-out seed

```
              ┌─────────────────────────────┐
              │  FL Server (global QNetwork)│
              │  fedavg() ← client weights  │
              └────┬────────┬────────┬──────┘
                   │        │        │  broadcast global weights
              ┌────▼──┐ ┌───▼───┐ ┌──▼────┐
              │seed=0 │ │seed=42│ │seed=99│  ← non-IID clients
              └───────┘ └───────┘ └───────┘
                              evaluated on seed=7 (held-out)
```

---

## Project Structure

```
Federated-DQN-Maze/
├── train_dqn.py              # Single-agent baseline training
├── train_federated.py        # Federated training (FedAvg / FedProx)
├── prepare_fl.py             # Maze diversity audit + baseline training
├── play.py                   # Play the maze manually (arrow keys / WASD)
├── test_env.py               # Quick environment smoke test (random agent)
├── requirements.txt
├── agent/
│   ├── dqn_agent.py          # DQN agent + FL helpers (get/load/proximal weights)
│   ├── q_network.py          # Nature-DQN CNN: Conv×3 → FC×2
│   └── replay_buffer.py      # Uniform circular replay buffer
├── maze_env/
│   ├── definitions.py        # Enums: MazeCellType, MazeAction, MazeTerminalReason
│   ├── maze_builder.py       # DFS maze gen + BFS exit-distance map
│   ├── maze_env.py           # Gym-style env: LOS rendering, frame stack, shaping
│   └── moving_obstacle.py    # Ping-pong hazard
└── federated/
    ├── fl_client.py          # FLClient: local env + agent, train_round()
    └── fl_server.py          # FLServer: global QNetwork, fedavg(), evaluate()
```

---

## Environment

### Observation Space
Shape `(5, 40, 40)` — float32 tensor:

| Channel(s) | Content |
|---|---|
| 0–3 | 4 stacked greyscale LOS frames (8 px/cell, 40×40 canvas) |
| 4 | Step-fraction broadcast to all pixels (breaks perceptual aliasing) |

Line-of-sight uses Bresenham ray-casting — walls block vision of cells behind them.

### Action Space
5 discrete actions: `NoOp, Up, Left, Down, Right`

### Rewards
| Event | Reward |
|---|---|
| Reach exit | +1.0, episode ends |
| Each step | −1/max\_steps |
| Timeout | episode ends |
| Shaping | `0.05 × (φ(s′) − φ(s))`, `φ = −BFS_dist_to_exit` |

Potential-based shaping (`φ(s′)−φ(s)`) provides dense gradient signal without changing the optimal policy.

---

## Agent

Nature-DQN CNN with Double DQN updates:

| Layer | Detail |
|---|---|
| Conv1 | 5→32, 4×4, stride 2 |
| Conv2 | 32→64, 3×3, stride 2 |
| Conv3 | 64→64, 3×3, stride 1 |
| FC1 | 512 units, ReLU |
| FC2 | 5 Q-values |

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Discount γ | 0.99 |
| Batch size | 32 |
| Buffer capacity | 200k (baseline) / 15k (FL) |
| Warmup steps | 5,000 (baseline) / 2,000 (FL) |
| ε decay | 1.0 → 0.1 over 50k steps |
| Eval ε | 0.05 |
| Target sync | every 1,000 grad steps |

---

## Training

### Single-agent baseline

```bash
python3 train_dqn.py
```

Trains on seed=0, 21×21 maze, no obstacles. Achieves 100% success by episode ~400, mean_return≈+6.68 at episode 1000.

Checkpoint saved to `checkpoints/baseline_seed0_ep1000.pt`.

### Federated training

```bash
# FedAvg — step-weighted aggregation (default)
python3 train_federated.py

# FedAvg — equal-weight aggregation
python3 train_federated.py --equal_weight

# FedProx — proximal penalty to reduce client drift
python3 train_federated.py --mu 0.1 --equal_weight
```

Each run saves:
- `checkpoints/fl_<tag>_final.pt` — final global model
- `checkpoints/summary_<tag>.txt` — eval history on held-out seed

FL config (in `train_federated.py`):

| Parameter | Value |
|---|---|
| Client seeds | 0, 42, 99 |
| Held-out eval seed | 7 |
| FL rounds | 50 |
| Local episodes/round | 30 |
| Buffer capacity | 15k per client |
| ε decay | 1.0 → 0.1 over 120k steps (spans full training) |

### Play the maze yourself

```bash
python3 play.py
```

---

## FL Experiment Results

### Baseline (single-agent, seed=0, evaluated on seed=7)

The baseline agent reaches 100% success on its training maze (seed=0) but has not been evaluated zero-shot on seed=7 — this is the comparison target.

### Federated results (held-out seed=7)

All experiments so far achieve **0% success on seed=7**. Clients individually converge (seed=0 client reaches +2–3 mean return) but the averaged global model fails everywhere.

**Diagnosis:** This is a textbook heterogeneous FL failure. Q-values learned on three structurally different maze topologies are on incompatible scales — averaging them destroys all three local policies simultaneously. The `sr0/42/99` diagnostic columns (per-client success rate of the *global* model) consistently show 100%/0%/0%, confirming seed=0's Q-functions dominate the average.

**Experiments run:**

| Method | Rounds | Local eps | Held-out SR |
|---|---|---|---|
| FedAvg (step-weighted) | 30 | 50 | 0% |
| FedAvg+EqW | 60 | 15 | 0% |
| FedProx mu=0.1+EqW | 30 | 50 | 0% |
| FedAvg (eps_reset/round) | 60 | 15 | 0% |
| FedAvg (eps_decay=120k) | 50 | 30 | *in progress* |

### Next steps

- [ ] Personalised FL: shared CNN trunk, per-client output heads
- [ ] FedMA: layer-wise weight matching before averaging
- [ ] Reduce non-IID severity: seeds with similar BFS distance distributions
- [ ] Document failure as a finding: motivates personalised FL for RL

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch >= 2.0`, `numpy >= 1.24`, `pygame >= 2.0`


