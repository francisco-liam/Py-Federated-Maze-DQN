# Federated DQN Maze

CS 791 — Edge AI · Final Project · Liam Francisco

A pixel-based Deep Q-Network (DQN) agent trained to navigate procedurally generated mazes using only a limited line-of-sight view, extended to a federated learning experiment where multiple agents train on structurally different maze topologies and periodically share weights.

---

## Overview

The environment is a pure-Python maze with Pygame rendering. The DQN agent observes a 5×5-cell egocentric window rendered to pixels (40×40), with walls occluding sight lines. Four frames are stacked (Atari-style) and a step-fraction channel appended to break perceptual aliasing.

```
              ┌─────────────────────────────┐
              │   FL Server (global conv)   │
              │   fedavg(conv only) ←───┐   │
              └────┬────────┬───────┬───┘   │
         conv ↓    │   conv ↓  conv ↓       │ conv only
              ┌────▼──┐ ┌───▼───┐ ┌──▼────┐ │
              │seed=0 │ │seed=42│ │seed=99│─┘ ← non-IID clients
              │ own FC│ │ own FC│ │ own FC│    (FC layers private)
              └───────┘ └───────┘ └───────┘
                        evaluated on seed=7 (held-out)
```

The key finding: **naive FedAvg (all layers) fails on heterogeneous DQN**. Q-values encode maze-specific BFS distances and cannot be averaged across different topologies. Averaging only the convolutional layers (topology-agnostic feature detectors) while keeping per-client FC layers private fixes the problem.

---

## Results

### Single-agent baseline (seed=0)
- **100% success rate**, mean return = **+6.68** at 1,000 episodes
- Confirms DQN and the environment work correctly

### Comparison table — generalisation to held-out seed=7

| Method | Client SR (own maze) | Held-out SR | Notes |
|---|---|---|---|
| Single-agent baseline | 100% (seed=0 only) | — | Upper bound on own seed |
| Local-only DQN | ~100% own seed* | 0% | No knowledge sharing |
| Centralized DQN (pooled) | 0% | 0% | Single Q-fn can't serve 3 topologies |
| FedAvg — all layers | 100% / 0% / 0% | 0% | seed=0 dominates aggregation |
| FedAvg — partial conv | ~100% all clients | **100%**† | Our fix |
| FedProx µ=0.1 — partial conv | ~100% all clients | **100%**† | µ hurts conv quality slightly |

*verified via training logs · †with 200 fine-tune episodes on seed=7

### Transfer test (200 fine-tune eps on seed=7, 50 eval eps)

| Init | Success rate | Mean return | Δ vs random |
|---|---|---|---|
| FedAvg partial conv | **100%** | +0.484 ± 0.035 | **+84%** |
| FedProx partial conv | **100%** | +0.121 ± 0.066 | **+32%** |
| Random init | 16–68% | varies | baseline |

The FL conv features enable fast adaptation: 100% success in 200 episodes vs 16–68% from random initialisation over the same budget.

---

## Project Structure

```
Py-Federated-Maze-DQN/
├── train_dqn.py              # Single-agent baseline
├── train_federated.py        # FL training (FedAvg / FedProx + partial conv)
├── train_centralized.py      # Centralized upper-bound (pooled seeds)
├── train_local_only.py       # Local-only lower-bound (no sharing)
├── transfer_test.py          # FL conv vs random init fine-tune comparison
├── run_all.sh                # Run all experiments in order
├── play.py                   # Play the maze manually (arrow keys / WASD)
├── test_env.py               # Environment smoke test (random agent)
├── requirements.txt
├── results/                  # All summary .txt and transfer test logs
├── checkpoints/              # Saved model weights
├── logs/                     # Full training logs from run_all.sh
├── agent/
│   ├── dqn_agent.py          # DQN agent + FL helpers
│   ├── q_network.py          # Nature-DQN CNN: Conv×3 → FC×2
│   └── replay_buffer.py      # Uniform replay buffer
├── maze_env/
│   ├── maze_env.py           # Gym-style env: LOS rendering, frame stack, shaping
│   ├── maze_builder.py       # DFS maze gen + BFS exit-distance map
│   ├── definitions.py        # Enums: MazeCellType, MazeAction, etc.
│   └── moving_obstacle.py    # Ping-pong hazard
└── federated/
    ├── fl_server.py          # FLServer: global conv, fedavg(), transfer_evaluate()
    └── fl_client.py          # FLClient: local env + agent, train_round()
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
| Shaping | `0.05 × (φ(s′) − φ(s))`, `φ = −BFS_dist_to_exit` |

Potential-based shaping provides dense gradient signal without changing the optimal policy.

---

## Network Architecture

Nature-DQN CNN with Double DQN updates:

| Layer | Detail |
|---|---|
| Conv1 | 5→32, 4×4, stride 2 |
| Conv2 | 32→64, 3×3, stride 2 |
| Conv3 | 64→64, 3×3, stride 1 |
| FC1 | 512 units, ReLU |
| FC2 | 5 Q-values |

**Partial parameter sharing:** Only `conv.*` layers are shared via FedAvg. `fc.*` layers remain private to each client. Implemented in `federated/fl_server.py` (`_SHARED_KEY`) and `agent/dqn_agent.py` (`load_global_weights`).

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: `torch >= 2.0`, `numpy >= 1.24`, `pygame >= 2.0`

---

## Running Experiments

### Single-agent baseline
```bash
python train_dqn.py
```

### All experiments (reproduces all results)
```bash
bash run_all.sh
```

Runs in order: local-only → centralized → FedAvg (all layers) → FedAvg (partial conv) → FedProx (partial conv) → transfer tests. Logs saved to `logs/`, summaries to `checkpoints/` and `results/`.

### Individual runs
```bash
# Federated — FedAvg partial conv (main result)
python train_federated.py --equal_weight

# Federated — FedProx partial conv
python train_federated.py --equal_weight --mu 0.1

# Transfer test
python transfer_test.py --checkpoint checkpoints/fl_fedavg_eqw_final.pt --finetune 200 --eval 50

# Centralized upper bound
python train_centralized.py

# Local-only lower bound
python train_local_only.py

# Play manually
python play.py
```

---

## Key Finding

> Naive FedAvg fails on heterogeneous DQN because Q-value magnitudes are topology-dependent. Partial parameter sharing — averaging only convolutional layers while keeping FC layers per-client — resolves the incompatibility. Shared conv features reduce fine-tuning time on unseen mazes from inconsistent (16–68%) to reliable (100%) within the same episode budget.

Centralized training (all seeds pooled into one agent) also fails, showing the problem is fundamental to heterogeneous Q-learning — not specific to FL communication overhead.
