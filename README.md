# Federated DQN Maze

A Deep Q-Network (DQN) reinforcement learning baseline for procedurally generated maze navigation. The agent learns to reach the exit while avoiding moving obstacles, using only a local egocentric view of its surroundings. The project is structured as a single-client baseline in preparation for federated multi-agent training.

---

## Overview

The environment is a pure Python/Pygame maze that mirrors the interface of a previous Unity ML-Agents integration. The DQN agent uses a standard online/target network setup with experience replay, epsilon-greedy exploration, and Huber loss.

```
Agent ──(obs 26f)──► QNetwork (MLP) ──► 5 Q-values ──► action
  │                                                        │
  └──────────────── ReplayBuffer ◄────── env.step() ───────┘
```

---

## Project Structure

```
Federated-DQN-Maze/
├── train_dqn.py              # Training loop entry point
├── play.py                   # Play the maze manually (arrow keys / WASD)
├── test_env.py               # Quick environment smoke test (random agent)
├── requirements.txt
├── .gitignore
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py          # DQN agent (exploration, training, checkpointing)
│   ├── q_network.py          # 3-layer MLP Q-network
│   └── replay_buffer.py      # Uniform circular experience replay buffer
└── maze_env/
    ├── __init__.py
    ├── definitions.py        # Enums: MazeCellType, MazeAction, MazeTerminalReason
    ├── maze_builder.py       # Procedural DFS maze generation + obstacle placement
    ├── maze_env.py           # Gym-style environment wrapper (reset / step / close)
    └── moving_obstacle.py    # Deterministic ping-pong hazard
```

---

## Environment

### Observation Space
A flat vector of **26 floats**:

| Index | Description |
|-------|-------------|
| 0–24  | 5×5 egocentric local grid (row-major, north→south / west→east) |
| 25    | Remaining step fraction (1.0 = episode start, 0.0 = timeout) |

Cell encodings in the local grid:

| Value | Cell type |
|-------|-----------|
| 0.00  | Floor |
| 0.50  | Exit |
| 0.75  | Moving obstacle |
| 1.00  | Wall / out-of-bounds |

### Action Space
5 discrete actions:

| ID | Action |
|----|--------|
| 0  | NoOp   |
| 1  | Up     |
| 2  | Left   |
| 3  | Down   |
| 4  | Right  |

### Rewards & Terminals

| Event | Outcome |
|-------|---------|
| Reach the exit | +1.0, episode ends |
| Collide with a moving obstacle | Terminal (death) |
| Step penalty | Small negative reward each step |
| Timeout | Episode ends |

A positive total return indicates a successful episode (reached exit before penalties exceeded +1.0).

### Maze Generation
The `MazeBuilder` uses an **iterative DFS / recursive backtracker** algorithm with:
- Configurable dimensions (forced odd, ≥ 5), default 21×21
- Start/exit placement via double-BFS diameter approximation
- Deterministic ping-pong obstacles placed on discovered patrol segments
- Obstacle-aware solvability validation
- Automatic fallback to a hardcoded 15×11 layout if all procedural attempts fail
- Optional Pygame rendering

---

## Agent

`DQNAgent` implements a standard DQN with the following features:

| Feature | Detail |
|---------|--------|
| Q-network | 3-layer MLP: 26 → 128 → 128 → 5 |
| Exploration | Linear ε-greedy decay: 1.0 → 0.05 over 10k steps |
| Replay buffer | Uniform circular buffer, capacity 50k |
| Loss | Smooth-L1 (Huber) |
| Target network | Hard sync every 1,000 gradient steps |
| Gradient clipping | max norm 10.0 |
| Optimizer | Adam, lr=1e-3 |
| Discount factor | γ = 0.99 |

---

## Training

### Hyperparameters (defaults in `train_dqn.py`)

| Parameter | Value |
|-----------|-------|
| `OBS_SIZE` | 26 |
| `N_ACTIONS` | 5 |
| `HIDDEN_SIZE` | 128 |
| `LR` | 1e-3 |
| `GAMMA` | 0.99 |
| `BATCH_SIZE` | 64 |
| `BUFFER_CAPACITY` | 50,000 |
| `WARMUP_STEPS` | 1,000 |
| `TRAIN_FREQ` | 4 (gradient step every 4 env steps) |
| `TARGET_UPDATE_FREQ` | 1,000 gradient steps |
| `EPS_START / EPS_END` | 1.0 / 0.05 |
| `EPS_DECAY_STEPS` | 10,000 |
| `MAX_EPISODES` | 2,000 |
| `EVAL_EVERY` | 50 episodes |
| `EVAL_EPISODES` | 10 episodes |
| `SAVE_EVERY` | 200 episodes |

### Training Loop Design
The loop is **step-based**, not episode-based. When `done=True`, the environment auto-resets internally and returns the first observation of the new episode as `next_obs`, so the training loop requires no special reset handling.

Checkpoints are saved to `checkpoints/dqn_ep{N}.pt`. To resume from a checkpoint, set `RESUME_FROM` in `train_dqn.py`.

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements:
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `pygame >= 2.0.0`

---

## Usage

### Train the DQN agent

```bash
python train_dqn.py
```

Training output (every 10 episodes):
```
Ep   10 | return +0.823 | avg100 +0.412 | eps 0.950 | steps 4,231 | buf 4,231 | loss 0.0341
  [EVAL ep=50] mean_return=+0.612 ±0.211  success_rate=60%
  Saved: checkpoints/dqn_ep200.pt
```

To resume from a checkpoint, edit `train_dqn.py`:
```python
RESUME_FROM: Optional[str] = "checkpoints/dqn_ep200.pt"
```

Checkpoints are saved to `checkpoints/` (gitignored) every 200 episodes.

### Play the maze yourself

```bash
python play.py
```

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move |
| R | New maze |
| Q / Escape | Quit |

### Test the environment (random agent)

```bash
python test_env.py
```

Runs 3 episodes with random actions and prints the return. Set `RENDER = False` in the file for headless mode.

---

## Roadmap

- [ ] Prioritized experience replay
- [ ] Double DQN / Dueling network heads
- [ ] CNN observation encoder (reshape 25 grid cells to 5×5)
- [ ] Federated aggregation: multiple independent agents, periodic weight averaging
- [ ] Curriculum: progressive maze size / obstacle count increase
