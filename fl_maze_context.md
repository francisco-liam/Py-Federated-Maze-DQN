# FL Maze DQN — Session Context (April 20 2026)

## Project
/home/liamf/Py-Federated-Maze-DQN — Federated DQN on procedural mazes

## What works
- Single-agent baseline: 100% success, mean_return=+6.682, seed=0, 1000 eps
  - checkpoint: checkpoints/baseline_seed0_ep1000.pt
- Pixel obs (5,40,40): 4-frame LOS stack + step-fraction channel
- CNN: Conv(5→32,4×4/s2)→Conv(32→64,3×3/s2)→Conv(64→64,3×3/s1)→FC(512)→FC(5)
- Reward shaping: φ(s')-φ(s), φ=-BFS_dist, scale=0.05

## FL results so far — ALL 0% on held-out seed=7
- FedAvg (step-weighted, 30 rounds, 50 eps/round): 0%
- FedAvg+EqW (equal weight, 60 rounds, 15 eps/round): 0%  
- FedProx mu=0.1+EqW (30 rounds, 50 eps/round): 0%
- FedAvg (60 rounds, 30 eps/round, eps_reset per round): 0% — currently running

## Root cause diagnosis
- Client sr0/42/99 columns show: often 100%/0%/0% — seed 0 client dominates
- Clients' local policies converge but averaged global model fails everywhere
- Q-values from 3 different maze topologies are incompatible to average
- This is textbook heterogeneous FL failure (non-IID RL)

## Current config (train_federated.py)
- FL_ROUNDS=50, LOCAL_EPISODES=30, EPS_DECAY=120_000
- BUFFER_CAP=15_000, WARMUP=2_000, LR=1e-4
- CLIENT_SEEDS=[0,42,99], HELD_OUT_SEED=7
- Equal weight and FedProx available via --equal_weight, --mu flags

## Key files
- federated/fl_server.py: FLServer with fedavg(equal_weight=False)
- federated/fl_client.py: FLClient with per-episode progress print
- agent/dqn_agent.py: _eps_steps counter (separate from steps_done), FL methods
- train_federated.py: saves summary_<tag>.txt to checkpoints/

## Literature — FL for DQNs
- FedAvg on DQN works for homogeneous envs (same game), fails heterogeneous — confirmed by our results
- Collaborative DRL (Zhuo et al. 2019): share gradients not weights — avoids Q-value scale incompatibility
- FedProx for RL: proximal term helps but doesn't fully solve heterogeneous RL (also confirmed)
- Federated Actor-Critic: policy nets (probs) more compatible to average than Q-nets (absolute values)
- Personalised FedRL: shared CNN trunk + local FC heads — most cited fix for our exact problem
- FedKL / knowledge distillation FL: server distills client policies rather than averaging weights
- Partial parameter sharing: average only conv layers, keep per-client FC layers
  → search: "partial parameter sharing federated reinforcement learning"

## Why our setting is hard
Q-values encode cumulative discounted reward — different maze topologies → different BFS distances →
different Q-value magnitudes. FedAvg designed for supervised learning where outputs share target space.
Q-values do not. Conv layers (learn corridors/dead-ends) ARE compatible; FC layers are not.

## What to try tomorrow (priority order)
1. Check result of current run (50 rounds, 30 eps, eps_decay=120k)
2. Partial parameter sharing — average only conv layers, freeze per-client FC layers
   - In fl_server.py: fedavg() filters keys — only "conv." keys get averaged
   - In fl_client.py / dqn_agent.py: load_global_weights() only loads conv keys
   - This is the lowest-effort highest-impact fix
3. If that works: add FedProx on top (proximal term on conv layers only)
4. If still failing: accept as finding, document that FedAvg fails on heterogeneous DQN,
   propose personalised FL (shared trunk + local heads) as future work
   - This IS a valid research result — motivates the more sophisticated approaches above
