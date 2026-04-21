# Project Context — Federated DQN Maze

Pick-up notes for resuming work. Last updated: April 20 2026.

---

## Where We Are

Single-agent DQN baseline is solid:
- 100% success rate on seed=0, mean_return ≈ +6.68 at 1000 episodes
- Checkpoint: `checkpoints/baseline_seed0_ep1000.pt`

Federated experiments have all returned **0% success on held-out seed=7**:

| Method | Rounds | Local eps/round | Result |
|---|---|---|---|
| FedAvg (step-weighted) | 30 | 50 | 0% |
| FedAvg + equal weight | 60 | 15 | 0% |
| FedProx mu=0.1 + EqW | 30 | 50 | 0% |
| FedAvg (eps reset/round) | 60 | 15 | 0% |
| FedAvg (eps_decay=120k) | 50 | 30 | *check result* |

Diagnostic: `sr0/42/99` columns (global model eval on each client seed) consistently show
**100% / 0% / 0%** — seed=0's Q-functions dominate the average and destroy seeds 42 and 99.

---

## Root Cause

Q-values encode cumulative discounted reward. Different maze topologies have different BFS
distances to exit, so Q-value magnitudes are incompatible across clients. FedAvg was designed
for supervised learning (shared label space). Raw Q-value averaging does not work.

The convolutional layers (detecting corridors, corners, dead-ends) *are* topology-agnostic
and should be compatible to average. The fully-connected layers (mapping features to Q-values)
are not.

---

## Next Steps (priority order)

### 1. Check the current run result
```bash
cat checkpoints/summary_fedavg.txt
```
If still 0%, proceed to step 2.

### 2. Partial parameter sharing — **most promising, lowest effort**

Average only the `conv.*` layers across clients; keep per-client `fc.*` layers untouched.

**Changes needed:**

`federated/fl_server.py` — filter keys in `fedavg()`:
```python
# Only average convolutional layers
SHARED_LAYERS = lambda k: k.startswith("conv.")

def fedavg(self, client_results, equal_weight=False):
    ...
    for key in client_results[0][0].keys():
        if SHARED_LAYERS(key):
            new_weights[key] = ...  # average as normal
        # fc.* keys: don't touch the global model's fc layers
```

`agent/dqn_agent.py` — `load_global_weights()` only loads conv keys:
```python
def load_global_weights(self, state_dict):
    current = self.online_net.state_dict()
    for k, v in state_dict.items():
        if k.startswith("conv."):
            current[k] = v
    self.online_net.load_state_dict(current)
    self.target_net.load_state_dict(self.online_net.state_dict())
    self.optimizer = optim.Adam(self.online_net.parameters(), lr=self._lr)
```

### 3. FedProx on conv layers only

Once partial sharing works, add `--mu 0.1` — proximal term now anchors only the
conv layers, reducing client drift in the shared feature extractor.

### 4. If still failing — accept as a finding

This is a valid research result:
> "Naive FedAvg fails on heterogeneous DQN because Q-value magnitudes are
> topology-dependent. Partial parameter sharing (conv layers only) is required
> for successful federated DRL on structurally diverse environments."

Future work section: personalised FL (shared CNN trunk + per-client FC heads trained
jointly), FedMA (layer-wise matching), knowledge distillation FL.

---

## Relevant Literature

- **FedAvg on DQN** — works for homogeneous envs (same game, different seeds), fails heterogeneous
- **Collaborative DRL** (Zhuo et al. 2019) — share gradients not weights; avoids scale incompatibility
- **FedProx for RL** — proximal term reduces drift but does not fix Q-value incompatibility
- **Federated Actor-Critic** — policy nets (action probabilities) more compatible to average than Q-nets
- **Personalised FedRL** — shared CNN trunk + per-client output heads; most cited fix for this problem
- **Partial parameter sharing** — average only early layers; local heads stay client-specific
  - Search: *"partial parameter sharing federated reinforcement learning"*
- **FedKL / knowledge distillation FL** — server distills client policies rather than averaging weights directly

---

## Key Files

| File | Purpose |
|---|---|
| `train_dqn.py` | Single-agent baseline |
| `train_federated.py` | FL training — `--mu`, `--equal_weight` flags |
| `federated/fl_server.py` | `FLServer`: `fedavg(equal_weight=False)`, `evaluate()` |
| `federated/fl_client.py` | `FLClient`: per-episode progress print, `train_round()` |
| `agent/dqn_agent.py` | `DQNAgent`: `_eps_steps` counter, FL methods |
| `checkpoints/summary_*.txt` | Per-run eval history on held-out seed=7 |

## Quick Commands

```bash
# Check current run result
cat checkpoints/summary_fedavg.txt

# Run FedAvg
python3 train_federated.py

# Run FedAvg with equal weighting
python3 train_federated.py --equal_weight

# Run FedProx
python3 train_federated.py --mu 0.1 --equal_weight

# Evaluate baseline on held-out seed=7
python3 -c "
from federated import FLServer
from maze_env import MazeEnv
env_kwargs = dict(width=21, height=21, obstacle_count=0, shaping_scale=0.05,
                  n_stack=4, cell_obs_px=8, max_episode_steps=200)
tmp = MazeEnv(seed=0, **env_kwargs); obs_shape = tmp.obs_size; tmp.close()
s = FLServer(obs_shape=obs_shape, n_actions=5, device='cuda')
s.load('checkpoints/baseline_seed0_ep1000.pt')
r = s.evaluate(seed=7, n_episodes=50, env_kwargs=env_kwargs)
print(r)
"
```
