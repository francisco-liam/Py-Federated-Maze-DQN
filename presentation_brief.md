# Presentation Brief — Federated DQN Maze Navigation

CS 791 · Edge AI · Liam Francisco

Use this document to help an AI assistant draft presentation slides or talking points. It contains the full context, results, and suggested narrative structure.

---

## Course Context

CS 791 — Edge AI. Final project on federated learning applied to reinforcement learning. The proposal originally used Unity ML-Agents; the implementation was done in pure Python/Pygame instead (simpler, no engine dependency, same research contribution).

---

## One-Sentence Summary

> We show that naive federated averaging fails on heterogeneous deep Q-networks because Q-values are topology-dependent, and that averaging only the convolutional layers (shared perceptual features) while keeping fully-connected layers (maze-specific Q-values) private fixes the problem and enables reliable generalisation to unseen mazes.

---

## Suggested Slide Structure

### Slide 1 — Title
- Federated DQN Maze Navigation
- CS 791 · Edge AI · Liam Francisco · April 2026

### Slide 2 — Motivation
- Federated learning (FL) trains models across distributed clients without sharing raw data
- Most FL research is for supervised learning (shared label space)
- Reinforcement learning is harder: each agent generates its own experience, rewards depend on local environment dynamics
- **Research question:** Can FL improve policy generalisation for agents trained in heterogeneous environments?

### Slide 3 — Setup
- Environment: procedurally generated 21×21 mazes (Python/Pygame)
- Agent observes a 5×5 line-of-sight window rendered to 40×40 pixels
- DQN with 3 conv layers + 2 FC layers
- 3 clients each on a different maze seed (0, 42, 99) — structurally different layouts
- Evaluated on held-out seed=7 (never seen during training)
- Reward: +1 for exit, −1/max_steps per step, potential-based shaping for dense signal

### Slide 4 — Single-Agent Baseline
- One agent on one maze: **100% success rate**, mean return +6.68 at 1,000 episodes
- DQN and the environment work correctly
- This is the gold standard — what we want FL to approach across multiple mazes

### Slide 5 — Naive FL Fails
- FedAvg (all layers): clients locally converge but global model collapses
  - Per-client success rate of global model: **100% / 0% / 0%** — seed=0 dominates
- **Why:** Q-values encode cumulative discounted reward over specific maze topology
  - Exit is 20 steps left in seed=0 → Q(left) ≈ +3
  - Exit is 60 steps right in seed=42 → Q(left) ≈ −2
  - Averaging these produces noise, not a policy
- Even **centralized training** (all seeds pooled into one agent) fails — 0% everywhere
  - This is the key insight: the problem is not FL overhead, it is Q-value incompatibility

### Slide 6 — Root Cause Diagram
```
Input image → [Conv1 → Conv2 → Conv3] → [FC1 → FC2] → Q-values
               ↑ topology-agnostic ↑      ↑ topology-specific ↑
                   safe to average            NOT safe to average

Corridor looks the same in any maze.     Q(left) = +3 in seed=0, −2 in seed=42.
```

### Slide 7 — Partial Parameter Sharing (Our Fix)
- Only average conv layers across clients; FC layers stay private
- Each client receives updated shared features, keeps its own Q-value head
- Implementation: 3 lines in `fl_server.py`, 4 lines in `dqn_agent.py`
- All 3 clients maintain ~100% success on their own mazes throughout training
- Client returns approach the single-agent baseline (+6.29, +5.81, +4.51 at round 35 vs +6.68 baseline)

### Slide 8 — Transfer Test Results
- Can shared conv features help a new agent adapt faster to seed=7?
- Method: load FL conv weights into fresh agent, fine-tune FC for 200 episodes, evaluate for 50 episodes

| Init | Success Rate | Return | Δ vs random |
|---|---|---|---|
| FL conv (FedAvg) | **100%** | +0.484 | **+84%** |
| FL conv (FedProx) | **100%** | +0.121 | +32% |
| Random init | 16–68% | varies | baseline |

- FL conv init: consistent 100% success
- Random init: inconsistent 16–68% (high variance, still exploring)

### Slide 9 — Why FedProx Underperforms FedAvg
- FedProx adds proximal penalty: `(µ/2) ||w − w_global||²` to each client's loss
- This anchors conv weights close to the global model — limits how much they can improve
- When the thing you're sharing (conv layers) is the thing the proximal term restricts, it hurts
- FedAvg with no constraint allows conv layers to adapt freely → better shared features

### Slide 10 — Full Comparison Table

| Method | Client SR | Held-out SR (200 eps fine-tune) |
|---|---|---|
| Single-agent baseline | 100% (seed=0 only) | — |
| Local-only | ~100% own maze | 0% |
| Centralized (pooled) | 0% | 0% |
| FedAvg — all layers | 100% / 0% / 0% | 0% |
| **FedAvg — partial conv** | **~100% all** | **100% (+84%)** |
| FedProx — partial conv | ~100% all | 100% (+32%) |

### Slide 11 — Conclusions
1. Naive FedAvg fails on heterogeneous DQN — same reason as centralized pooling
2. Partial parameter sharing (conv only) resolves Q-value incompatibility
3. Shared conv features enable fast, reliable adaptation to unseen mazes
4. FedProx is counterproductive here — proximal penalty limits the shared feature quality
5. The right decomposition matters: share what is environment-agnostic, keep what is environment-specific

### Slide 12 — Future Work
- Personalised FL: jointly train shared CNN trunk + per-client output heads (FedPer)
- FedMA: layer-wise weight matching before averaging (avoids permutation misalignment)
- Knowledge distillation FL: server distills client policies rather than averaging weights
- Gradient sharing instead of weight sharing (avoids Q-value scale incompatibility entirely)
- Test with more heterogeneous environments (different maze sizes, obstacle counts)

---

## Key Numbers to Memorise

- Baseline: **100% SR, +6.68 return** (seed=0, 1000 eps)
- FedAvg all layers: **100% / 0% / 0%** client SR, **0% held-out**
- FedAvg partial conv: **~100% all clients, 100% held-out, +84% vs random**
- Fine-tune budget: **200 episodes** (vs 1000 for full training)
- Transfer advantage: FL conv → 100% SR; random init → 16–68% SR

---

## Potential Questions and Answers

**Q: Why not just train longer in the centralized setting?**
A: The failure is not about data quantity — it is about the Q-function being asked to represent incompatible value scales simultaneously. More training would not resolve the gradient conflict.

**Q: Could you use a different RL algorithm that avoids this problem?**
A: Policy gradient methods (e.g. PPO, A3C) output action probabilities rather than absolute Q-values. Probabilities are more compatible across environments — this is a direction future work could explore.

**Q: How different are the mazes?**
A: Each is a randomly generated 21×21 DFS maze with a different seed. The topology (corridor layout, dead-ends, exit location) is entirely different. Agents can navigate their own maze to the exit but have no knowledge of other mazes' layouts.

**Q: Is the 30 fine-tune episode result in the summary file misleading?**
A: Yes. The inline evaluation used only 30 fine-tune episodes, which is below the warmup threshold for FC learning. The dedicated transfer test with 200 episodes is the accurate measurement.

**Q: Why does FedProx hurt?**
A: FedProx anchors all shared weights (here, conv layers) to the previous global model. For supervised FL this reduces harmful client drift. But here we *want* the conv layers to drift — they need to keep improving their feature extraction. The proximal penalty fights this.
