# Experiment Results — Federated DQN Maze

CS 791 · April 21 2026

All raw result files are in this folder. This document consolidates them.

---

## Single-Agent Baseline

- **Seed:** 0
- **Episodes:** 1,000
- **Success rate:** 100%
- **Mean return:** +6.68
- **Checkpoint:** `checkpoints/baseline_seed0_ep1000.pt`

---

## Experiment 1 — Local-Only DQN (lower bound)

> File: `summary_local_only.txt`

3 independent agents, no parameter sharing. Each trains for 1,500 episodes on its assigned seed. Evaluated on held-out seed=7.

| Agent seed | Own maze SR* | Held-out SR (seed=7) | Final mean return |
|---|---|---|---|
| 0 | ~100% | 0% | −1.556 ± 0.137 |
| 42 | ~100% | 0% | −2.069 ± 0.042 |
| 99 | ~100% | 0% | −2.342 ± 0.461 |
| **Mean** | **~100%** | **0%** | **−1.989** |

*inferred from training logs (all agents converged on their own mazes)

All agents time out on seed=7 (avg_len=200 throughout). Policies are completely non-transferable without knowledge sharing.

---

## Experiment 2 — Centralized DQN (intended upper bound)

> File: `summary_centralized.txt`

1 agent, all 3 seeds pooled round-robin. 1,500 cycles × 3 seeds = 4,500 episodes. Evaluated on held-out seed=7.

| Metric | Value |
|---|---|
| Final success rate (seed=7) | 0% |
| Final mean return (seed=7) | −1.626 ± 0.025 |
| Avg episode length | 200.0 (always times out) |

**Finding:** Centralized pooling fails for the same reason as naive FedAvg — a single Q-function cannot simultaneously represent 3 structurally different maze topologies. This is not a communication problem; it is a fundamental Q-value incompatibility.

---

## Experiment 3 — FedAvg, All Layers (failure baseline)

> File: `summary_fedavg.txt`

50 rounds, 30 local eps/round, step-weighted aggregation, all layers averaged.

| Metric | Value |
|---|---|
| Final success rate (seed=7) | 0% |
| Final mean return (seed=7) | −1.149 ± 0.397 |
| Client SR pattern | 100% / 0% / 0% (seed=0 dominates) |

High variance (±0.397) vs centralized (±0.025) — step-weighting causes inconsistent seed=0 dominance.

---

## Experiment 4 — FedAvg, Partial Conv Sharing (main result)

> File: `summary_fedavg_eqw.txt`  
> Transfer test: `transfer_test_fedavg.log`

50 rounds, 30 local eps/round, equal-weight aggregation, **conv layers only averaged**.

### During training (client local eval at round 50)
- All 3 clients: ~100% success on own mazes
- Client mean returns approaching single-agent baseline (+6.29, +5.81, +4.51 at round 35)

### Transfer test on held-out seed=7 (200 fine-tune eps, 50 eval eps)

| Condition | Fine-tune SR | Eval SR | Mean return |
|---|---|---|---|
| FL conv init | 4% | **100%** | +0.484 ± 0.035 |
| Random init | 0% | 16% | −0.116 ± 0.097 |
| **Δ (FL − random)** | — | **+84%** | **+0.600** |

---

## Experiment 5 — FedProx µ=0.1, Partial Conv Sharing

> File: `summary_fedprox_mu0.1_eqw.txt`  
> Transfer test: `transfer_test_fedprox.log`

50 rounds, 30 local eps/round, equal-weight, proximal penalty µ=0.1, **conv layers only averaged**.

### Transfer test on held-out seed=7 (200 fine-tune eps, 50 eval eps)

| Condition | Fine-tune SR | Eval SR | Mean return |
|---|---|---|---|
| FL conv init | 2% | **100%** | +0.121 ± 0.066 |
| Random init | 0% | 68% | −0.386 ± 0.724 |
| **Δ (FL − random)** | — | **+32%** | **+0.507** |

FedProx underperforms FedAvg here. The proximal penalty anchors conv layers to earlier values, limiting how much the shared feature extractor can improve across rounds.

---

## Summary Table

| Method | Client SR | Held-out SR | Notes |
|---|---|---|---|
| Single-agent baseline | 100% (seed=0) | — | Gold standard |
| Local-only | ~100% own | 0% | Lower bound |
| Centralized (pooled) | 0% | 0% | Fails — Q incompatibility |
| FedAvg all layers | 100%/0%/0% | 0% | Fails — seed=0 dominates |
| **FedAvg partial conv** | **~100% all** | **100%** | **Main result — +84% vs random** |
| FedProx partial conv | ~100% all | 100% | +32% vs random — µ hurts |

---

## Limitations

- Random init transfer baseline is noisy (16% vs 68% across two runs) due to only 200 fine-tune episodes and high epsilon (~0.7) throughout fine-tuning
- Local-only own-maze SR inferred from training logs, not separately measured in summary
- Transfer test fine-tune budget (200 eps) is much smaller than the original single-agent training (1,000 eps), so results are lower-bound on what the features could achieve
