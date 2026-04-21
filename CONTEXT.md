# Project Context — Federated DQN Maze

Pick-up notes for resuming work. Last updated: April 21 2026.

---

## Status: COMPLETE

All experiments have been run. Results are in `results/`.

---

## What Works

- Single-agent DQN baseline: 100% success, mean_return=+6.68, seed=0, 1000 eps
  - checkpoint: `checkpoints/baseline_seed0_ep1000.pt`
- Partial conv sharing (FedAvg+EqW): all clients ~100% on own maze, 100% transfer to seed=7
  - checkpoint: `checkpoints/fl_fedavg_eqw_final.pt`

---

## Root Cause (confirmed)

Q-values encode cumulative discounted reward. Different maze topologies have
different BFS distances to exit → incompatible Q-value magnitudes across clients.
FedAvg was designed for supervised learning (shared label space). Raw Q-value
averaging does not work — and neither does centralized pooling for the same reason.

Conv layers (detect corridors, corners, dead-ends) ARE topology-agnostic → safe to average.
FC layers (map features → Q-values) are NOT → must stay per-client.

---

## Final Results

| Method | Client SR | Held-out SR (seed=7) | Notes |
|---|---|---|---|
| Single-agent baseline | 100% seed=0 | — | |
| Local-only DQN | ~100% own seed | 0% | lower bound |
| Centralized DQN | 0% | 0% | pooling fails too |
| FedAvg all layers | 100%/0%/0% | 0% | seed=0 dominates |
| FedAvg partial conv | ~100% all | 100%* | **main result** |
| FedProx µ=0.1 partial conv | ~100% all | 100%* | µ hurts slightly |

*200 fine-tune episodes on seed=7

Transfer test (200 fine-tune eps, 50 eval eps, seed=7):
- FedAvg partial conv:  100% SR, +0.484 return, Δ+84% vs random
- FedProx partial conv: 100% SR, +0.121 return, Δ+32% vs random
- Random init baseline: 16–68% SR (noisy due to short budget)

---

## Key Files

| File | Purpose |
|---|---|
| `federated/fl_server.py` | `_SHARED_KEY` filters conv.* only; `transfer_evaluate()` fine-tunes FC on new seed |
| `federated/fl_client.py` | `local_evaluate()` — evaluates client's own model |
| `agent/dqn_agent.py` | `load_global_weights()` — only loads conv.* keys |
| `train_federated.py` | Main FL loop, `--equal_weight`, `--mu` flags |
| `train_centralized.py` | Pooled upper bound |
| `train_local_only.py` | Isolated lower bound |
| `transfer_test.py` | FL conv vs random init fine-tune comparison |
| `run_all.sh` | Reproduces all experiments |
| `results/` | All summary .txt + transfer test logs |

---

## If Resuming

All experiments are done. To re-run everything from scratch:
```bash
bash run_all.sh
```

To re-run only the transfer tests (fastest, uses saved checkpoints):
```bash
python transfer_test.py --checkpoint checkpoints/fl_fedavg_eqw_final.pt --finetune 200 --eval 50
python transfer_test.py --checkpoint checkpoints/fl_fedprox_mu0.1_eqw_final.pt --finetune 200 --eval 50
```
