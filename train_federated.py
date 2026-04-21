"""
Federated DQN training — FedAvg and FedProx.

Usage
-----
    python3 train_federated.py           # FedAvg  (mu=0)
    python3 train_federated.py --mu 0.1  # FedProx (mu=0.1)

Design
------
- 3 clients train on maze seeds {0, 42, 99} (different layouts = non-IID).
- Each round: server broadcasts global weights → clients train locally for
  LOCAL_EPISODES episodes → server aggregates via FedAvg.
- FedProx adds a proximal penalty (mu/2)*||w - w_global||^2 to each
  client's loss, reducing client drift.
- The global model is evaluated on held-out seed=7 every EVAL_EVERY rounds.

Output
------
  checkpoints/fl_fedavg_final.pt          (or fl_fedprox_mu0.1_final.pt)
  Console summary block — paste back for comparison.
"""

import argparse
import os

import numpy as np
import torch

from federated import FLClient, FLServer

# ---------------------------------------------------------------------------
# Config
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

# FL schedule
# Each round = LOCAL_EPISODES episodes of local training per client.
# With ~200 steps/episode on a 21×21 maze, 50 episodes ≈ 10k steps per round.
# This gives each client's Q-network enough signal to be meaningful before
# aggregation — the key difference from training with only 5–10 eps/round
# (which causes Q-values to diverge before they can learn anything useful).
FL_ROUNDS         = 50
LOCAL_EPISODES    = 30    # episodes per client per round
EVAL_EVERY        = 5     # evaluate every N rounds
EVAL_EPISODES     = 20
# Fine-tune episodes for transfer eval on held-out seed=7.
# Warmup=2000 steps; ~200 steps/ep → 10 eps to warm up, then FC starts training.
FINETUNE_EPISODES = 30
EVAL_EPS       = 0.05
SAVE_EVERY     = 10

# Agent hypers — all training from random init
LR             = 1e-4
GAMMA          = 0.99
BATCH_SIZE     = 32
# Each transition = 2×(5,40,40) float32 = 64 KB.
# 3 clients × 15k × 64 KB ≈ 2.9 GB — safe for most machines.
# (Single-agent baseline used 200k, but FL clients only see ~10k steps/round.)
BUFFER_CAP     = 15_000
WARMUP_STEPS   = 2_000
TRAIN_FREQ     = 4
TARGET_UPDATE  = 500
EPS_START      = 1.0
EPS_END        = 0.1
# Decay over ~20 rounds of local training so clients are mostly greedy
# in the second half of FL but still exploratory early on.
# 30 eps × 200 steps × 20 rounds = 120,000 steps
EPS_DECAY      = 120_000


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mu", type=float, default=0.0,
        help="FedProx proximal coefficient. 0.0 = FedAvg (default).",
    )
    p.add_argument(
        "--equal_weight", action="store_true",
        help="Average client weights equally instead of weighting by steps taken. "
             "Prevents fast-learning clients from dominating aggregation.",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    mu     = args.mu
    eq     = args.equal_weight
    method = (f"FedProx(mu={mu})" if mu > 0 else "FedAvg") + ("+EqW" if eq else "")
    tag    = (f"fedprox_mu{mu}" if mu > 0 else "fedavg") + ("_eqw" if eq else "")

    print("=" * 68)
    print(f"  Federated DQN — {method}")
    print(f"  Clients: seeds {CLIENT_SEEDS}   Held-out: seed {HELD_OUT_SEED}")
    print(f"  Rounds: {FL_ROUNDS}   Local eps/round: {LOCAL_EPISODES}   Device: {DEVICE}")
    print("=" * 68)

    os.makedirs("checkpoints", exist_ok=True)

    # Shared kwargs
    env_kwargs = dict(
        width=MAZE_W, height=MAZE_H,
        obstacle_count=OBSTACLE_COUNT, shaping_scale=SHAPING_SCALE,
        n_stack=N_STACK, cell_obs_px=CELL_OBS_PX,
        max_episode_steps=MAX_EP_STEPS,
    )
    agent_kwargs = dict(
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAP, target_update_freq=TARGET_UPDATE,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay_steps=EPS_DECAY,
        warmup_steps=WARMUP_STEPS, train_freq=TRAIN_FREQ,
    )

    # Derive obs_shape from a throwaway env
    from maze_env import MazeEnv
    _tmp      = MazeEnv(seed=0, **env_kwargs)
    obs_shape = _tmp.obs_size
    _tmp.close()
    print(f"  obs_shape={obs_shape}\n")

    # Server
    server = FLServer(obs_shape=obs_shape, n_actions=N_ACTIONS, device=DEVICE)

    # Clients
    clients = [
        FLClient(
            client_id=i, seed=seed,
            obs_shape=obs_shape, n_actions=N_ACTIONS, device=DEVICE,
            agent_kwargs=agent_kwargs, env_kwargs=env_kwargs,
        )
        for i, seed in enumerate(CLIENT_SEEDS)
    ]

    eval_history = []   # (round, mean_return, success_rate)

    # Print header — client avgs + held-out eval + one column per client seed
    client_seed_hdr = "  ".join(f"sr{s}" for s in CLIENT_SEEDS)
    hdr = (f"{'Rnd':>4}  "
           + "  ".join(f"s{s}avg" for s in CLIENT_SEEDS)
           + f"  {'XferMean':>10}  {'HeldSR':>6}  {client_seed_hdr}")
    print(hdr)
    print("-" * len(hdr))

    for rnd in range(1, FL_ROUNDS + 1):
        global_weights = server.get_weights()

        # --- Local training ---
        client_results = []
        client_avgs    = []
        for client in clients:
            weights, steps, ep_returns = client.train_round(
                global_weights, LOCAL_EPISODES, proximal_mu=mu,
            )
            client_results.append((weights, steps))
            client_avgs.append(
                np.mean(ep_returns) if ep_returns else float("nan")
            )

        # --- Aggregation ---
        server.fedavg(client_results, equal_weight=eq)

        # --- Evaluation ---
        eval_str = ""
        if rnd % EVAL_EVERY == 0:
            # Held-out seed — fine-tune FC on seed=7 with shared conv features
            stats = server.transfer_evaluate(
                seed=HELD_OUT_SEED, n_finetune=FINETUNE_EPISODES,
                n_eval=EVAL_EPISODES, env_kwargs=env_kwargs,
                agent_kwargs=agent_kwargs, eval_eps=EVAL_EPS,
            )
            eval_history.append((rnd, stats["mean_return"], stats["success_rate"]))
            eval_str = f"{stats['mean_return']:>+10.3f}  {stats['success_rate']:>5.0%}"

            # Evaluate each client's local model (shared conv + per-client FC)
            client_sr = []
            for client in clients:
                cs = client.local_evaluate(n_episodes=10, eval_eps=EVAL_EPS)
                client_sr.append(f"{cs['success_rate']:>4.0%}")
            eval_str += "  " + "  ".join(client_sr)

            if rnd % SAVE_EVERY == 0:
                server.save(f"checkpoints/fl_{tag}_round{rnd}.pt")

        avg_str = "  ".join(f"{v:+.2f}" for v in client_avgs)
        print(f"{rnd:>4}  {avg_str}  {eval_str}")

    # Final save + 50-episode eval
    final_path = f"checkpoints/fl_{tag}_final.pt"
    server.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    final = server.transfer_evaluate(
        seed=HELD_OUT_SEED, n_finetune=FINETUNE_EPISODES, n_eval=50,
        env_kwargs=env_kwargs, agent_kwargs=agent_kwargs, eval_eps=EVAL_EPS,
    )

    # Cleanup
    for c in clients:
        c.close()

    # -------------------------------------------------------------------
    summary_lines = [
        "=" * 68,
        "SUMMARY",
        "=" * 68,
        f"method={method}  mu={mu}",
        f"rounds={FL_ROUNDS}  local_eps={LOCAL_EPISODES}  "
        f"clients={CLIENT_SEEDS}  held_out={HELD_OUT_SEED}",
        "",
        "Transfer eval history (conv init → fine-tune FC → eval) on held-out seed:",
    ]
    for rnd, mr, sr in eval_history:
        bar = "#" * int(sr * 20)
        summary_lines.append(
            f"  round={rnd:>3}  mean_return={mr:+.3f}  success={sr:.0%}  [{bar:<20}]"
        )
    summary_lines += [
        "",
        f"Final 50-ep eval:  mean_return={final['mean_return']:+.3f}  "
        f"±{final['std_return']:.3f}  "
        f"success_rate={final['success_rate']:.0%}",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    print("\n" + summary_text)

    summary_path = f"checkpoints/summary_{tag}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
