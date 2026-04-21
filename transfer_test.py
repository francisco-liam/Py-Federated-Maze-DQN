"""
Transfer test: does loading FL conv weights help on held-out seed=7?

Compares two conditions over FINETUNE_EPISODES episodes:
  A) FL init   — conv layers from fl_fedavg_eqw_final.pt, FC random
  B) Random    — all weights random

Both use identical hyperparameters and the same env seed.
Prints per-episode success and a summary comparison.
"""

import argparse
import numpy as np
import torch

from agent import DQNAgent
from maze_env import MazeEnv

# ── Config (match train_federated.py) ──────────────────────────────────────
HELD_OUT_SEED    = 7
FINETUNE_EPISODES = 200
EVAL_EPISODES    = 50
EVAL_EPS         = 0.05

MAZE_W, MAZE_H   = 21, 21
OBSTACLE_COUNT   = 0
SHAPING_SCALE    = 0.05
N_STACK          = 4
CELL_OBS_PX      = 8
MAX_EP_STEPS     = 200
N_ACTIONS        = 5

LR               = 1e-4
GAMMA            = 0.99
BATCH_SIZE       = 32
BUFFER_CAP       = 15_000
WARMUP_STEPS     = 2_000
TRAIN_FREQ       = 4
TARGET_UPDATE    = 500
EPS_START        = 1.0
EPS_END          = 0.1
EPS_DECAY        = 120_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_agent(obs_shape):
    return DQNAgent(
        obs_shape=obs_shape, n_actions=N_ACTIONS,
        device=DEVICE, **agent_kwargs,
    )


def run_condition(label, agent, env_seed, n_finetune, n_eval):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Fine-tuning for {n_finetune} episodes on seed={env_seed}...")

    env = MazeEnv(seed=env_seed, **env_kwargs)
    obs = env.reset()
    ep_return, eps_done = 0.0, 0
    successes_ft = 0

    while eps_done < n_finetune:
        action = agent.select_action(obs)
        next_obs, reward, done = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.maybe_train()
        ep_return += reward
        obs = next_obs
        if done:
            success = ep_return > 0
            successes_ft += int(success)
            print(
                f"  [{label}] ep {eps_done+1:>3}/{n_finetune}"
                f"  return={ep_return:+.2f}"
                f"  eps={agent.epsilon():.3f}"
                f"  {'SUCCESS' if success else ''}",
                flush=True,
            )
            ep_return = 0.0
            eps_done += 1

    ft_sr = successes_ft / n_finetune
    print(f"\n  Fine-tune success rate: {ft_sr:.0%}")

    # ── Eval ────────────────────────────────────────────────────────────
    print(f"  Evaluating for {n_eval} episodes...")
    returns, successes = [], 0
    with torch.no_grad():
        for _ in range(n_eval):
            obs = env.reset()
            ep_r, done = 0.0, False
            while not done:
                if np.random.rand() < EVAL_EPS:
                    action = np.random.randint(N_ACTIONS)
                else:
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32,
                        device=torch.device(DEVICE)
                    ).unsqueeze(0)
                    action = int(agent.online_net(obs_t).argmax(dim=1).item())
                obs, r, done = env.step(action)
                ep_r += r
            returns.append(ep_r)
            if ep_r > 0:
                successes += 1

    env.close()
    return dict(
        label        = label,
        ft_sr        = ft_sr,
        mean_return  = float(np.mean(returns)),
        std_return   = float(np.std(returns)),
        success_rate = successes / n_eval,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default="checkpoints/fl_fedavg_eqw_final.pt",
        help="FL model checkpoint to load conv weights from.",
    )
    p.add_argument(
        "--finetune", type=int, default=FINETUNE_EPISODES,
        help="Fine-tune episodes per condition.",
    )
    p.add_argument(
        "--eval", type=int, default=EVAL_EPISODES,
        help="Eval episodes per condition.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    _tmp      = MazeEnv(seed=0, **env_kwargs)
    obs_shape = _tmp.obs_size
    _tmp.close()

    # ── Condition A: FL conv init ────────────────────────────────────────
    agent_fl = make_agent(obs_shape)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    # load_global_weights only copies conv.* keys
    agent_fl.load_global_weights(ckpt)

    result_fl = run_condition(
        label     = f"FL conv init ({args.checkpoint})",
        agent     = agent_fl,
        env_seed  = HELD_OUT_SEED,
        n_finetune= args.finetune,
        n_eval    = args.eval,
    )

    # ── Condition B: Random init ─────────────────────────────────────────
    agent_rand = make_agent(obs_shape)

    result_rand = run_condition(
        label     = "Random init",
        agent     = agent_rand,
        env_seed  = HELD_OUT_SEED,
        n_finetune= args.finetune,
        n_eval    = args.eval,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRANSFER TEST SUMMARY")
    print(f"  seed={HELD_OUT_SEED}  fine-tune={args.finetune} eps  eval={args.eval} eps")
    print(f"{'='*60}")
    for r in [result_fl, result_rand]:
        print(
            f"  {r['label']}\n"
            f"    fine-tune SR : {r['ft_sr']:.0%}\n"
            f"    eval SR      : {r['success_rate']:.0%}\n"
            f"    eval return  : {r['mean_return']:+.3f} ± {r['std_return']:.3f}\n"
        )

    delta_sr = result_fl["success_rate"] - result_rand["success_rate"]
    delta_ret = result_fl["mean_return"] - result_rand["mean_return"]
    print(f"  Δ success_rate (FL − random): {delta_sr:+.0%}")
    print(f"  Δ mean_return  (FL − random): {delta_ret:+.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
