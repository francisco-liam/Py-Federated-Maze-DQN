"""
Generate demo.mp4 — 20-second split-screen comparison:
  Left:  random agent stumbling around seed=7
  Right: FL conv pre-trained agent solving seed=7 quickly

Usage:
  python make_demo.py [--checkpoint checkpoints/fl_fedavg_eqw_final.pt]
                      [--finetune 150]
                      [--fps 30]
                      [--output demo.mp4]

No display required — uses pygame in dummy (headless) mode.
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── Headless pygame ──────────────────────────────────────────────────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame

# ── Project imports ──────────────────────────────────────────────────────────
from agent import DQNAgent
from maze_env import MazeEnv
from maze_env.definitions import MazeCellType

# ── Config ───────────────────────────────────────────────────────────────────
HELD_OUT_SEED  = 7
MAZE_W, MAZE_H = 21, 21
OBSTACLE_COUNT = 0
SHAPING_SCALE  = 0.05
N_STACK        = 4
CELL_OBS_PX    = 8
MAX_EP_STEPS   = 200
N_ACTIONS      = 5
CELL_PX        = 20          # pixels per cell in the video

LR             = 1e-4
GAMMA          = 0.99
BATCH_SIZE     = 32
BUFFER_CAP     = 15_000
WARMUP_STEPS   = 500     # fast warmup so fine-tune is effective in 200 eps
TRAIN_FREQ     = 2
TARGET_UPDATE  = 200
EPS_DECAY      = 15_000  # fast decay so greedy policy is meaningful after 200 eps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ENV_KW = dict(
    width=MAZE_W, height=MAZE_H,
    obstacle_count=OBSTACLE_COUNT, shaping_scale=SHAPING_SCALE,
    n_stack=N_STACK, cell_obs_px=CELL_OBS_PX,
    max_episode_steps=MAX_EP_STEPS,
)
_AGENT_KW = dict(
    lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
    buffer_capacity=BUFFER_CAP, target_update_freq=TARGET_UPDATE,
    eps_start=1.0, eps_end=0.05, eps_decay_steps=EPS_DECAY,
    warmup_steps=WARMUP_STEPS, train_freq=TRAIN_FREQ,
)

# Colours
C_WALL     = ( 60,  60,  60)
C_FLOOR    = (200, 200, 200)
C_START    = (100, 200, 100)
C_EXIT     = (100, 150, 255)
C_OBSTACLE = (230, 140,  30)
C_SCRATCH  = (220,  50,  50)   # red player dot — scratch agent
C_FL       = ( 50, 180,  50)   # green player dot — FL agent
C_BLACK    = (  0,   0,   0)
C_WHITE    = (255, 255, 255)
C_BANNER_L = ( 80,  20,  20)   # dark red title bar
C_BANNER_R = ( 20,  60,  20)   # dark green title bar
C_SUCCESS  = ( 50, 200,  50)
C_FAIL     = (200,  50,  50)

BANNER_H = 30   # pixels for title bar at top of each pane

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_agent(obs_shape) -> DQNAgent:
    return DQNAgent(obs_shape=obs_shape, n_actions=N_ACTIONS, device=DEVICE, **_AGENT_KW)


def finetune(agent: DQNAgent, n_episodes: int) -> None:
    """Train agent on held-out seed for n_episodes (headless)."""
    env = MazeEnv(seed=HELD_OUT_SEED, **_ENV_KW)
    obs = env.reset()
    ep_return, eps_done, successes = 0.0, 0, 0
    while eps_done < n_episodes:
        action = agent.select_action(obs)
        next_obs, reward, done = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        agent.maybe_train()
        ep_return += reward
        obs = next_obs
        if done:
            if ep_return > 0:
                successes += 1
            eps_done += 1
            ep_return = 0.0
    env.close()
    print(f"  Fine-tune done ({n_episodes} eps, {successes} successes). eps={agent.epsilon():.3f}")


def draw_env(surface: pygame.Surface, env: MazeEnv, player_color,
             y_offset: int = 0, override_pos=None) -> None:
    """Draw maze state onto surface at y_offset. override_pos overrides player dot position."""
    builder = env._builder
    W, H = builder.width, builder.height

    for (x, y), cell_type in builder.cells.items():
        sx = x * CELL_PX
        sy = y_offset + (H - 1 - y) * CELL_PX
        rect = pygame.Rect(sx, sy, CELL_PX, CELL_PX)
        if cell_type == MazeCellType.Wall:
            color = C_WALL
        elif cell_type == MazeCellType.Start:
            color = C_START
        elif cell_type == MazeCellType.Exit:
            color = C_EXIT
        else:
            color = C_FLOOR
        pygame.draw.rect(surface, color, rect)

    px, py = override_pos if override_pos is not None else env._player_pos
    sx = px * CELL_PX
    sy = y_offset + (H - 1 - py) * CELL_PX
    inset = max(2, CELL_PX // 6)
    pygame.draw.rect(surface, player_color,
                     pygame.Rect(sx + inset, sy + inset, CELL_PX - 2 * inset, CELL_PX - 2 * inset))


def draw_pane(surface: pygame.Surface, env: MazeEnv,
              player_color, label: str, banner_color,
              x_off: int, pane_w: int, font: pygame.font.Font,
              overlay_text: str = "", overlay_color=C_WHITE,
              override_pos=None) -> None:
    """Render a single pane (banner + maze + optional overlay) into a sub-view."""
    sub = surface.subsurface(pygame.Rect(x_off, 0, pane_w, surface.get_height()))

    # Banner
    banner_rect = pygame.Rect(0, 0, pane_w, BANNER_H)
    pygame.draw.rect(sub, banner_color, banner_rect)
    label_surf = font.render(label, True, C_WHITE)
    sub.blit(label_surf, label_surf.get_rect(center=banner_rect.center))

    # Maze
    sub.fill(C_WALL, pygame.Rect(0, BANNER_H, pane_w, sub.get_height() - BANNER_H))
    draw_env(sub, env, player_color, y_offset=BANNER_H, override_pos=override_pos)

    # Overlay (SUCCESS / TIMEOUT)
    if overlay_text:
        ov_surf = font.render(overlay_text, True, overlay_color)
        ov_rect = ov_surf.get_rect(center=(pane_w // 2, BANNER_H + (sub.get_height() - BANNER_H) // 2))
        # semi-transparent background
        bg = pygame.Surface((ov_rect.width + 20, ov_rect.height + 10))
        bg.set_alpha(180)
        bg.fill(C_BLACK)
        sub.blit(bg, (ov_rect.x - 10, ov_rect.y - 5))
        sub.blit(ov_surf, ov_rect)


def grab_frame(surface: pygame.Surface) -> np.ndarray:
    """Return (H, W, 3) uint8 numpy array from a pygame Surface."""
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2))   # (W,H,3) → (H,W,3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/fl_fedavg_eqw_final.pt",
                    help="Checkpoint used for both agents (same training run)")
    ap.add_argument("--finetune", type=int, default=200, dest="finetune",
                    help="Fine-tune episodes for the partial-conv agent")
    ap.add_argument("--episodes",   type=int, default=8,
                    help="Stop after FL agent completes this many episodes")
    ap.add_argument("--fps",        type=int, default=30)
    ap.add_argument("--speed",      type=int, default=2,
                    help="Env steps per rendered frame (higher = faster playback)")
    ap.add_argument("--output",     default="demo.mp4")
    args = ap.parse_args()

    try:
        import imageio
    except ImportError:
        sys.exit("imageio not found — run: pip install imageio[ffmpeg]")

    total_frames = None  # episode-count based; not used
    pane_w = MAZE_W * CELL_PX
    pane_h = MAZE_H * CELL_PX + BANNER_H
    total_w = pane_w * 2

    # ── Build env to get obs_shape ────────────────────────────────────────────
    _tmp = MazeEnv(seed=HELD_OUT_SEED, **_ENV_KW)
    obs_shape = _tmp.obs_size
    _tmp.close()
    print(f"obs_shape={obs_shape}  device={DEVICE}")

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # ── Left: Naive FedAvg — full global state dict, no fine-tuning ───────────
    # The global FC was never trained on seed=7 (server FC stays at random init;
    # only conv layers were averaged). Running this greedy demonstrates failure.
    naive_agent = make_agent(obs_shape)
    naive_agent.online_net.load_state_dict(ckpt)
    naive_agent.target_net.load_state_dict(ckpt)
    print("Naive FedAvg agent: full state dict loaded, no fine-tuning.")

    # ── Right: Partial conv — only conv transferred, FC fine-tuned ───────────
    fl_agent = make_agent(obs_shape)
    fl_agent.load_global_weights(ckpt)   # loads conv.* only
    if args.finetune > 0:
        print(f"Fine-tuning partial-conv agent for {args.finetune} episodes on seed={HELD_OUT_SEED}...")
        finetune(fl_agent, args.finetune)

    # ── Pygame setup ──────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 15, bold=True)
    surface = pygame.Surface((total_w, pane_h))

    # ── Envs — agents train AND render at the same time ───────────────────────
    env_naive = MazeEnv(seed=HELD_OUT_SEED, regenerate_on_reset=False, **_ENV_KW)
    env_fl    = MazeEnv(seed=HELD_OUT_SEED, regenerate_on_reset=False, **_ENV_KW)
    obs_naive = env_naive.reset()
    obs_fl    = env_fl.reset()

    # Episode tracking
    naive_episodes, naive_success = 0, 0
    fl_episodes,    fl_success    = 0, 0
    naive_ret, fl_ret = 0.0, 0.0
    overlay_naive, overlay_fl = "", ""
    overlay_timer_naive, overlay_timer_fl = 0, 0
    freeze_naive, freeze_fl = 0, 0           # frames to hold current position
    OVERLAY_FRAMES = int(args.fps * 1.5)     # show result banner for 1.5 s
    FREEZE_FRAMES  = int(args.fps * 1.0)     # freeze env for 1 s on episode end

    frames = []
    frame_idx = 0
    print(f"Rendering until FL agent completes {args.episodes} episodes @ {args.fps}fps speed={args.speed}...")

    while fl_episodes < args.episodes:
        # ── Run steps — greedy policies, no further training ──────────────────
        for _ in range(args.speed):
            # Naive FedAvg — full global state dict, no fine-tuning
            if freeze_naive <= 0:
                action_naive = naive_agent.select_greedy_action(obs_naive)
                next_naive, r_naive, done_naive = env_naive.step(action_naive)
                naive_ret += r_naive
                obs_naive = next_naive
                if done_naive:
                    naive_episodes += 1
                    if naive_ret > 0:
                        naive_success += 1
                        overlay_naive = "SUCCESS!"
                    else:
                        overlay_naive = "TIMEOUT"
                    overlay_timer_naive = OVERLAY_FRAMES
                    freeze_naive = FREEZE_FRAMES
                    naive_ret = 0.0

            # Partial conv FL — conv shared, FC fine-tuned on seed=7
            if freeze_fl <= 0:
                action_fl = fl_agent.select_greedy_action(obs_fl)
                next_fl, r_fl, done_fl = env_fl.step(action_fl)
                fl_ret += r_fl
                obs_fl = next_fl
                if done_fl:
                    fl_episodes += 1
                    if fl_ret > 0:
                        fl_success += 1
                        overlay_fl = "SUCCESS!"
                    else:
                        overlay_fl = "TIMEOUT"
                    overlay_timer_fl = OVERLAY_FRAMES
                    freeze_fl = FREEZE_FRAMES
                    fl_ret = 0.0

        # ── Decrement freeze and overlay timers ───────────────────────────────
        if freeze_naive > 0:
            freeze_naive -= 1
        if freeze_fl > 0:
            freeze_fl -= 1
        if overlay_timer_naive > 0:
            overlay_timer_naive -= 1
        else:
            overlay_naive = ""
        if overlay_timer_fl > 0:
            overlay_timer_fl -= 1
        else:
            overlay_fl = ""

        # ── Draw ──────────────────────────────────────────────────────────────
        naive_sr = naive_success / max(1, naive_episodes)
        fl_sr    = fl_success    / max(1, fl_episodes)

        naive_label = f"Naive FedAvg (avg ALL layers)  SR={naive_sr:.0%}"
        fl_label    = f"Partial Conv FL ({args.finetune}ep tune)  SR={fl_sr:.0%}"

        naive_ov_color = C_SUCCESS if "SUCCESS" in overlay_naive else C_FAIL
        fl_ov_color    = C_SUCCESS if "SUCCESS" in overlay_fl    else C_FAIL
        # Show player at exit during SUCCESS freeze, otherwise at current pos
        naive_override = env_naive._builder.exit_cell if overlay_naive == "SUCCESS!" else None
        fl_override    = env_fl._builder.exit_cell    if overlay_fl   == "SUCCESS!" else None

        draw_pane(surface, env_naive, C_SCRATCH, naive_label, C_BANNER_L,
                  0, pane_w, font, overlay_naive, naive_ov_color, naive_override)
        draw_pane(surface, env_fl,   C_FL,       fl_label,    C_BANNER_R,
                  pane_w, pane_w, font, overlay_fl, fl_ov_color, fl_override)

        frames.append(grab_frame(surface))

        if frame_idx % (args.fps * 5) == 0:
            print(f"  frame {frame_idx}  "
                  f"naive SR={naive_sr:.0%} ({naive_episodes} eps)  "
                  f"fl SR={fl_sr:.0%} ({fl_episodes}/{args.episodes} eps)")
        frame_idx += 1

    env_naive.close()
    env_fl.close()
    pygame.quit()

    # ── Write video ───────────────────────────────────────────────────────────
    print(f"Writing {len(frames)} frames to {args.output} ...")
    writer = imageio.get_writer(args.output, fps=args.fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Done — {args.output}")
    print(f"\nFinal SR:  naive={naive_success}/{naive_episodes}  partial_conv={fl_success}/{fl_episodes}")


if __name__ == "__main__":
    main()
