"""
Play the maze yourself using arrow keys or WASD.

Controls:
  Arrow keys / WASD  — move
  R                  — restart episode
  Q / Escape         — quit

A new random maze is generated each episode (use_random_seed=True).
Change SEED and set USE_RANDOM_SEED=False for the same maze every run.
"""

import pygame
from maze_env import MazeEnv
from maze_env.definitions import MazeAction

# --- Config ---
USE_RANDOM_SEED = True
SEED            = 42
CELL_PX         = 32      # pixels per grid cell

# Action mapping: pygame key → MazeAction
KEY_MAP = {
    pygame.K_UP:    MazeAction.Up,
    pygame.K_w:     MazeAction.Up,
    pygame.K_LEFT:  MazeAction.Left,
    pygame.K_a:     MazeAction.Left,
    pygame.K_DOWN:  MazeAction.Down,
    pygame.K_s:     MazeAction.Down,
    pygame.K_RIGHT: MazeAction.Right,
    pygame.K_d:     MazeAction.Right,
}


def main():
    env = MazeEnv(
        seed            = SEED,
        use_random_seed = USE_RANDOM_SEED,
        regenerate_on_reset = True,
        render_mode     = "human",
        cell_px         = CELL_PX,
        max_episode_steps = 500,
    )

    obs      = env.reset()
    env.render()   # initialise pygame display before the event loop
    done     = False
    total_r  = 0.0
    steps    = 0
    episode  = 1

    print("Controls: Arrow keys / WASD to move | R to restart | Q/Esc to quit")
    print(f"Episode {episode}")

    clock = pygame.time.Clock()

    running = True
    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    obs = env.reset()
                    env.render()
                    done    = False
                    total_r = 0.0
                    steps   = 0
                    episode += 1
                    print(f"\nEpisode {episode}")

                elif event.key in KEY_MAP and not done:
                    action = KEY_MAP[event.key]

        if action is not None and not done:
            obs, reward, done = env.step(int(action))
            env.render()
            total_r += reward
            steps   += 1

            # Update window title with live stats
            pygame.display.set_caption(
                f"Maze — Ep {episode} | Steps {steps} | Return {total_r:+.2f}"
            )

            if done:
                result = "SUCCESS! Reached exit" if total_r > 0 else \
                         "DEAD — hit obstacle" if reward == -1.0 else "TIMEOUT"
                print(f"  {result} | steps={steps} | return={total_r:+.3f}")
                print("  Press R to restart or Q to quit.")

        clock.tick(30)  # cap at 30 fps so the window stays responsive

    env.close()


if __name__ == "__main__":
    main()
