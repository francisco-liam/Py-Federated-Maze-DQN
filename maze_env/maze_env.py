"""
Pure-Python maze RL environment with optional Pygame rendering.

Drop-in replacement for UnityMazeEnv (unity_env_wrapper.py).
Identical public interface:
  obs              = env.reset()
  obs, reward, done = env.step(action)
  env.close()

When done=True, step() auto-resets internally and returns the FIRST
observation of the new episode as obs — matching the ML-Agents Unity
wrapper's behaviour.  This means the training loop in train_dqn.py
requires no changes.

Observation vector (26 floats) matches MazeAgent.CollectObservations:
  [0–24]  5×5 egocentric local grid (row-major, north-to-south / west-to-east)
          0.00 = floor  |  0.50 = exit  |  0.75 = obstacle  |  1.00 = wall/OOB
  [25]    Remaining step fraction (1.0 = episode start → 0.0 = timeout)

Action mapping (matches MazeAction enum):
  0 = NoOp  |  1 = Up  |  2 = Left  |  3 = Down  |  4 = Right
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .definitions import MazeAction, MazeCellType, MazeTerminalReason
from .maze_builder import MazeBuilder
from .moving_obstacle import MovingObstacle

Pos = Tuple[int, int]

# ------------------------------------------------------------------
# Observation constants (match MazeAgent.cs)
# ------------------------------------------------------------------
_HALF_VIEW = 2
_VIEW_SIZE = 2 * _HALF_VIEW + 1   # 5

# Greyscale pixel values for the egocentric observation frame [0.0, 1.0]
_PX_OCCLUDED = 0.00   # cell hidden behind a wall
_PX_WALL     = 0.25   # visible wall
_PX_FLOOR    = 0.70   # open floor
_PX_OBSTACLE = 0.55   # moving hazard
_PX_EXIT     = 0.85   # goal cell
_PX_PLAYER   = 1.00   # agent's own cell


def _bresenham_between(x0: int, y0: int, x1: int, y1: int) -> list:
    """
    Return a list of grid cells strictly between (x0, y0) and (x1, y1)
    following Bresenham's line algorithm.  Neither endpoint is included.
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if (x, y) == (x1, y1):
            break
        cells.append((x, y))
    return cells


# Action → (dx, dy) deltas in grid space (y-up)
_DELTAS: dict = {
    MazeAction.NoOp:  ( 0,  0),
    MazeAction.Up:    ( 0,  1),
    MazeAction.Left:  (-1,  0),
    MazeAction.Down:  ( 0, -1),
    MazeAction.Right: ( 1,  0),
}

# ------------------------------------------------------------------
# Pygame colour palette
# ------------------------------------------------------------------
_COLOR_WALL     = ( 60,  60,  60)
_COLOR_FLOOR    = (200, 200, 200)
_COLOR_START    = (100, 200, 100)
_COLOR_EXIT     = (100, 150, 255)
_COLOR_PLAYER   = (220,  50,  50)
_COLOR_OBSTACLE = (230, 140,  30)


class MazeEnv:
    """
    Gym-style maze environment implemented entirely in Python.

    Parameters
    ----------
    width, height : int
        Target maze dimensions (forced to odd, ≥ 5).
    use_procedural : bool
        True = DFS procedural generation.  False = hardcoded fallback layout.
    seed : int
        Fixed seed for deterministic generation.  Ignored when use_random_seed=True.
    use_random_seed : bool
        Pick a fresh seed on each generation attempt.
    max_generation_attempts : int
        How many times to retry generation before using the fallback layout.
    regenerate_on_reset : bool
        When True, rebuild the maze (new layout + new obstacles) at each episode reset.
    obstacle_count : int
        Number of patrol obstacles to place.
    min_patrol_distance : int
        Minimum Manhattan length for a valid patrol segment.
    avoid_start_exit_for_obstacles : bool
        Skip segments whose endpoints touch the start or exit cell.
    allow_fewer_obstacles : bool
        Accept the maze if fewer than obstacle_count could be placed.
    min_start_exit_path_distance : int
        Minimum BFS path length from start to exit.
    min_safe_reachable_cells : int
        Minimum number of cells reachable from start when obstacle patrols are blocked.
    max_episode_steps : int
        Hard timeout per episode (0 = no limit).
    exit_reward : float
        Reward granted on reaching the exit.
    death_penalty : float
        Reward (negative) granted on obstacle collision.
    step_penalty : float
        Reward (negative) applied every step.
    timeout_penalty : float
        Reward (negative) applied on timeout termination.
    render_mode : str or None
        None = headless.  "human" = Pygame window.
    cell_px : int
        Pixel size of each grid cell when rendering.
    """

    n_actions = 5

    def __init__(
        self,
        width:                          int   = 21,
        height:                         int   = 21,
        use_procedural:                 bool  = True,
        seed:                           int   = 12345,
        use_random_seed:                bool  = False,
        max_generation_attempts:        int   = 30,
        regenerate_on_reset:            bool  = False,
        obstacle_count:                 int   = 3,
        min_patrol_distance:            int   = 2,
        avoid_start_exit_for_obstacles: bool  = True,
        allow_fewer_obstacles:          bool  = False,
        min_start_exit_path_distance:   int   = 8,
        min_safe_reachable_cells:       int   = 8,
        max_episode_steps:              int   = 200,
        exit_reward:                    float = 1.0,
        death_penalty:                  float = -1.0,
        step_penalty:                   float = -0.01,
        timeout_penalty:                float = -0.5,
        shaping_scale:                  float = 0.05,
        n_stack:                        int   = 4,
        cell_obs_px:                    int   = 8,
        render_mode:                    Optional[str] = None,
        cell_px:                        int   = 32,
    ) -> None:
        # Reward / episode config
        self._regenerate_on_reset = regenerate_on_reset
        self._max_episode_steps   = max_episode_steps
        self._exit_reward         = exit_reward
        self._death_penalty       = death_penalty
        self._step_penalty        = step_penalty
        self._timeout_penalty     = timeout_penalty
        self._shaping_scale       = shaping_scale

        # Rendering
        self._render_mode = render_mode
        self._cell_px     = cell_px
        self._screen      = None
        self._clock       = None

        # Observation frame stack
        self._n_stack     = n_stack
        self._cell_obs_px = cell_obs_px
        self._obs_h       = _VIEW_SIZE * cell_obs_px
        self._obs_w       = _VIEW_SIZE * cell_obs_px
        self._frame_buf: deque = deque(maxlen=n_stack)

        # Build maze
        self._builder = MazeBuilder(
            width                          = width,
            height                         = height,
            use_procedural                 = use_procedural,
            seed                           = seed,
            use_random_seed                = use_random_seed,
            max_generation_attempts        = max_generation_attempts,
            obstacle_count                 = obstacle_count,
            min_patrol_distance            = min_patrol_distance,
            avoid_start_exit_for_obstacles = avoid_start_exit_for_obstacles,
            allow_fewer_obstacles          = allow_fewer_obstacles,
            min_start_exit_path_distance   = min_start_exit_path_distance,
            min_safe_reachable_cells       = min_safe_reachable_cells,
        )

        # Episode state (populated by reset)
        self._player_pos: Pos              = (0, 0)
        self._obstacles:  List[MovingObstacle] = []
        self._step_count: int              = 0
        self._is_done:    bool             = False

        # Initial maze build + episode reset
        self._obstacles = self._builder.build()
        self._reset_episode()

    # ------------------------------------------------------------------
    # Public RL interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        If regenerate_on_reset=True the maze layout is rebuilt (new seed or
        same fixed seed + attempt counter).  Otherwise only player and
        obstacle positions are reset.
        """
        if self._regenerate_on_reset:
            self._obstacles = self._builder.build()
        self._reset_episode()
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply *action* and advance one environment turn.

        Turn order (matching MazeGameManager.ExecuteTurn):
          1. Player tries to move.
          2. All obstacles step.
          3. Death check (same-cell OR crossing collision).
          4. Exit check.
          5. Timeout check.

        When done=True the environment auto-resets internally and the
        returned obs is the FIRST observation of the new episode.
        """
        self._step_count += 1
        reward = self._step_penalty

        prev_pos        = self._player_pos
        self._player_pos = self._try_move(prev_pos, MazeAction(action))
        curr_pos         = self._player_pos

        # Potential-based reward shaping: F(s,s') = phi(s') - phi(s)
        # phi(s) = -dist_to_exit(s): moving closer gives +scale, farther gives -scale, still gives 0.
        if self._shaping_scale > 0.0:
            phi_prev = -self._builder.dist_to_exit(prev_pos)
            phi_curr = -self._builder.dist_to_exit(curr_pos)
            reward  += self._shaping_scale * (phi_curr - phi_prev)

        # Obstacle steps happen after the player moves
        for obs in self._obstacles:
            obs.step()

        # 1. Death
        if self._check_death(prev_pos, curr_pos):
            return self._terminal(reward + self._death_penalty, MazeTerminalReason.DeathByHazard)

        # 2. Exit
        if self._builder.is_exit(self._player_pos):
            return self._terminal(reward + self._exit_reward, MazeTerminalReason.ExitReached)

        # 3. Timeout
        if self._max_episode_steps > 0 and self._step_count >= self._max_episode_steps:
            return self._terminal(reward + self._timeout_penalty, MazeTerminalReason.Timeout)

        return self._observe(), reward, False

    def render(self) -> None:
        """Draw the current state to a Pygame window (no-op if render_mode != "human")."""
        if self._render_mode != "human":
            return
        self._ensure_pygame()
        self._draw()

    def close(self) -> None:
        """Shut down Pygame if it was initialised."""
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # Context manager — matches UnityMazeEnv
    def __enter__(self) -> "MazeEnv":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def obs_size(self) -> tuple:
        """Observation shape: (n_stack + 1, H, W).  The +1 is the step-fraction channel."""
        return (self._n_stack + 1, self._obs_h, self._obs_w)

    # ------------------------------------------------------------------
    # Episode internals
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._player_pos = self._builder.start_cell
        self._step_count = 0
        self._is_done    = False
        for obs in self._obstacles:
            obs.reset()
        self._frame_buf.clear()

    def _try_move(self, pos: Pos, action: MazeAction) -> Pos:
        if action == MazeAction.NoOp:
            return pos
        dx, dy  = _DELTAS[action]
        target   = (pos[0] + dx, pos[1] + dy)
        return target if self._builder.is_walkable(target) else pos

    def _check_death(self, player_prev: Pos, player_curr: Pos) -> bool:
        for obs in self._obstacles:
            # Standard overlap: both end up on the same cell
            if obs.current_pos == player_curr:
                return True
            # Crossing collision: player and obstacle swap cells in one turn
            if obs.prev_pos == player_curr and obs.current_pos == player_prev:
                return True
        return False

    def _terminal(
        self, reward: float, reason: MazeTerminalReason
    ) -> Tuple[np.ndarray, float, bool]:
        """Finalise the episode, auto-reset, and return the new start obs."""
        if self._regenerate_on_reset:
            self._obstacles = self._builder.build()
        self._reset_episode()
        return self._observe(), reward, True

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """
        Render the egocentric 5×5 view, stack n_stack frames, then append a
        5th channel encoding the remaining step fraction (uniform value across
        all pixels).  This breaks perceptual aliasing between identical-looking
        locations visited at different points in the episode.

        Returns (n_stack + 1, H, W) float32 array in [0, 1].
        """
        frame = self._render_frame()
        self._frame_buf.append(frame)
        frames = list(self._frame_buf)
        while len(frames) < self._n_stack:
            frames.insert(0, frames[0])
        stacked = np.stack(frames, axis=0)   # (n_stack, H, W)

        # Step-fraction channel: 1.0 at episode start → 0.0 at timeout
        if self._max_episode_steps > 0:
            step_frac = 1.0 - self._step_count / self._max_episode_steps
        else:
            step_frac = 1.0
        step_ch = np.full((1, self._obs_h, self._obs_w), step_frac, dtype=np.float32)

        return np.concatenate([stacked, step_ch], axis=0)   # (n_stack+1, H, W)

    def _render_frame(self) -> np.ndarray:
        """
        Render the 5×5 egocentric view as a (H, W) float32 greyscale image.
        Cells hidden behind walls are painted black (0.0).
        """
        px_size = self._cell_obs_px
        frame   = np.zeros((self._obs_h, self._obs_w), dtype=np.float32)

        player_pos     = self._player_pos
        visible        = self._compute_visible_cells(player_pos)
        obstacle_cells = {o.current_pos for o in self._obstacles}
        exit_pos       = self._builder.exit_cell

        for row, dy in enumerate(range(_HALF_VIEW, -_HALF_VIEW - 1, -1)):
            for col, dx in enumerate(range(-_HALF_VIEW, _HALF_VIEW + 1)):
                abs_pos = (player_pos[0] + dx, player_pos[1] + dy)
                y0, y1  = row * px_size, (row + 1) * px_size
                x0, x1  = col * px_size, (col + 1) * px_size

                # Player's own cell — always visible
                if abs_pos == player_pos:
                    frame[y0:y1, x0:x1] = _PX_PLAYER
                    continue

                if abs_pos not in visible:
                    frame[y0:y1, x0:x1] = _PX_OCCLUDED
                    continue

                bx, by = abs_pos
                if (bx < 0 or bx >= self._builder.width or
                        by < 0 or by >= self._builder.height):
                    frame[y0:y1, x0:x1] = _PX_WALL
                elif not self._builder.is_walkable(abs_pos):
                    frame[y0:y1, x0:x1] = _PX_WALL
                elif abs_pos in obstacle_cells:
                    frame[y0:y1, x0:x1] = _PX_OBSTACLE
                elif abs_pos == exit_pos:
                    frame[y0:y1, x0:x1] = _PX_EXIT
                else:
                    frame[y0:y1, x0:x1] = _PX_FLOOR

        return frame

    def _compute_visible_cells(self, player_pos: Pos) -> set:
        """
        Return the set of absolute grid positions visible from *player_pos*
        within the 5×5 egocentric view, applying wall-based LOS occlusion.
        """
        visible = set()
        px, py  = player_pos
        for dy in range(_HALF_VIEW, -_HALF_VIEW - 1, -1):
            for dx in range(-_HALF_VIEW, _HALF_VIEW + 1):
                target = (px + dx, py + dy)
                if self._is_visible(player_pos, target):
                    visible.add(target)
        return visible

    def _is_visible(self, player_pos: Pos, target_pos: Pos) -> bool:
        """
        LOS check via Bresenham's line.  A target cell is visible if no wall
        (or OOB cell) lies strictly between the player and the target.
        """
        if player_pos == target_pos:
            return True
        px, py = player_pos
        tx, ty = target_pos
        for cx, cy in _bresenham_between(px, py, tx, ty):
            if (cx < 0 or cx >= self._builder.width or
                    cy < 0 or cy >= self._builder.height):
                return False
            if not self._builder.is_walkable((cx, cy)):
                return False
        return True

    # ------------------------------------------------------------------
    # Pygame rendering
    # ------------------------------------------------------------------

    def _ensure_pygame(self) -> None:
        if self._screen is not None:
            return
        import pygame
        pygame.init()
        w = self._builder.width  * self._cell_px
        h = self._builder.height * self._cell_px
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Maze DQN")
        self._clock = pygame.time.Clock()

    def _draw(self) -> None:
        import pygame

        screen   = self._screen
        px_size  = self._cell_px
        grid_h   = self._builder.height

        screen.fill(_COLOR_WALL)

        # Draw tiles
        for (x, y), cell_type in self._builder.cells.items():
            # y=0 is bottom in our grid → top of screen is (height-1)
            screen_y = (grid_h - 1 - y) * px_size
            screen_x = x * px_size
            rect     = pygame.Rect(screen_x, screen_y, px_size, px_size)

            if cell_type == MazeCellType.Wall:
                color = _COLOR_WALL
            elif cell_type == MazeCellType.Start:
                color = _COLOR_START
            elif cell_type == MazeCellType.Exit:
                color = _COLOR_EXIT
            else:
                color = _COLOR_FLOOR

            pygame.draw.rect(screen, color, rect)

        # Draw obstacles (inset by 2 px so grid lines are visible)
        for obs in self._obstacles:
            ox, oy   = obs.current_pos
            screen_y = (grid_h - 1 - oy) * px_size
            screen_x = ox * px_size
            rect     = pygame.Rect(screen_x + 2, screen_y + 2, px_size - 4, px_size - 4)
            pygame.draw.rect(screen, _COLOR_OBSTACLE, rect)

        # Draw player (inset by 4 px)
        px, py   = self._player_pos
        screen_y = (grid_h - 1 - py) * px_size
        screen_x = px * px_size
        rect     = pygame.Rect(screen_x + 4, screen_y + 4, px_size - 8, px_size - 8)
        pygame.draw.rect(screen, _COLOR_PLAYER, rect)

        pygame.display.flip()
        self._clock.tick(30)

        # Only consume QUIT events so that callers with their own event loops
        # (e.g. play.py) still receive keyboard events from the queue.
        if pygame.event.peek(pygame.QUIT):
            pygame.event.clear(pygame.QUIT)
            self.close()
