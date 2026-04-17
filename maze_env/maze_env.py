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

from typing import List, Optional, Tuple

import numpy as np

from .definitions import MazeAction, MazeCellType, MazeTerminalReason
from .maze_builder import MazeBuilder
from .moving_obstacle import MovingObstacle

Pos = Tuple[int, int]

# ------------------------------------------------------------------
# Observation constants (match MazeAgent.cs)
# ------------------------------------------------------------------
_HALF_VIEW      = 2
_VIEW_SIZE      = 2 * _HALF_VIEW + 1          # 5
_GRID_OBS_COUNT = _VIEW_SIZE * _VIEW_SIZE      # 25
_OBS_SIZE       = _GRID_OBS_COUNT + 1          # 26

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

    obs_size  = _OBS_SIZE
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

        # Rendering
        self._render_mode = render_mode
        self._cell_px     = cell_px
        self._screen      = None
        self._clock       = None

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

    # ------------------------------------------------------------------
    # Episode internals
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._player_pos = self._builder.start_cell
        self._step_count = 0
        self._is_done    = False
        for obs in self._obstacles:
            obs.reset()

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
        Build the 26-float observation vector matching MazeAgent.CollectObservations.

        Layout:
          [0–24]  5×5 egocentric local grid, row-major north-to-south then west-to-east.
                  Centre cell (index 12) is always the player's position.
          [25]    Remaining step fraction = 1 − step_count / max_episode_steps.
        """
        obs  = np.zeros(_OBS_SIZE, dtype=np.float32)
        px, py   = self._player_pos
        ex, ey   = self._builder.exit_cell
        exit_pos = (ex, ey)

        # Build obstacle position set once for O(1) look-ups in the inner loop
        obstacle_cells = {o.current_pos for o in self._obstacles}

        idx = 0
        for dy in range(_HALF_VIEW, -_HALF_VIEW - 1, -1):   # +2 … -2
            for dx in range(-_HALF_VIEW, _HALF_VIEW + 1):   # -2 … +2
                obs[idx] = self._encode_cell(
                    (px + dx, py + dy), exit_pos, obstacle_cells
                )
                idx += 1

        # [25] Remaining step fraction
        if self._max_episode_steps > 0:
            obs[25] = 1.0 - self._step_count / self._max_episode_steps
        else:
            obs[25] = 1.0

        return obs

    def _encode_cell(
        self,
        cell:            Pos,
        exit_pos:        Pos,
        obstacle_cells:  set,
    ) -> float:
        """
        Single-float cell encoding with priority:
          wall / OOB > obstacle > exit > floor
        """
        x, y = cell
        if x < 0 or x >= self._builder.width or y < 0 or y >= self._builder.height:
            return 1.0   # out-of-bounds treated as wall

        if not self._builder.is_walkable(cell):
            return 1.0   # wall

        if cell in obstacle_cells:
            return 0.75  # moving hazard

        if cell == exit_pos:
            return 0.5   # exit goal

        return 0.0       # floor (includes player's own cell at index 12)

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

        # Allow the window to be closed without crashing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
