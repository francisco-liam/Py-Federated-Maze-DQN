"""
Maze generation and layout management — Python port of MazeBuilder.cs.

Responsibilities:
  - Procedural maze generation (iterative DFS / recursive backtracker).
  - Fallback to a hardcoded layout when all generation attempts fail.
  - Start / exit assignment via double-BFS diameter approximation.
  - Full connectivity validation (BFS flood-fill).
  - Obstacle patrol-segment discovery and greedy placement.
  - Obstacle-aware solvability validation (treats patrol cells as blocked).
  - Seed-driven deterministic generation with optional random-seed mode.

Coordinate system
-----------------
Grid cells are addressed as (x, y) tuples where x grows right and y grows
up, matching Unity World-space convention.  The fallback layout string array
is stored top-row-first (row 0 = highest y), matching BuildGridFromLayout.cs.
"""

import time
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .definitions import MazeCellType
from .moving_obstacle import MovingObstacle

Pos  = Tuple[int, int]
Grid = Dict[Pos, MazeCellType]

# Up, Right, Down, Left — matches CardinalDirections in MazeBuilder.cs
_CARDINALS: List[Pos] = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Fallback layout used when all procedural attempts fail.
# '#' = Wall, '.' = Floor, 'S' = Start, 'E' = Exit.
# Row 0 is the top row (highest y value after the y-flip in _grid_from_layout).
_FALLBACK_LAYOUT: List[str] = [
    "###############",
    "#S....#.......#",
    "#.###.#.#####.#",
    "#...#.#.....#.#",
    "###.#.#####.#.#",
    "#...#.....#.#.#",
    "#.#######.#.#.#",
    "#.......#.#...#",
    "#.#####.#.###.#",
    "#.....#.....E##",
    "###############",
]

_CHAR_MAP: Dict[str, MazeCellType] = {
    '#': MazeCellType.Wall,
    '.': MazeCellType.Floor,
    'S': MazeCellType.Start,
    'E': MazeCellType.Exit,
}


class MazeBuilder:
    """
    Generates and stores the logical maze grid.

    After build() returns, the following attributes are valid:
      cells       : Dict[Pos, MazeCellType] — full grid
      start_cell  : Pos
      exit_cell   : Pos
      width       : int
      height      : int
      last_used_seed : int
    """

    def __init__(
        self,
        width:                          int  = 21,
        height:                         int  = 21,
        use_procedural:                 bool = True,
        seed:                           int  = 12345,
        use_random_seed:                bool = False,
        max_generation_attempts:        int  = 30,
        obstacle_count:                 int  = 3,
        min_patrol_distance:            int  = 2,
        avoid_start_exit_for_obstacles: bool = True,
        allow_fewer_obstacles:          bool = False,
        min_start_exit_path_distance:   int  = 8,
        min_safe_reachable_cells:       int  = 8,
    ) -> None:
        self._width                 = width
        self._height                = height
        self._use_procedural        = use_procedural
        self._fixed_seed            = seed
        self._use_random_seed       = use_random_seed
        self._max_attempts          = max_generation_attempts
        self._obstacle_count        = obstacle_count
        self._min_patrol_dist       = min_patrol_distance
        self._avoid_start_exit      = avoid_start_exit_for_obstacles
        self._allow_fewer_obstacles = allow_fewer_obstacles
        self._min_se_path_dist      = min_start_exit_path_distance
        self._min_safe_cells        = min_safe_reachable_cells

        # Populated by build()
        self.cells:          Grid = {}
        self.start_cell:     Pos  = (0, 0)
        self.exit_cell:      Pos  = (0, 0)
        self.width:          int  = 0
        self.height:         int  = 0
        self.last_used_seed: int  = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> List[MovingObstacle]:
        """
        (Re)build the maze.  Returns a fresh list of MovingObstacle objects
        whose initial positions are already set — call obstacle.reset() to
        return to those positions at each episode reset.
        """
        self.cells = {}

        ok, grid, obs_defs = self._try_generate()
        if not ok:
            print("[MazeGen] All generation attempts failed — using fallback layout.")
            grid    = _grid_from_layout(_FALLBACK_LAYOUT)
            obs_defs = []

        self._commit_grid(grid)
        return [MovingObstacle(a, b) for a, b in obs_defs]

    def is_walkable(self, pos: Pos) -> bool:
        cell = self.cells.get(pos)
        return cell is not None and cell != MazeCellType.Wall

    def is_exit(self, pos: Pos) -> bool:
        return pos == self.exit_cell

    # ------------------------------------------------------------------
    # Top-level generation loop
    # ------------------------------------------------------------------

    def _try_generate(self) -> Tuple[bool, Optional[Grid], List[Tuple[Pos, Pos]]]:
        if not self._use_procedural:
            return self._validate_fixed_layout(_grid_from_layout(_FALLBACK_LAYOUT))

        attempts = max(1, self._max_attempts)
        for attempt in range(attempts):
            seed = self._seed_for_attempt(attempt)
            rng  = random.Random(seed)
            self.last_used_seed = seed

            grid = self._gen_procedural(rng)

            ok, start, exit_, reason = self._assign_start_exit(grid, rng)
            if not ok:
                print(f"[MazeGen] Attempt {attempt+1}/{attempts} REJECT (start/exit): {reason}")
                continue

            if not self._check_connectivity(grid, start, exit_, attempt, attempts):
                continue

            ok2, obs_defs, reason2 = self._place_obstacles(grid, start, exit_, rng)
            if not ok2:
                print(f"[MazeGen] Attempt {attempt+1}/{attempts} REJECT (obstacles): {reason2}")
                continue

            ok3, reason3 = self._check_obstacle_solvability(grid, start, exit_, obs_defs)
            if not ok3:
                print(f"[MazeGen] Attempt {attempt+1}/{attempts} REJECT (solvability): {reason3}")
                continue

            print(
                f"[MazeGen] Attempt {attempt+1}/{attempts} PASS — "
                f"obstacles={len(obs_defs)}/{self._obstacle_count}  seed={seed}"
            )
            return True, grid, obs_defs

        return False, None, []

    def _validate_fixed_layout(
        self, grid: Grid
    ) -> Tuple[bool, Optional[Grid], List[Tuple[Pos, Pos]]]:
        ok, start, exit_, reason = self._find_start_exit(grid)
        if not ok:
            print(f"[MazeGen] Fixed layout invalid: {reason}")
            return False, None, []

        rng = random.Random(self._fixed_seed)
        ok2, obs_defs, reason2 = self._place_obstacles(grid, start, exit_, rng)
        if not ok2:
            return False, None, []

        ok3, _ = self._check_obstacle_solvability(grid, start, exit_, obs_defs)
        return ok3, grid, obs_defs

    # ------------------------------------------------------------------
    # Procedural DFS generation
    # ------------------------------------------------------------------

    def _gen_procedural(self, rng: random.Random) -> Grid:
        w = _normalize_dim(self._width)
        h = _normalize_dim(self._height)

        grid: Grid = {(x, y): MazeCellType.Wall for x in range(w) for y in range(h)}

        origin: Pos = (1, 1)
        grid[origin] = MazeCellType.Floor
        stack: List[Pos] = [origin]

        while stack:
            current = stack[-1]
            neighbors = self._unvisited_two_step_neighbors(current, grid, w, h)
            if not neighbors:
                stack.pop()
                continue

            nxt     = rng.choice(neighbors)
            between: Pos = ((current[0] + nxt[0]) // 2, (current[1] + nxt[1]) // 2)
            grid[between] = MazeCellType.Floor
            grid[nxt]     = MazeCellType.Floor
            stack.append(nxt)

        return grid

    @staticmethod
    def _unvisited_two_step_neighbors(
        cell: Pos, grid: Grid, w: int, h: int
    ) -> List[Pos]:
        result: List[Pos] = []
        for dx, dy in _CARDINALS:
            cx = cell[0] + dx * 2
            cy = cell[1] + dy * 2
            if 0 <= cx < w and 0 <= cy < h and grid.get((cx, cy)) == MazeCellType.Wall:
                result.append((cx, cy))
        return result

    # ------------------------------------------------------------------
    # Start / exit assignment (double-BFS diameter)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_start_exit(
        grid: Grid,
    ) -> Tuple[bool, Optional[Pos], Optional[Pos], str]:
        start = exit_ = None
        for pos, cell in grid.items():
            if cell == MazeCellType.Start:
                start = pos
            elif cell == MazeCellType.Exit:
                exit_ = pos
        if start is None or exit_ is None:
            return False, None, None, "Layout must contain S and E."
        return True, start, exit_, ""

    def _assign_start_exit(
        self, grid: Grid, rng: random.Random
    ) -> Tuple[bool, Optional[Pos], Optional[Pos], str]:
        walkable = [p for p, t in grid.items() if t != MazeCellType.Wall]
        if len(walkable) < 2:
            return False, None, None, "Need at least two walkable cells."

        anchor    = rng.choice(walkable)
        farthest_a = self._farthest_reachable(grid, anchor, blocked=None)
        farthest_b = self._farthest_reachable(grid, farthest_a, blocked=None)
        start, exit_ = farthest_a, farthest_b

        dist_map = _bfs_distances(grid, start, blocked=None)
        if exit_ not in dist_map:
            return False, None, None, "Start cannot reach exit."

        path_dist = dist_map[exit_]
        req = max(0, self._min_se_path_dist)
        if req > 0 and len(walkable) > req + 1 and path_dist < req:
            return (
                False, None, None,
                f"Path distance {path_dist} < required {req}."
            )

        grid[start] = MazeCellType.Start
        grid[exit_] = MazeCellType.Exit
        return True, start, exit_, ""

    def _farthest_reachable(
        self, grid: Grid, origin: Pos, blocked: Optional[Set[Pos]]
    ) -> Pos:
        dists = _bfs_distances(grid, origin, blocked)
        return max(dists, key=lambda p: dists[p]) if dists else origin

    # ------------------------------------------------------------------
    # Connectivity validation
    # ------------------------------------------------------------------

    def _check_connectivity(
        self,
        grid: Grid,
        start: Pos,
        exit_: Pos,
        attempt: int,
        total: int,
    ) -> bool:
        reachable     = _flood_fill(grid, start, blocked=None)
        total_walkable = sum(1 for t in grid.values() if t != MazeCellType.Wall)

        if exit_ not in reachable:
            print(
                f"[MazeGen] Attempt {attempt+1}/{total} REJECT (connectivity): "
                "start cannot reach exit."
            )
            return False

        if len(reachable) != total_walkable:
            print(
                f"[MazeGen] Attempt {attempt+1}/{total} REJECT (connectivity): "
                f"disconnected — {len(reachable)} reachable vs {total_walkable} walkable."
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def _place_obstacles(
        self, grid: Grid, start: Pos, exit_: Pos, rng: random.Random
    ) -> Tuple[bool, List[Tuple[Pos, Pos]], str]:
        defs: List[Tuple[Pos, Pos]] = []
        if self._obstacle_count <= 0:
            return True, defs, ""

        segments = self._build_patrol_segments(grid)
        if not segments:
            msg = "No valid patrol segments found."
            return (True, defs, msg) if self._allow_fewer_obstacles else (False, defs, msg)

        rng.shuffle(segments)
        occupied: Set[Pos] = set()

        for seg_a, seg_b, _ in segments:
            if len(defs) >= self._obstacle_count:
                break

            if self._avoid_start_exit and (
                seg_a in (start, exit_) or seg_b in (start, exit_)
            ):
                continue

            if seg_a in occupied or seg_b in occupied:
                continue

            defs.append((seg_a, seg_b))
            ok, _ = self._check_obstacle_solvability(grid, start, exit_, defs)
            if not ok:
                defs.pop()
                continue

            occupied.add(seg_a)
            occupied.add(seg_b)

        if len(defs) >= self._obstacle_count:
            return True, defs, ""

        if self._allow_fewer_obstacles:
            return True, defs, f"Placed {len(defs)}/{self._obstacle_count}."

        return False, defs, f"Placed {len(defs)}/{self._obstacle_count}, full count required."

    def _check_obstacle_solvability(
        self,
        grid: Grid,
        start: Pos,
        exit_: Pos,
        obs_defs: List[Tuple[Pos, Pos]],
    ) -> Tuple[bool, str]:
        blocked = _danger_cells(obs_defs)

        if start in blocked:
            return False, "Start cell overlaps obstacle patrol."
        if exit_ in blocked:
            return False, "Exit cell overlaps obstacle patrol."

        safe_walkable = sum(
            1 for p, t in grid.items()
            if t != MazeCellType.Wall and p not in blocked
        )
        min_safe = max(2, self._min_safe_cells)
        if safe_walkable < min_safe:
            return False, f"Too few safe cells: {safe_walkable} < {min_safe}."

        safe_reachable = _flood_fill(grid, start, blocked=blocked)
        if exit_ not in safe_reachable:
            return False, "Cannot reach exit when obstacle patrols are treated as blocked."

        if len(safe_reachable) != safe_walkable:
            return (
                False,
                f"Safe space disconnected: {len(safe_reachable)} reachable vs {safe_walkable} safe walkable.",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Patrol segment discovery
    # ------------------------------------------------------------------

    def _build_patrol_segments(self, grid: Grid) -> List[Tuple[Pos, Pos, int]]:
        """
        Return list of (point_a, point_b, manhattan_distance) for all valid
        horizontal and vertical straight-line patrol corridors.

        Sorted: distance descending, then by pointA.y, pointA.x, pointB.y,
        pointB.x ascending (matches C# comparer).  The sort order is stable
        for deterministic obstacle placement.
        """
        segments: List[Tuple[Pos, Pos, int]] = []

        for pos, cell in grid.items():
            if cell == MazeCellType.Wall:
                continue
            for direction in ((1, 0), (0, 1)):   # right, up
                end  = _farthest_in_direction(pos, direction, grid)
                dist = abs(end[0] - pos[0]) + abs(end[1] - pos[1])
                if dist >= self._min_patrol_dist:
                    segments.append((pos, end, dist))

        segments.sort(
            key=lambda s: (-s[2], s[0][1], s[0][0], s[1][1], s[1][0])
        )
        return segments

    # ------------------------------------------------------------------
    # Grid commit
    # ------------------------------------------------------------------

    def _commit_grid(self, grid: Grid) -> None:
        self.cells = dict(grid)
        xs = [p[0] for p in grid]
        ys = [p[1] for p in grid]
        self.width  = max(xs) + 1 if xs else 0
        self.height = max(ys) + 1 if ys else 0

        self.start_cell = (0, 0)
        self.exit_cell  = (0, 0)
        for pos, cell in grid.items():
            if cell == MazeCellType.Start:
                self.start_cell = pos
            elif cell == MazeCellType.Exit:
                self.exit_cell = pos

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _seed_for_attempt(self, attempt: int) -> int:
        if self._use_random_seed:
            return int(time.time() * 1000) + attempt * 7919
        return self._fixed_seed + attempt


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no self)
# ---------------------------------------------------------------------------

def _normalize_dim(v: int) -> int:
    """Force dimension to be odd and >= 5, matching NormalizeMazeDimension."""
    v = max(5, v)
    return v if v % 2 == 1 else v + 1


def _bfs_distances(
    grid: Grid, start: Pos, blocked: Optional[Set[Pos]]
) -> Dict[Pos, int]:
    """BFS from *start*, returning {pos: distance} for all reachable walkable cells."""
    if grid.get(start) == MazeCellType.Wall:
        return {}
    if blocked and start in blocked:
        return {}

    dist: Dict[Pos, int] = {start: 0}
    queue: deque[Pos] = deque([start])

    while queue:
        current = queue.popleft()
        for dx, dy in _CARDINALS:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in dist:
                continue
            if grid.get(nxt, MazeCellType.Wall) == MazeCellType.Wall:
                continue
            if blocked and nxt in blocked:
                continue
            dist[nxt] = dist[current] + 1
            queue.append(nxt)

    return dist


def _flood_fill(
    grid: Grid, start: Pos, blocked: Optional[Set[Pos]]
) -> Set[Pos]:
    return set(_bfs_distances(grid, start, blocked).keys())


def _danger_cells(obs_defs: List[Tuple[Pos, Pos]]) -> Set[Pos]:
    """Union of all cells covered by each patrol segment."""
    blocked: Set[Pos] = set()
    for a, b in obs_defs:
        blocked.update(_cells_along_segment(a, b))
    return blocked


def _cells_along_segment(a: Pos, b: Pos) -> List[Pos]:
    dx = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
    dy = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
    cells: List[Pos] = [a]
    current = a
    while current != b:
        current = (current[0] + dx, current[1] + dy)
        cells.append(current)
    return cells


def _farthest_in_direction(start: Pos, direction: Pos, grid: Grid) -> Pos:
    """Walk in *direction* until a wall or OOB, return last walkable cell."""
    current = start
    nxt     = (current[0] + direction[0], current[1] + direction[1])
    while grid.get(nxt, MazeCellType.Wall) != MazeCellType.Wall:
        current = nxt
        nxt     = (current[0] + direction[0], current[1] + direction[1])
    return current


def _grid_from_layout(layout: List[str]) -> Grid:
    """
    Convert a string-array layout to a Grid dict.
    Row 0 of the layout is the top row, mapped to the highest y value,
    matching the C# BuildGridFromLayout y-flip: y = height - 1 - row.
    """
    h = len(layout)
    grid: Grid = {}
    for row, line in enumerate(layout):
        y = h - 1 - row
        for col, ch in enumerate(line):
            grid[(col, y)] = _CHAR_MAP[ch]
    return grid
