"""
Moving hazard — Python port of MovingObstacle.cs (TwoPointPingPong mode).

The patrol logic is a direct translation of the C# Step() method:
  1. Record previous position.
  2. Get the current target (B or A).
  3. If already at the target, flip direction and get the new target.
  4. If still at the target (A == B edge case), do nothing.
  5. Otherwise move one cell toward the target (horizontal movement first,
     matching MoveOneCellToward).
"""

from typing import Tuple

Pos = Tuple[int, int]


def _step_toward(current: Pos, target: Pos) -> Pos:
    """Move one grid cell toward *target*, horizontal axis first."""
    dx = target[0] - current[0]
    if dx != 0:
        return (current[0] + (1 if dx > 0 else -1), current[1])
    dy = target[1] - current[1]
    if dy != 0:
        return (current[0], current[1] + (1 if dy > 0 else -1))
    return current


class MovingObstacle:
    """
    Deterministic ping-pong hazard. Patrols between point_a and point_b,
    moving one cell per Step() call. Initially targets point_b.
    """

    def __init__(self, point_a: Pos, point_b: Pos) -> None:
        self._a = point_a
        self._b = point_b
        self.current_pos: Pos = point_a
        self.prev_pos:    Pos = point_a
        self._targeting_b: bool = True   # matches GetInitialTargetIndex() == 1

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return obstacle to its initial position and direction."""
        self.current_pos  = self._a
        self.prev_pos     = self._a
        self._targeting_b = True

    def step(self) -> None:
        """Advance obstacle by one cell. Called once per environment turn."""
        self.prev_pos = self.current_pos
        target = self._b if self._targeting_b else self._a

        # At target → flip direction
        if self.current_pos == target:
            self._targeting_b = not self._targeting_b
            target = self._b if self._targeting_b else self._a

        # A == B edge case (or can't move further)
        if self.current_pos == target:
            return

        self.current_pos = _step_toward(self.current_pos, target)
