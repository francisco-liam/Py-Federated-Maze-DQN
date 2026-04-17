"""
Shared enums and data types — Python port of MazeDefinitions.cs.
"""

from enum import IntEnum
from dataclasses import dataclass


class MazeCellType(IntEnum):
    Floor = 0
    Wall  = 1
    Start = 2
    Exit  = 3


class MazeAction(IntEnum):
    NoOp  = 0
    Up    = 1
    Left  = 2
    Down  = 3
    Right = 4


class MazeTerminalReason(IntEnum):
    NONE         = 0
    ExitReached  = 1
    DeathByHazard = 2
    Timeout      = 3


@dataclass
class MazeStepResult:
    reward:          float
    is_terminal:     bool
    terminal_reason: MazeTerminalReason

    @staticmethod
    def non_terminal(reward: float) -> "MazeStepResult":
        return MazeStepResult(reward, False, MazeTerminalReason.NONE)

    @staticmethod
    def terminal(reward: float, reason: MazeTerminalReason) -> "MazeStepResult":
        return MazeStepResult(reward, True, reason)
