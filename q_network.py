"""
Q-Network for the maze DQN baseline.

Architecture: 3-layer MLP.
- Deep enough to learn maze navigation from a local grid view.
- Shallow enough to converge fast on CPU for early debugging runs.
- No CNN for the first baseline: the 5x5 grid is only 25 cells,
  so the spatial structure does not justify the complexity of convolutions.
  A CNN upgrade is straightforward later (reshape obs to [1, 5, 5] + 1 extra).

Input:  26 floats
  - 25 floats: 5x5 egocentric local grid, row-major north-to-south/west-to-east
      0.00 = floor   0.50 = exit   0.75 = obstacle   1.00 = wall/OOB
  - 1 float:  remaining step fraction (1.0 = episode start, 0.0 = timeout)

Output: 5 Q-values, one per discrete action
  0 = NoOp  |  1 = Up  |  2 = Left  |  3 = Down  |  4 = Right
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_size: int = 26, n_actions: int = 5, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch, obs_size)
        Returns:
            Q-values of shape (batch, n_actions)
        """
        return self.net(x)
